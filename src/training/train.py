"""
Training loop for the Thermal Pattern Analysis pipeline.

Supports:
    - AdamW optimiser with cosine annealing scheduler
    - Early stopping
    - TensorBoard logging
    - Checkpoint saving / resuming
    - Mixed-precision training (if GPU available)
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

from tqdm import tqdm
from pathlib import Path
from typing import Optional

from src.models.anomaly_detector import ThermalPatternPipeline
from src.training.losses import CombinedLoss
from src.evaluation.metrics import MetricsCalculator


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Full training manager for the ThermalPatternPipeline.
    """

    def __init__(
        self,
        model: ThermalPatternPipeline,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Loss
        self.criterion = CombinedLoss.from_config(config)

        # Classification head (simple linear head for binary)
        self.classifier = nn.Linear(
            config.model.feature_extractor.embedding_dim, 2
        ).to(device)

        # Optimiser: model params + classifier
        all_params = list(model.parameters()) + list(self.classifier.parameters())
        self.optimizer = AdamW(
            all_params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.epochs,
        )

        # Early stopping
        es_cfg = config.training.early_stopping
        self.early_stopping = EarlyStopping(
            patience=es_cfg.patience,
            min_delta=es_cfg.min_delta,
        )

        # Logging
        log_dir = config.paths.get("logs", "logs")
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
            print("  ⚠ TensorBoard not available — logging to console only")
        self.metrics = MetricsCalculator()

        # Checkpoint dir
        self.ckpt_dir = Path(config.paths.get("checkpoints", "checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Mixed-precision scaler
        self.scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    def train_epoch(self, epoch: int) -> dict:
        """Run one training epoch."""
        self.model.train()
        self.classifier.train()

        epoch_loss = 0.0
        all_preds, all_labels = [], []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        for sequences, labels in pbar:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward
            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    results = self.model(sequences)
                    logits = self.classifier(results["encoding"])
                    loss_dict = self.criterion(
                        results["encoding"], labels, logits
                    )
                    loss = loss_dict["total_loss"]

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                results = self.model(sequences)
                logits = self.classifier(results["encoding"])
                loss_dict = self.criterion(
                    results["encoding"], labels, logits
                )
                loss = loss_dict["total_loss"]

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Update baseline with normal samples
            normal_mask = labels == 0
            if normal_mask.any():
                self.model.anomaly_detector.update_baseline(
                    results["encoding"][normal_mask].detach()
                )

            # Track metrics
            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(len(self.train_loader), 1)
        metrics = self.metrics.compute_all(all_labels, all_preds)
        metrics["loss"] = avg_loss
        return metrics

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> dict:
        """Run one validation epoch."""
        self.model.eval()
        self.classifier.eval()

        epoch_loss = 0.0
        all_preds, all_labels, all_scores = [], [], []

        for sequences, labels in tqdm(
            self.val_loader, desc=f"Epoch {epoch+1} [Val]"
        ):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            results = self.model(sequences)
            logits = self.classifier(results["encoding"])
            loss_dict = self.criterion(results["encoding"], labels, logits)

            epoch_loss += loss_dict["total_loss"].item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(
                torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            )

        avg_loss = epoch_loss / max(len(self.val_loader), 1)
        metrics = self.metrics.compute_all(all_labels, all_preds, all_scores)
        metrics["loss"] = avg_loss
        return metrics

    def train(self) -> dict:
        """
        Full training loop with early stopping, checkpointing,
        and TensorBoard logging.

        Returns:
            Best validation metrics dict.
        """
        epochs = self.config.training.epochs
        best_val_loss = float("inf")
        best_metrics = {}

        print(f"\n{'='*60}")
        print(f"  Training — {epochs} epochs on {self.device}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            t0 = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)
            # Validate
            val_metrics = self.validate_epoch(epoch)
            # Step scheduler
            self.scheduler.step()

            elapsed = time.time() - t0

            # TensorBoard
            if self.writer is not None:
                for key, val in train_metrics.items():
                    self.writer.add_scalar(f"train/{key}", val, epoch)
                for key, val in val_metrics.items():
                    self.writer.add_scalar(f"val/{key}", val, epoch)
                self.writer.add_scalar(
                    "lr", self.optimizer.param_groups[0]["lr"], epoch
                )

            # Console summary
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train loss: {train_metrics['loss']:.4f} | "
                f"Val loss: {val_metrics['loss']:.4f} | "
                f"Val acc: {val_metrics.get('accuracy', 0):.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Checkpoint best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_metrics = val_metrics
                self._save_checkpoint(epoch, val_metrics, is_best=True)

            # Early stopping
            if self.early_stopping(val_metrics["loss"]):
                print(f"\n⏹  Early stopping at epoch {epoch+1}")
                break

        if self.writer is not None:
            self.writer.close()
        print(f"\n{'='*60}")
        print(f"  Training complete — Best val loss: {best_val_loss:.4f}")
        print(f"{'='*60}\n")

        return best_metrics

    def _save_checkpoint(
        self, epoch: int, metrics: dict, is_best: bool = False
    ):
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "classifier_state_dict": self.classifier.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
        }
        path = self.ckpt_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(state, path)

        if is_best:
            best_path = self.ckpt_dir / "best_model.pt"
            torch.save(state, best_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Resume training from a saved checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.classifier.load_state_dict(ckpt["classifier_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"✓ Resumed from epoch {ckpt['epoch'] + 1}")
        return ckpt["epoch"] + 1
