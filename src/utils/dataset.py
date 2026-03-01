"""
PyTorch Dataset for thermal image sequences.

Organises thermal images into fixed-length sequences for
training the CNN + LSTM pipeline.  Supports train / val / test splits
and optional augmentation.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional

from src.preprocessing import ThermalImageProcessor, ThermalAugmentor


class ThermalSequenceDataset(Dataset):
    """
    PyTorch Dataset that yields (sequence_tensor, label) pairs.

    Directory layout expected:
        data/sequences/
            normal/
                seq_001/
                    img_001.png
                    img_002.png
                    ...
            abnormal/
                seq_010/
                    ...

    Each sequence folder contains chronologically ordered thermal images.
    """

    LABEL_MAP = {"normal": 0, "abnormal": 1}

    def __init__(
        self,
        sequences_dir: str,
        processor: ThermalImageProcessor,
        augmentor: Optional[ThermalAugmentor] = None,
        sequence_length: int = 20,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        """
        Args:
            sequences_dir: Root directory containing label sub-folders.
            processor:     ThermalImageProcessor instance.
            augmentor:     Optional ThermalAugmentor (applied only in train).
            sequence_length: Fixed number of images per sequence.
            split:         One of 'train', 'val', 'test'.
            train_ratio:   Fraction of data for training.
            val_ratio:     Fraction of data for validation.
            seed:          Random seed for reproducible splits.
        """
        super().__init__()
        self.processor = processor
        self.augmentor = augmentor if split == "train" else None
        self.sequence_length = sequence_length

        # Discover sequences
        all_sequences = self._discover_sequences(sequences_dir)
        random.seed(seed)
        random.shuffle(all_sequences)

        # Split
        n = len(all_sequences)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if split == "train":
            self.sequences = all_sequences[:n_train]
        elif split == "val":
            self.sequences = all_sequences[n_train : n_train + n_val]
        else:
            self.sequences = all_sequences[n_train + n_val :]

    def _discover_sequences(self, root: str) -> List[Tuple[str, int]]:
        """
        Walk the directory tree and return a list of
        (sequence_folder_path, label) tuples.
        """
        sequences = []
        root = Path(root)

        for label_name, label_id in self.LABEL_MAP.items():
            label_dir = root / label_name
            if not label_dir.exists():
                continue

            for seq_dir in sorted(label_dir.iterdir()):
                if seq_dir.is_dir():
                    img_files = sorted([
                        str(f) for f in seq_dir.iterdir()
                        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
                    ])
                    if len(img_files) >= 1:
                        sequences.append((img_files, label_id))

        return sequences

    def _pad_or_truncate(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Ensure the sequence has exactly *sequence_length* frames."""
        if len(images) >= self.sequence_length:
            return images[: self.sequence_length]

        # Pad by repeating the last frame
        while len(images) < self.sequence_length:
            images.append(images[-1].copy())
        return images

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_paths, label = self.sequences[idx]

        # Load and preprocess
        images = [self.processor.process(p) for p in img_paths]
        images = self._pad_or_truncate(images)

        # Augment (train only)
        if self.augmentor is not None:
            images = self.augmentor.augment_sequence(images)

        # Stack → (T, 1, H, W)
        tensors = [
            torch.from_numpy(img).unsqueeze(0) for img in images
        ]
        sequence_tensor = torch.stack(tensors, dim=0)  # (T, 1, H, W)

        return sequence_tensor, label


def create_dataloaders(
    config,
    processor: ThermalImageProcessor,
    augmentor: Optional[ThermalAugmentor] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from config.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    seq_dir = config.data.sequences_dir
    seq_len = config.data.sequence_length
    batch = config.training.batch_size
    workers = config.data.get("num_workers", 0)

    datasets = {}
    for split in ("train", "val", "test"):
        datasets[split] = ThermalSequenceDataset(
            sequences_dir=seq_dir,
            processor=processor,
            augmentor=augmentor if split == "train" else None,
            sequence_length=seq_len,
            split=split,
            train_ratio=config.data.train_split,
            val_ratio=config.data.val_split,
            seed=config.get("seed", 42),
        )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
