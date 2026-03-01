#!/usr/bin/env python3
"""
Data Preparation v3 — Realistic anomaly detection WITHIN Power Transformers.

Strategy:
  - Normal  = Original Power Transformer images (clean)
  - Abnormal = Same images WITH injected thermal anomalies:
      • Localised hotspots (overheating component)
      • Temperature gradient shifts (cooling failure)
      • Scattered micro-hotspots (insulation breakdown)
      • Intensity spikes (electrical arc / fault)

This creates a genuinely challenging binary classification task
with expected accuracy in the 85-95% range.

Usage:
    python prepare_data_v3.py
    python prepare_data_v3.py --seq-len 5 --difficulty medium
"""

import os
import sys
import shutil
import argparse
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────
# Anomaly injection functions
# ──────────────────────────────────────────────────────────────────────

def inject_hotspot(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Add a localised hotspot — simulates overheating component.
    A bright Gaussian blob at a random position.
    """
    h, w = img.shape[:2]
    result = img.astype(np.float64)

    # Random position (preferring centre region)
    cx = rng.randint(int(w * 0.2), int(w * 0.8))
    cy = rng.randint(int(h * 0.2), int(h * 0.8))
    radius = rng.randint(15, 45)
    intensity = rng.uniform(40, 90)

    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    hotspot = intensity * np.exp(-dist ** 2 / (2 * radius ** 2))

    result += hotspot
    return np.clip(result, 0, 255).astype(np.uint8)


def inject_gradient_anomaly(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Add asymmetric temperature gradient — simulates cooling failure.
    One side becomes significantly hotter.
    """
    h, w = img.shape[:2]
    result = img.astype(np.float64)

    direction = rng.choice(["left", "right", "top", "bottom"])
    strength = rng.uniform(30, 70)

    if direction == "left":
        gradient = np.tile(np.linspace(strength, 0, w), (h, 1))
    elif direction == "right":
        gradient = np.tile(np.linspace(0, strength, w), (h, 1))
    elif direction == "top":
        gradient = np.tile(np.linspace(strength, 0, h).reshape(-1, 1), (1, w))
    else:
        gradient = np.tile(np.linspace(0, strength, h).reshape(-1, 1), (1, w))

    result += gradient
    return np.clip(result, 0, 255).astype(np.uint8)


def inject_scattered_hotspots(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Add multiple small hotspots — simulates insulation breakdown.
    Several small bright spots scattered across the image.
    """
    h, w = img.shape[:2]
    result = img.astype(np.float64)

    n_spots = rng.randint(5, 15)
    for _ in range(n_spots):
        cx = rng.randint(int(w * 0.1), int(w * 0.9))
        cy = rng.randint(int(h * 0.1), int(h * 0.9))
        radius = rng.randint(5, 18)
        intensity = rng.uniform(35, 75)

        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        spot = intensity * np.exp(-dist ** 2 / (2 * radius ** 2))
        result += spot

    return np.clip(result, 0, 255).astype(np.uint8)


def inject_intensity_spike(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Brighten a rectangular region — simulates electrical arc / fault.
    A region of the image becomes uniformly hotter.
    """
    h, w = img.shape[:2]
    result = img.copy().astype(np.float64)

    # Random rectangle
    rx = rng.randint(int(w * 0.15), int(w * 0.5))
    ry = rng.randint(int(h * 0.15), int(h * 0.5))
    rw = rng.randint(int(w * 0.15), int(w * 0.4))
    rh = rng.randint(int(h * 0.15), int(h * 0.4))
    spike = rng.uniform(30, 65)

    # Smooth the edges with a Gaussian blur
    mask = np.zeros((h, w), dtype=np.float64)
    mask[ry:ry+rh, rx:rx+rw] = spike
    mask = cv2.GaussianBlur(mask, (21, 21), 10)

    result += mask
    return np.clip(result, 0, 255).astype(np.uint8)


def inject_anomaly(img: np.ndarray, rng: random.Random, difficulty: str = "medium") -> np.ndarray:
    """
    Randomly pick one or more anomaly types and apply them.

    Difficulty:
      - easy:   strong, obvious anomalies
      - medium: moderate anomalies (realistic)
      - hard:   subtle anomalies (challenging detection)
    """
    # Convert to grayscale if color
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    anomaly_funcs = [
        inject_hotspot,
        inject_gradient_anomaly,
        inject_scattered_hotspots,
        inject_intensity_spike,
    ]

    if difficulty == "easy":
        # Apply 2-3 anomalies (very obvious)
        n_anomalies = rng.randint(2, 3)
    elif difficulty == "hard":
        # Apply just 1 subtle anomaly
        n_anomalies = 1
    else:  # medium
        n_anomalies = rng.randint(1, 2)

    chosen = rng.sample(anomaly_funcs, min(n_anomalies, len(anomaly_funcs)))
    result = gray
    for func in chosen:
        result = func(result, rng)

    return result


# ──────────────────────────────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────────────────────────────

def find_images(directory: str) -> list:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    return sorted([
        os.path.join(directory, f) for f in os.listdir(directory)
        if Path(f).suffix.lower() in extensions
    ])


def build_sequences(image_paths: list, output_dir: Path, seq_length: int, seed: int):
    """Build sequences from image paths with overlapping windows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    paths = list(image_paths)
    rng.shuffle(paths)

    seq_count = 0

    # Non-overlapping
    for i in range(0, len(paths), seq_length):
        batch = paths[i:i + seq_length]
        if len(batch) < max(2, seq_length // 2):
            continue
        seq_count += 1
        seq_dir = output_dir / f"seq_{seq_count:03d}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        for j, src in enumerate(batch):
            dst = seq_dir / f"frame_{j+1:03d}.png"
            if isinstance(src, tuple):
                cv2.imwrite(str(dst), src[1])  # (path, image) tuple
            else:
                shutil.copy2(src, dst)

    # Overlapping (stride = 2)
    rng.shuffle(paths)
    stride = max(1, seq_length // 2)
    for i in range(0, len(paths) - seq_length, stride):
        batch = paths[i:i + seq_length]
        seq_count += 1
        seq_dir = output_dir / f"seq_{seq_count:03d}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        for j, src in enumerate(batch):
            dst = seq_dir / f"frame_{j+1:03d}.png"
            if isinstance(src, tuple):
                cv2.imwrite(str(dst), src[1])
            else:
                shutil.copy2(src, dst)

    return seq_count


def parse_args():
    p = argparse.ArgumentParser(description="Prepare data v3 — realistic anomaly injection")
    p.add_argument("--raw", type=str, default="data/raw/Power Transformers")
    p.add_argument("--output", type=str, default="data/sequences")
    p.add_argument("--seq-len", type=int, default=5)
    p.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard"])
    p.add_argument("--anomaly-ratio", type=float, default=1.0, help="Ratio of abnormal images to normal (1.0 = equal)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    raw = Path(args.raw)

    if not raw.exists():
        print(f"✗ Power Transformers folder not found: {raw}")
        print(f"  Expected: data/raw/Power Transformers/")
        sys.exit(1)

    # Clean output
    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Data Prep v3 — Realistic Anomaly Injection")
    print(f"  Difficulty: {args.difficulty}")
    print(f"{'='*60}")

    # Find all Power Transformer images
    all_images = find_images(str(raw))
    print(f"\n  Power Transformer images: {len(all_images)}")

    # Split: half for normal, half as base for abnormal injection
    rng = random.Random(args.seed)
    rng.shuffle(all_images)

    n_normal = len(all_images)  # ALL originals are normal
    n_abnormal = int(n_normal * args.anomaly_ratio)

    print(f"  Normal images: {n_normal} (original)")
    print(f"  Abnormal images: {n_abnormal} (anomaly-injected)")

    # Create abnormal images by injecting anomalies
    print(f"\n  Injecting thermal anomalies ({args.difficulty} difficulty)...")
    abnormal_dir = output / "_temp_abnormal"
    abnormal_dir.mkdir(parents=True, exist_ok=True)

    abnormal_sources = rng.choices(all_images, k=n_abnormal)  # sample with replacement
    abnormal_paths = []

    for i, src_path in enumerate(tqdm(abnormal_sources, desc="  Injecting")):
        img = cv2.imread(src_path)
        if img is None:
            continue

        injected = inject_anomaly(img, random.Random(args.seed + i), args.difficulty)
        out_path = abnormal_dir / f"abnormal_{i+1:04d}.png"
        cv2.imwrite(str(out_path), injected)
        abnormal_paths.append(str(out_path))

    print(f"  ✓ Created {len(abnormal_paths)} anomaly images")

    # Build sequences
    print(f"\n  Building sequences (length={args.seq_len})...")

    n_normal_seq = build_sequences(
        all_images, output / "normal", args.seq_len, args.seed
    )
    n_abnormal_seq = build_sequences(
        abnormal_paths, output / "abnormal", args.seq_len, args.seed + 1
    )

    # Clean temp
    shutil.rmtree(abnormal_dir)

    total = n_normal_seq + n_abnormal_seq
    train_n = int(total * 0.7)
    val_n = int(total * 0.15)
    test_n = total - train_n - val_n

    print(f"\n{'='*60}")
    print(f"  ✅  Data preparation complete!")
    print(f"")
    print(f"  Normal sequences   : {n_normal_seq}")
    print(f"  Abnormal sequences : {n_abnormal_seq}")
    print(f"  Total sequences    : {total}")
    print(f"  Est. split         : ~{train_n} train / ~{val_n} val / ~{test_n} test")
    print(f"  Difficulty         : {args.difficulty}")
    print(f"  Output             : {output}/")
    print(f"{'='*60}")
    print(f"\n  Next step:  python train.py")


if __name__ == "__main__":
    main()
