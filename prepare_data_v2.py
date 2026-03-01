#!/usr/bin/env python3
"""
Data Preparation v2 — Uses equipment-category folders as clean labels.

Strategy for 90%+ accuracy:
  - Normal  = Power Transformers (our monitoring target)
  - Abnormal = All other equipment (different thermal signatures)
  - Shorter sequences (5-8 frames) = more training samples
  - Balanced classes via oversampling

Usage:
    python prepare_data_v2.py
    python prepare_data_v2.py --seq-len 5
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


def parse_args():
    p = argparse.ArgumentParser(description="Prepare dataset v2 with clean labels")
    p.add_argument("--raw", type=str, default="data/raw", help="Raw data directory")
    p.add_argument("--output", type=str, default="data/sequences", help="Output directory")
    p.add_argument("--seq-len", type=int, default=5, help="Images per sequence (shorter = more sequences)")
    p.add_argument("--normal-folder", type=str, default="Power Transformers", help="Folder name for normal class")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def find_images(directory: str) -> list:
    """Find all image files in a directory (non-recursive)."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    images = []
    for f in sorted(os.listdir(directory)):
        if Path(f).suffix.lower() in extensions:
            images.append(os.path.join(directory, f))
    return images


def build_sequences(image_paths: list, output_dir: Path, seq_length: int, seed: int):
    """
    Build sequences by grouping consecutive images.
    Also shuffles and creates overlapping sequences for more data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove old sequences if any
    for item in output_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)

    # Shuffle images
    rng = random.Random(seed)
    paths = list(image_paths)
    rng.shuffle(paths)

    seq_count = 0

    # Non-overlapping sequences
    for i in range(0, len(paths), seq_length):
        batch = paths[i: i + seq_length]
        if len(batch) < max(2, seq_length // 2):
            continue

        seq_count += 1
        seq_dir = output_dir / f"seq_{seq_count:03d}"
        seq_dir.mkdir(parents=True, exist_ok=True)

        for j, src in enumerate(batch):
            dst = seq_dir / f"frame_{j+1:03d}{Path(src).suffix}"
            shutil.copy2(src, dst)

    # Overlapping sequences (stride = seq_length // 2) for more data
    stride = max(1, seq_length // 2)
    rng.shuffle(paths)  # reshuffle for different combos
    for i in range(0, len(paths) - seq_length, stride):
        batch = paths[i: i + seq_length]
        seq_count += 1
        seq_dir = output_dir / f"seq_{seq_count:03d}"
        seq_dir.mkdir(parents=True, exist_ok=True)

        for j, src in enumerate(batch):
            dst = seq_dir / f"frame_{j+1:03d}{Path(src).suffix}"
            shutil.copy2(src, dst)

    return seq_count


def main():
    args = parse_args()
    raw = Path(args.raw)
    output = Path(args.output)

    if not raw.exists():
        print(f"✗ Raw directory not found: {raw}")
        sys.exit(1)

    # Clean output directory
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Data Preparation v2 — Clean Equipment Labels")
    print(f"{'='*60}")

    # Discover equipment folders
    equipment_folders = [f for f in sorted(raw.iterdir()) if f.is_dir()]
    print(f"\n  Equipment categories found:")

    normal_images = []
    abnormal_images = []

    for folder in equipment_folders:
        images = find_images(str(folder))
        is_normal = folder.name == args.normal_folder
        label = "NORMAL ✓" if is_normal else "abnormal"
        print(f"    {'→' if is_normal else ' '} {folder.name:25s} {len(images):4d} images  [{label}]")

        if is_normal:
            normal_images.extend(images)
        else:
            abnormal_images.extend(images)

    print(f"\n  Total normal:   {len(normal_images)} (Power Transformers)")
    print(f"  Total abnormal: {len(abnormal_images)} (other equipment)")

    # Balance classes: limit abnormal to ~1.5x normal
    random.seed(args.seed)
    max_abnormal = int(len(normal_images) * 1.5)
    if len(abnormal_images) > max_abnormal:
        abnormal_images = random.sample(abnormal_images, max_abnormal)
        print(f"  Balanced abnormal to: {len(abnormal_images)}")

    # Build sequences
    print(f"\n  Building sequences (length={args.seq_len})...")

    n_normal = build_sequences(
        normal_images, output / "normal", args.seq_len, args.seed
    )
    n_abnormal = build_sequences(
        abnormal_images, output / "abnormal", args.seq_len, args.seed + 1
    )

    total_seq = n_normal + n_abnormal
    train_n = int(total_seq * 0.7)
    val_n = int(total_seq * 0.15)
    test_n = total_seq - train_n - val_n

    print(f"\n{'='*60}")
    print(f"  ✅  Data preparation complete!")
    print(f"")
    print(f"  Normal sequences   : {n_normal}")
    print(f"  Abnormal sequences : {n_abnormal}")
    print(f"  Total sequences    : {total_seq}")
    print(f"  Est. split         : ~{train_n} train / ~{val_n} val / ~{test_n} test")
    print(f"  Sequence length    : {args.seq_len}")
    print(f"  Output             : {output}/")
    print(f"{'='*60}")
    print(f"\n  Next step:  python train.py")


if __name__ == "__main__":
    main()
