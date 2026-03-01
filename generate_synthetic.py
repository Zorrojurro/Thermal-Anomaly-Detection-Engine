#!/usr/bin/env python3
"""
Generate synthetic thermal images for testing the pipeline.

Creates realistic-looking synthetic infrared images with:
  - Normal patterns: smooth, uniform heat distribution
  - Abnormal patterns: localised hotspots, uneven heating

This lets you test the full training pipeline immediately
without waiting for a real dataset download.

Usage:
    python generate_synthetic.py                       # default 200 images
    python generate_synthetic.py --count 500
    python generate_synthetic.py --count 100 --preview  # show samples
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path


def make_base_transformer(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    """
    Generate a base thermal image resembling a power transformer.

    Uses overlapping Gaussian blobs to simulate the natural heat
    distribution of transformer components (core, windings, bushings).
    """
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w), dtype=np.float64)

    # Background ambient temperature (cool)
    img += rng.uniform(30, 60)

    # Transformer body (warm rectangle in the centre)
    body_x = int(w * 0.25)
    body_y = int(h * 0.2)
    body_w = int(w * 0.5)
    body_h = int(h * 0.6)
    img[body_y:body_y + body_h, body_x:body_x + body_w] += rng.uniform(40, 70)

    # Add Gaussian blobs for internal components
    n_blobs = rng.randint(3, 7)
    for _ in range(n_blobs):
        cx = rng.randint(body_x + 20, body_x + body_w - 20)
        cy = rng.randint(body_y + 20, body_y + body_h - 20)
        radius = rng.randint(30, 80)
        intensity = rng.uniform(20, 50)

        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        blob = intensity * np.exp(-dist ** 2 / (2 * radius ** 2))
        img += blob

    # Bushings (top protrusions)
    for bx in [body_x + body_w * 0.25, body_x + body_w * 0.5, body_x + body_w * 0.75]:
        bx = int(bx)
        by = body_y - 20
        r = rng.randint(15, 30)
        intensity = rng.uniform(30, 60)
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - bx) ** 2 + (y_grid - by) ** 2)
        blob = intensity * np.exp(-dist ** 2 / (2 * r ** 2))
        img += blob

    # Add noise
    noise = rng.normal(0, 3, (h, w))
    img += noise

    return img


def make_normal_image(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    """Generate a normal thermal image (uniform heat distribution)."""
    img = make_base_transformer(h, w, seed)

    # Slight temporal variation (simulates different load conditions)
    rng = np.random.RandomState(seed + 10000)
    variation = rng.uniform(-5, 5)
    img += variation

    # Normalize to 0-255
    img = np.clip(img, 0, None)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
    return img.astype(np.uint8)


def make_abnormal_image(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    """
    Generate an abnormal thermal image with hotspot anomalies.

    Anomaly types:
      - Concentrated hotspot (overheating component)
      - Temperature gradient anomaly (cooling failure)
      - Multiple scattered hotspots (insulation breakdown)
    """
    img = make_base_transformer(h, w, seed)
    rng = np.random.RandomState(seed + 20000)

    anomaly_type = rng.choice(["hotspot", "gradient", "scattered"])

    body_x = int(w * 0.25)
    body_y = int(h * 0.2)
    body_w = int(w * 0.5)
    body_h = int(h * 0.6)

    if anomaly_type == "hotspot":
        # Single intense hotspot
        cx = rng.randint(body_x + 30, body_x + body_w - 30)
        cy = rng.randint(body_y + 30, body_y + body_h - 30)
        r = rng.randint(20, 50)
        intensity = rng.uniform(80, 150)

        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        hotspot = intensity * np.exp(-dist ** 2 / (2 * r ** 2))
        img += hotspot

    elif anomaly_type == "gradient":
        # Asymmetric temperature gradient (one side much hotter)
        direction = rng.choice(["left", "right", "top", "bottom"])
        gradient = np.zeros((h, w), dtype=np.float64)

        if direction == "left":
            gradient = np.tile(np.linspace(80, 0, w), (h, 1))
        elif direction == "right":
            gradient = np.tile(np.linspace(0, 80, w), (h, 1))
        elif direction == "top":
            gradient = np.tile(np.linspace(80, 0, h), (w, 1)).T
        else:
            gradient = np.tile(np.linspace(0, 80, h), (w, 1)).T

        # Only apply inside the transformer body
        mask = np.zeros((h, w), dtype=np.float64)
        mask[body_y:body_y + body_h, body_x:body_x + body_w] = 1.0
        img += gradient * mask

    elif anomaly_type == "scattered":
        # Multiple small hotspots
        n_spots = rng.randint(4, 10)
        for _ in range(n_spots):
            cx = rng.randint(body_x, body_x + body_w)
            cy = rng.randint(body_y, body_y + body_h)
            r = rng.randint(8, 25)
            intensity = rng.uniform(50, 100)

            y_grid, x_grid = np.ogrid[:h, :w]
            dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
            spot = intensity * np.exp(-dist ** 2 / (2 * r ** 2))
            img += spot

    # Normalize to 0-255
    img = np.clip(img, 0, None)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
    return img.astype(np.uint8)


def apply_thermal_colormap(gray: np.ndarray) -> np.ndarray:
    """Apply an infrared-style colormap (like a real FLIR camera)."""
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic thermal images for pipeline testing"
    )
    p.add_argument(
        "--output", type=str, default="data/raw/synthetic",
        help="Output directory (default: data/raw/synthetic)",
    )
    p.add_argument(
        "--count", type=int, default=200,
        help="Total number of images to generate (split 60/40 normal/abnormal)",
    )
    p.add_argument(
        "--seq-ready", action="store_true",
        help="Directly output into data/sequences/ format (skip prepare_data.py)",
    )
    p.add_argument(
        "--seq-len", type=int, default=20,
        help="Images per sequence (only with --seq-ready)",
    )
    p.add_argument(
        "--color", action="store_true",
        help="Save with infrared colormap (like real FLIR output)",
    )
    p.add_argument(
        "--preview", action="store_true",
        help="Generate 10 samples and display info only",
    )
    return p.parse_args()


def main():
    args = parse_args()

    n_total = args.count if not args.preview else 10
    n_normal = int(n_total * 0.6)
    n_abnormal = n_total - n_normal

    print(f"Generating {n_total} synthetic thermal images...")
    print(f"  Normal:   {n_normal}")
    print(f"  Abnormal: {n_abnormal}")

    if args.seq_ready:
        # Output directly into sequence folders
        out_normal = Path("data/sequences/normal")
        out_abnormal = Path("data/sequences/abnormal")
    else:
        out_normal = Path(args.output) / "normal"
        out_abnormal = Path(args.output) / "abnormal"

    out_normal.mkdir(parents=True, exist_ok=True)
    out_abnormal.mkdir(parents=True, exist_ok=True)

    # Generate normal images
    if args.seq_ready:
        seq_len = args.seq_len
        seq_idx = 0
        for i in range(n_normal):
            if i % seq_len == 0:
                seq_idx += 1
                seq_dir = out_normal / f"seq_{seq_idx:03d}"
                seq_dir.mkdir(parents=True, exist_ok=True)

            img = make_normal_image(seed=i)
            if args.color:
                img = apply_thermal_colormap(img)

            frame_idx = (i % seq_len) + 1
            cv2.imwrite(str(seq_dir / f"frame_{frame_idx:03d}.png"), img)
    else:
        for i in range(n_normal):
            img = make_normal_image(seed=i)
            if args.color:
                img = apply_thermal_colormap(img)
            cv2.imwrite(str(out_normal / f"normal_{i+1:04d}.png"), img)

    print(f"  ✓ Generated {n_normal} normal images")

    # Generate abnormal images
    if args.seq_ready:
        seq_idx = 0
        for i in range(n_abnormal):
            if i % seq_len == 0:
                seq_idx += 1
                seq_dir = out_abnormal / f"seq_{seq_idx:03d}"
                seq_dir.mkdir(parents=True, exist_ok=True)

            img = make_abnormal_image(seed=i)
            if args.color:
                img = apply_thermal_colormap(img)

            frame_idx = (i % seq_len) + 1
            cv2.imwrite(str(seq_dir / f"frame_{frame_idx:03d}.png"), img)
    else:
        for i in range(n_abnormal):
            img = make_abnormal_image(seed=i)
            if args.color:
                img = apply_thermal_colormap(img)
            cv2.imwrite(str(out_abnormal / f"abnormal_{i+1:04d}.png"), img)

    print(f"  ✓ Generated {n_abnormal} abnormal images")

    # Summary
    if args.seq_ready:
        n_normal_seq = (n_normal + seq_len - 1) // seq_len
        n_abnormal_seq = (n_abnormal + seq_len - 1) // seq_len
        print(f"\n{'='*60}")
        print(f"  ✅  Synthetic dataset ready!")
        print(f"  Normal sequences   : {n_normal_seq}")
        print(f"  Abnormal sequences : {n_abnormal_seq}")
        print(f"  Output             : data/sequences/")
        print(f"{'='*60}")
        print(f"\n  Next step:  python train.py")
    else:
        print(f"\n{'='*60}")
        print(f"  ✅  Synthetic images generated!")
        print(f"  Output: {args.output}/")
        print(f"{'='*60}")
        print(f"\n  Next step:  python prepare_data.py --raw {args.output}")


if __name__ == "__main__":
    main()
