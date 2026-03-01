#!/usr/bin/env python3
"""
Data Preparation Script — Automatic organisation of thermal images.

This script:
  1. Scans a raw image directory
  2. Analyses each image for thermal anomalies (hotspot detection)
  3. Classifies images as normal / abnormal automatically
  4. Groups them into sequences of N images
  5. Saves to data/sequences/normal/ and data/sequences/abnormal/

Usage:
    python prepare_data.py --raw data/raw
    python prepare_data.py --raw data/raw --seq-len 20 --preview
"""

import os
import sys
import shutil
import argparse
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────
# Thermal anomaly heuristics
# ──────────────────────────────────────────────────────────────────────

def analyse_thermal_image(img_path: str) -> dict:
    """
    Analyse a thermal image and compute anomaly indicators.

    Uses multiple heuristics:
      - Mean intensity (higher = hotter overall)
      - Intensity standard deviation (higher = uneven heating)
      - Hotspot ratio: fraction of pixels above 90th percentile
      - Max gradient magnitude (sharp temperature boundaries)
      - Skewness of intensity distribution

    Returns a dict of metrics + an anomaly_score (0–1).
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Convert to grayscale if colour (most IR datasets use rainbow palette)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Also extract the "heat" from HSV value channel
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        heat = hsv[:, :, 2]  # Value channel captures brightness
    else:
        gray = img
        heat = img

    h, w = gray.shape

    # Basic stats
    mean_val = np.mean(heat).item()
    std_val = np.std(heat).item()
    max_val = np.max(heat).item()
    min_val = np.min(heat).item()

    # Hotspot ratio: pixels above the 90th percentile
    p90 = np.percentile(heat, 90)
    hotspot_ratio = np.sum(heat > p90) / (h * w)

    # Gradient magnitude (edge strength → sharp thermal boundaries)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    mean_gradient = np.mean(grad_mag).item()

    # Skewness (positive = right-tailed = concentrated hotspots)
    centered = heat.astype(np.float64) - mean_val
    skewness = np.mean(centered ** 3) / (std_val ** 3 + 1e-8)

    # Composite anomaly score (0 = very normal, 1 = very abnormal)
    # Weighted combination of normalised indicators
    score = 0.0
    score += 0.30 * min(std_val / 80.0, 1.0)       # high variance
    score += 0.25 * min(hotspot_ratio / 0.15, 1.0)  # concentrated hotspots
    score += 0.20 * min(mean_gradient / 50.0, 1.0)  # sharp boundaries
    score += 0.15 * min(abs(skewness) / 2.0, 1.0)   # skewed distribution
    score += 0.10 * min(max_val / 255.0, 1.0)       # extreme temps

    return {
        "path": img_path,
        "mean": mean_val,
        "std": std_val,
        "max": max_val,
        "min": min_val,
        "hotspot_ratio": hotspot_ratio,
        "mean_gradient": mean_gradient,
        "skewness": skewness,
        "anomaly_score": round(score, 4),
    }


def classify_images(analyses: list, threshold: float = 0.45) -> dict:
    """
    Split images into normal / abnormal based on anomaly score.

    Uses a percentile-based adaptive threshold if the fixed threshold
    doesn't produce a reasonable split (at least 20% in each class).
    """
    scores = [a["anomaly_score"] for a in analyses]

    # Check if fixed threshold gives reasonable split
    n_abnormal = sum(1 for s in scores if s >= threshold)
    ratio = n_abnormal / len(scores) if scores else 0

    if ratio < 0.15 or ratio > 0.85:
        # Adaptive: use median as boundary
        threshold = float(np.percentile(scores, 65))
        print(f"  ⚙ Adaptive threshold: {threshold:.3f} "
              f"(fixed threshold gave {ratio:.0%} abnormal)")

    normal = [a for a in analyses if a["anomaly_score"] < threshold]
    abnormal = [a for a in analyses if a["anomaly_score"] >= threshold]

    return {"normal": normal, "abnormal": abnormal, "threshold": threshold}


# ──────────────────────────────────────────────────────────────────────
# Sequence builder
# ──────────────────────────────────────────────────────────────────────

def build_sequences(
    image_analyses: list,
    output_dir: str,
    label: str,
    seq_length: int = 20,
):
    """
    Group images into sequence folders of *seq_length* images each.

    Images are sorted by anomaly score within each class to create
    coherent sequences (similar-looking images together).
    """
    out = Path(output_dir) / label
    out.mkdir(parents=True, exist_ok=True)

    # Sort by anomaly score for coherent grouping
    sorted_imgs = sorted(image_analyses, key=lambda x: x["anomaly_score"])

    seq_count = 0
    for i in range(0, len(sorted_imgs), seq_length):
        batch = sorted_imgs[i : i + seq_length]
        if len(batch) < max(3, seq_length // 4):
            # Too few images for a meaningful sequence — skip
            continue

        seq_count += 1
        seq_dir = out / f"seq_{seq_count:03d}"
        seq_dir.mkdir(parents=True, exist_ok=True)

        for j, analysis in enumerate(batch):
            src = Path(analysis["path"])
            dst = seq_dir / f"frame_{j+1:03d}{src.suffix}"
            shutil.copy2(src, dst)

    return seq_count


# ──────────────────────────────────────────────────────────────────────
# Preview
# ──────────────────────────────────────────────────────────────────────

def preview_classification(classified: dict, n: int = 5):
    """Print a preview of the classification results."""
    print(f"\n{'─'*60}")
    print(f"  Normal images  : {len(classified['normal'])}")
    print(f"  Abnormal images: {len(classified['abnormal'])}")
    print(f"  Threshold      : {classified['threshold']:.3f}")
    print(f"{'─'*60}")

    for label in ("normal", "abnormal"):
        images = classified[label]
        print(f"\n  Top {min(n, len(images))} {label.upper()} (by anomaly score):")
        show = sorted(images, key=lambda x: x["anomaly_score"],
                       reverse=(label == "abnormal"))[:n]
        for a in show:
            name = Path(a["path"]).name
            print(f"    {name:30s}  score={a['anomaly_score']:.4f}  "
                  f"std={a['std']:.1f}  hotspot={a['hotspot_ratio']:.3f}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Automatically organise thermal images into sequences"
    )
    p.add_argument(
        "--raw", type=str, default="data/raw",
        help="Directory containing raw thermal images (can have sub-folders)",
    )
    p.add_argument(
        "--output", type=str, default="data/sequences",
        help="Output directory for organised sequences",
    )
    p.add_argument(
        "--seq-len", type=int, default=20,
        help="Number of images per sequence (default: 20)",
    )
    p.add_argument(
        "--threshold", type=float, default=0.45,
        help="Anomaly score threshold for classification (default: 0.45)",
    )
    p.add_argument(
        "--preview", action="store_true",
        help="Preview classification without copying files",
    )
    return p.parse_args()


def find_images(directory: str) -> list:
    """Recursively find all image files in a directory."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    images = []
    for root, _, files in os.walk(directory):
        for f in files:
            if Path(f).suffix.lower() in extensions:
                images.append(os.path.join(root, f))
    return sorted(images)


def main():
    args = parse_args()

    raw_dir = Path(args.raw)
    if not raw_dir.exists():
        print(f"✗ Raw directory not found: {raw_dir}")
        print(f"\n  Please download thermal images and place them in: {raw_dir}")
        print(f"  The script will scan all sub-folders automatically.")
        print(f"\n  Recommended dataset:")
        print(f"  → SciDB: https://www.scidb.cn/en/detail?dataSetId=e416c488169f484485ad7575dcfc43ce")
        print(f"  → Download → extract → copy images to {raw_dir}/")
        sys.exit(1)

    # 1. Find all images
    print(f"Scanning {raw_dir} for thermal images...")
    image_paths = find_images(str(raw_dir))
    print(f"  Found {len(image_paths)} images")

    if not image_paths:
        print("✗ No images found. Check the directory path.")
        sys.exit(1)

    # 2. Analyse each image
    print("Analysing thermal patterns...")
    analyses = []
    for path in tqdm(image_paths, desc="  Analysing"):
        result = analyse_thermal_image(path)
        if result:
            analyses.append(result)
    print(f"  Successfully analysed {len(analyses)} images")

    # 3. Classify
    print("Classifying normal vs abnormal...")
    classified = classify_images(analyses, threshold=args.threshold)
    preview_classification(classified)

    if args.preview:
        print("\n  [Preview mode — no files copied]")
        return

    # 4. Build sequences
    print(f"\nBuilding sequences (length={args.seq_len})...")
    n_normal = build_sequences(
        classified["normal"], args.output, "normal", args.seq_len
    )
    n_abnormal = build_sequences(
        classified["abnormal"], args.output, "abnormal", args.seq_len
    )

    print(f"\n{'='*60}")
    print(f"  ✅  Data preparation complete!")
    print(f"  Normal sequences   : {n_normal}")
    print(f"  Abnormal sequences : {n_abnormal}")
    print(f"  Output directory   : {args.output}/")
    print(f"{'='*60}")
    print(f"\n  Next step:  python train.py")


if __name__ == "__main__":
    main()
