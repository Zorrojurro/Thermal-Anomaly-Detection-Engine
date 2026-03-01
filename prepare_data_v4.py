#!/usr/bin/env python3
"""
Data Preparation v4 — Robust anomaly injection AFTER preprocessing.

Key insight: Anomalies must survive the preprocessing pipeline.
Instead of injecting on raw images (which CLAHE normalizes away),
we inject anomalies on preprocessed images so the model sees
clear, distinct patterns.

Strategy:
  - Preprocess ALL Power Transformer images first
  - Normal = preprocessed originals (saved as PNG)
  - Abnormal = preprocessed images WITH post-processing anomalies
  - Stronger, more varied anomalo types
  - Balanced 50-50 split

Usage:
    python prepare_data_v4.py
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
# Preprocessing (replicate the pipeline exactly)
# ──────────────────────────────────────────────────────────────────────

def preprocess_image(img_path: str, size=(224, 224)) -> np.ndarray:
    """Apply the exact same preprocessing as ThermalImageProcessor."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Resize
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    # Bilateral filter (denoise)
    denoised = cv2.bilateralFilter(resized, d=9, sigmaColor=75, sigmaSpace=75)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    return enhanced  # uint8, 0-255


# ──────────────────────────────────────────────────────────────────────
# Post-preprocessing anomaly injection
# ──────────────────────────────────────────────────────────────────────

def inject_hotspot(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Bright focused hotspot — overheating component."""
    h, w = img.shape
    result = img.astype(np.float64)

    # 1-2 hotspots
    n = rng.randint(1, 2)
    for _ in range(n):
        cx = rng.randint(w // 4, 3 * w // 4)
        cy = rng.randint(h // 4, 3 * h // 4)
        radius = rng.randint(12, 32)
        intensity = rng.uniform(45, 85)

        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        blob = intensity * np.exp(-dist ** 2 / (2 * radius ** 2))
        result += blob

    return np.clip(result, 0, 255).astype(np.uint8)


def inject_dark_spot(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Dark cold region — oil leak or cooling excess."""
    h, w = img.shape
    result = img.astype(np.float64)

    cx = rng.randint(w // 4, 3 * w // 4)
    cy = rng.randint(h // 4, 3 * h // 4)
    radius = rng.randint(18, 42)
    intensity = rng.uniform(40, 75)

    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    blob = intensity * np.exp(-dist ** 2 / (2 * radius ** 2))
    result -= blob

    return np.clip(result, 0, 255).astype(np.uint8)


def inject_stripe_pattern(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Horizontal or vertical stripes — electrical interference."""
    h, w = img.shape
    result = img.astype(np.float64)

    is_horizontal = rng.random() > 0.5
    n_stripes = rng.randint(3, 8)
    intensity = rng.uniform(20, 50)

    for _ in range(n_stripes):
        if is_horizontal:
            y_pos = rng.randint(0, h - 1)
            thickness = rng.randint(2, 6)
            y_start = max(0, y_pos - thickness // 2)
            y_end = min(h, y_pos + thickness // 2)
            result[y_start:y_end, :] += intensity
        else:
            x_pos = rng.randint(0, w - 1)
            thickness = rng.randint(2, 6)
            x_start = max(0, x_pos - thickness // 2)
            x_end = min(w, x_pos + thickness // 2)
            result[:, x_start:x_end] += intensity

    return np.clip(result, 0, 255).astype(np.uint8)


def inject_asymmetric_heat(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """One half significantly brighter — uneven load / cooling failure."""
    h, w = img.shape
    result = img.astype(np.float64)

    direction = rng.choice(["left", "right", "top", "bottom"])
    strength = rng.uniform(35, 60)

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


def inject_ring_pattern(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Ring / halo pattern — corona discharge effect."""
    h, w = img.shape
    result = img.astype(np.float64)

    cx = rng.randint(w // 3, 2 * w // 3)
    cy = rng.randint(h // 3, 2 * h // 3)
    inner_r = rng.randint(15, 30)
    outer_r = inner_r + rng.randint(10, 25)
    intensity = rng.uniform(35, 65)

    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    ring = np.exp(-((dist - (inner_r + outer_r) / 2) ** 2) / (2 * 8 ** 2))
    result += intensity * ring

    return np.clip(result, 0, 255).astype(np.uint8)


def inject_noise_patch(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Noisy rectangular region — sensor malfunction."""
    h, w = img.shape
    result = img.copy()

    rx = rng.randint(w // 5, w // 2)
    ry = rng.randint(h // 5, h // 2)
    rw = rng.randint(w // 5, w // 3)
    rh = rng.randint(h // 5, h // 3)

    noise = np.random.RandomState(rng.randint(0, 2**31)).randint(
        -45, 45, size=(rh, rw)
    )
    patch = result[ry:ry + rh, rx:rx + rw].astype(np.float64)
    patch += noise
    result[ry:ry + rh, rx:rx + rw] = np.clip(patch, 0, 255).astype(np.uint8)

    return result


def inject_anomaly(img: np.ndarray, seed: int) -> np.ndarray:
    """Apply 1-2 random anomalies with controlled randomness."""
    rng = random.Random(seed)
    funcs = [
        inject_hotspot,
        inject_dark_spot,
        inject_stripe_pattern,
        inject_asymmetric_heat,
        inject_ring_pattern,
        inject_noise_patch,
    ]

    n = 1  # only 1 anomaly type for subtlety
    chosen = rng.sample(funcs, n)

    result = img.copy()
    for func in chosen:
        result = func(result, random.Random(seed + hash(func.__name__)))

    return result


# ──────────────────────────────────────────────────────────────────────
# Sequence builder
# ──────────────────────────────────────────────────────────────────────

def save_sequences(images: list, output_dir: Path, seq_len: int, seed: int) -> int:
    """Group images into sequence folders."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    imgs = list(images)
    rng.shuffle(imgs)

    seq_count = 0

    # Non-overlapping
    for i in range(0, len(imgs), seq_len):
        batch = imgs[i:i + seq_len]
        if len(batch) < max(2, seq_len // 2):
            continue
        seq_count += 1
        seq_dir = output_dir / f"seq_{seq_count:03d}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        for j, (name, pixel_data) in enumerate(batch):
            cv2.imwrite(str(seq_dir / f"frame_{j+1:03d}.png"), pixel_data)

    # Overlapping (stride=2)
    rng.shuffle(imgs)
    stride = max(1, seq_len // 2)
    for i in range(0, len(imgs) - seq_len, stride):
        batch = imgs[i:i + seq_len]
        seq_count += 1
        seq_dir = output_dir / f"seq_{seq_count:03d}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        for j, (name, pixel_data) in enumerate(batch):
            cv2.imwrite(str(seq_dir / f"frame_{j+1:03d}.png"), pixel_data)

    return seq_count


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=str, default="data/raw/Power Transformers")
    p.add_argument("--output", type=str, default="data/sequences")
    p.add_argument("--seq-len", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def find_images(d: str) -> list:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    return sorted([os.path.join(d, f) for f in os.listdir(d) if Path(f).suffix.lower() in exts])


def main():
    args = parse_args()
    raw = Path(args.raw)

    if not raw.exists():
        print(f"✗ Not found: {raw}")
        sys.exit(1)

    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Data Prep v4 — Post-Preprocessing Anomaly Injection")
    print(f"{'='*60}")

    # Find images
    all_paths = find_images(str(raw))
    print(f"\n  Source images: {len(all_paths)}")

    # Step 1: Preprocess ALL images
    print("  Preprocessing all images...")
    preprocessed = []
    for path in tqdm(all_paths, desc="  Preprocessing"):
        result = preprocess_image(path)
        if result is not None:
            preprocessed.append((Path(path).stem, result))
    print(f"  ✓ Preprocessed {len(preprocessed)} images")

    # Step 2: Normal = preprocessed originals
    normal_images = [(name, img.copy()) for name, img in preprocessed]

    # Step 3: Abnormal = preprocessed + injected anomalies
    print("  Injecting anomalies on preprocessed images...")
    abnormal_images = []
    for i, (name, img) in enumerate(tqdm(preprocessed, desc="  Injecting")):
        anomalous = inject_anomaly(img, args.seed + i)
        abnormal_images.append((f"{name}_anomaly", anomalous))

    # Also create extra anomalies with different seeds for more data
    for i, (name, img) in enumerate(preprocessed[:len(preprocessed) // 2]):
        anomalous = inject_anomaly(img, args.seed + 10000 + i)
        abnormal_images.append((f"{name}_anomaly2", anomalous))

    print(f"  ✓ Created {len(abnormal_images)} anomaly images")

    # Step 4: Build sequences
    print(f"\n  Building sequences (length={args.seq_len})...")
    n_normal = save_sequences(normal_images, output / "normal", args.seq_len, args.seed)
    n_abnormal = save_sequences(abnormal_images, output / "abnormal", args.seq_len, args.seed + 1)

    total = n_normal + n_abnormal
    print(f"\n{'='*60}")
    print(f"  ✅  Data preparation complete!")
    print(f"")
    print(f"  Normal sequences   : {n_normal}")
    print(f"  Abnormal sequences : {n_abnormal}")
    print(f"  Total sequences    : {total}")
    print(f"  Est. train/val/test: ~{int(total*0.7)} / ~{int(total*0.15)} / ~{int(total*0.15)}")
    print(f"{'='*60}")
    print(f"\n  Next step:  python train.py")


if __name__ == "__main__":
    main()
