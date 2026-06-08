#!/usr/bin/env python
"""
Generate GT+LQ paired test data for face restoration evaluation.
Uses HQ reference faces from testdata/ref_faces/ as GT, applies synthetic
degradation to create LQ pairs.

Degradation pipeline follows the standard approach used in DifFace/PGDiff:
  - Gaussian blur (sigma)
  - Downsample + upsample (scale factor)
  - JPEG compression (quality factor)
  - Gaussian noise addition
"""

import os
import sys
import random
import argparse
from pathlib import Path

import cv2
import numpy as np

# ── Degradation functions ───────────────────────────────────────────────

def add_gaussian_blur(img, sigma):
    """Apply Gaussian blur with given sigma."""
    if sigma <= 0:
        return img
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def add_downsample(img, scale):
    """Downsample by factor then upsample back with bicubic."""
    if scale <= 1:
        return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def add_jpeg_compression(img, quality):
    """Apply JPEG compression artifacts."""
    if quality >= 100:
        return img
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def add_gaussian_noise(img, sigma):
    """Add Gaussian noise."""
    if sigma <= 0:
        return img
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_degradation(img, blur_sigma, ds_scale, jpeg_qf, noise_sigma, seed=None):
    """Apply full degradation pipeline in order: blur -> downsample -> JPEG -> noise."""
    if seed is not None:
        np.random.seed(seed)
    img = add_gaussian_blur(img, blur_sigma)
    img = add_downsample(img, ds_scale)
    img = add_jpeg_compression(img, jpeg_qf)
    img = add_gaussian_noise(img, noise_sigma)
    return img


# ── Parameter sets ───────────────────────────────────────────────────────

# Degradation levels: mild, moderate, severe
DEGRADE_PARAMS = [
    # (blur_sigma, ds_scale, jpeg_qf, noise_sigma, label)
    (2.0,  4,  65, 3,  "mild"),
    (3.0,  6,  55, 5,  "mild"),
    (4.0,  8,  50, 7,  "moderate"),
    (5.0, 10,  45, 8,  "moderate"),
    (6.0, 12,  40, 9,  "moderate"),
    (7.0, 14,  38, 10, "severe"),
    (8.0, 16,  35, 11, "severe"),
    (9.0, 18,  32, 12, "severe"),
    (10.0, 20, 30, 13, "severe"),
    (12.0, 24, 28, 15, "severe"),
]


def main():
    parser = argparse.ArgumentParser(description="Generate GT+LQ paired test data")
    parser.add_argument("--gt_dir", type=str, default="testdata/ref_faces",
                        help="Directory with HQ ground truth images")
    parser.add_argument("--out_dir", type=str, default="testdata/pairs",
                        help="Output root directory")
    parser.add_argument("--num_pairs", type=int, default=10,
                        help="Number of GT+LQ pairs to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parents[1]
    gt_dir = project_root / args.gt_dir
    out_dir = project_root / args.out_dir
    gt_out = out_dir / "GT"
    lq_out = out_dir / "LQ"

    if not gt_dir.exists():
        print(f"ERROR: GT directory not found: {gt_dir}")
        sys.exit(1)

    # Read all HQ images
    gt_files = sorted(gt_dir.glob("*.png")) + sorted(gt_dir.glob("*.jpg"))
    if not gt_files:
        print(f"ERROR: No images found in {gt_dir}")
        sys.exit(1)

    print(f"Found {len(gt_files)} HQ images in {gt_dir}")

    # Create output directories
    gt_out.mkdir(parents=True, exist_ok=True)
    lq_out.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Assign degradation params to images
    gt_assignments = []
    for i in range(args.num_pairs):
        gt_idx = i % len(gt_files)     # cycle through GT images
        params = DEGRADE_PARAMS[i]     # each pair gets different params
        gt_assignments.append((gt_files[gt_idx], params))

    # Generate pairs
    print(f"\nGenerating {args.num_pairs} GT+LQ pairs ...")
    print(f"{'Pair':>6s}  {'GT Image':>16s}  {'Blur':>6s}  {'DS':>4s}  {'JPEG':>6s}  {'Noise':>6s}  {'Level':>10s}")
    print("-" * 72)

    for idx, (gt_path, (blur_s, ds, jpeg, noise, level)) in enumerate(gt_assignments):
        gt_img = cv2.imread(str(gt_path))
        if gt_img is None:
            print(f"  SKIP: cannot read {gt_path}")
            continue

        h, w = gt_img.shape[:2]
        gt_name = f"{idx:04d}_{gt_path.stem}.png"

        # Generate LQ
        lq_img = apply_degradation(
            gt_img, blur_sigma=blur_s, ds_scale=ds,
            jpeg_qf=jpeg, noise_sigma=noise, seed=args.seed + idx
        )

        # Save
        gt_save_path = gt_out / gt_name
        lq_save_path = lq_out / gt_name
        cv2.imwrite(str(gt_save_path), gt_img)
        cv2.imwrite(str(lq_save_path), lq_img)

        print(f"  {idx+1:>4d}  {gt_path.name:>16s}  {blur_s:>4.1f}  {ds:>4d}  {jpeg:>4d}  {noise:>4d}  {level:>10s}")

    print(f"\nDone. GT saved to {gt_out}")
    print(f"      LQ saved to {lq_out}")
    print(f"Total pairs: {args.num_pairs}")


if __name__ == "__main__":
    main()
