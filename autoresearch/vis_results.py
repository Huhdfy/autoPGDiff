#!/usr/bin/env python3
"""可视化对比：加速实验输出 vs baseline"""
import cv2, numpy as np, os

vis_dir = "/mnt/workspace/autoPGDiff/results/vis_compare"
files = [
    ("1_input.png",           "Input (LQ)"),
    ("2_纯扩散(无引导).png",    "Pure Diff (无引导)\nV2=2.63 57s"),
    ("3_全引导(baseline).png", "Full Guide (baseline)\nV2=3.44 132s"),
    ("4_稀疏K=2.png",         "Sparse K=2\nV2=3.79 95s"),
    ("5_稀疏K=3.png",         "Sparse K=3\nV2=3.77 82s"),
    ("6_稀疏K=5.png",         "Sparse K=5\nV2=3.68 73s"),
    ("7_稀疏K=10.png",        "Sparse K=10\nV2=3.57 65s"),
    ("8_提前终止s=0.5.png",    "Early Stop s=0.5\nV2=3.87 95s"),
    ("9_DPM35步.png",         "DPM 35步 (对照)\nV2=0.80 12s"),
]

imgs = []
titles = []
for fname, title in files:
    path = os.path.join(vis_dir, fname)
    img = cv2.imread(path)
    if img is not None:
        img = cv2.resize(img, (256, 256))
        imgs.append(img)
        titles.append(title)

n = len(imgs)
cols = 3
rows = (n + cols - 1) // cols
h, w = 256, 256
canvas = np.ones((rows * h + (rows - 1) * 4, cols * w + (cols - 1) * 4, 3), dtype=np.uint8) * 40

for idx, (img, title) in enumerate(zip(imgs, titles)):
    r, c = idx // cols, idx % cols
    y0, x0 = r * (h + 4), c * (w + 4)
    canvas[y0:y0 + h, x0:x0 + w] = img
    cv2.putText(canvas, title.split("\n")[0], (x0 + 4, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    if "\n" in title:
        cv2.putText(canvas, title.split("\n")[1], (x0 + 4, y0 + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 255, 180), 1)

out_path = os.path.join(vis_dir, "comparison_grid.png")
cv2.imwrite(out_path, canvas)
print(f"Written: {out_path}")
print(f"Size: {canvas.shape[1]}x{canvas.shape[0]}")
