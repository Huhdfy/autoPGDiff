#!/usr/bin/env python3
"""Visual comparison using PIL/Pillow with proper font rendering."""
from PIL import Image, ImageDraw, ImageFont
import os

WORK = "/mnt/workspace/autoPGDiff/autoresearch/results"

configs = [
    ("Input (LQ)",           "input.png",
     ["SSIM=-  Sharp=-", "Edge=-  RC=-"]),
    ("Paper Baseline",       "paper_baseline/0040.png",
     ["SSIM=.910  Sharp=25.2", "Edge=.187  RC=1000 139s"]),
    ("Optimized Baseline",   "optim_baseline/0040.png",
     ["SSIM=.893  Sharp=33.8", "Edge=.208  RC=500 99s"]),
    ("K=2+sc=0.20",          "k2_sc020/0040.png",
     ["SSIM=.893  Sharp=33.7", "Edge=.208  RC=250 79s"]),
    ("K=2+sc=0.10",          "k2_sc010/0040.png",
     ["SSIM=.870  Sharp=46.3", "Edge=.242  RC=250 80s"]),
    ("Pure Diffusion",       "pure_diff/0040.png",
     ["SSIM=.607  Sharp=37.2", "Edge=.255  RC=0 61s"]),
]

TITLE = "PGDiff Face Restoration  |  +34% Detail vs Paper Baseline  |  K=2+sc=0.20 = same quality, 50% fewer Restorer Calls"

# Load images, resize to uniform height
H = 360
images = []
for name, relpath, labels in configs:
    path = os.path.join(WORK, relpath)
    img = Image.open(path)
    w0, h0 = img.size
    w = int(w0 * H / h0)
    img = img.resize((w, H), Image.LANCZOS)
    images.append((name, img, labels))

# Determine column width (use max width)
col_w = max(img.size[0] for _, img, _ in images)
row_h = H

# Load a font
try:
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
except:
    font_title = ImageFont.load_default()
    font_label = font_title
    font_big = font_title

# Layout:
# Row 1: config name bars
# Row 2: images
# Row 3: metric labels

LABEL_BAR_H = 28
METRIC_BAR_H = 36
TITLE_H = 40

grid_w = col_w * 3
grid_h = (LABEL_BAR_H + row_h + METRIC_BAR_H) * 2

canvas = Image.new("RGB", (grid_w, TITLE_H + grid_h), (245, 245, 245))
draw = ImageDraw.Draw(canvas)

# Title
draw.text((14, 10), TITLE, fill=(20, 20, 20), font=font_title)
draw.line([(0, TITLE_H - 1), (grid_w, TITLE_H - 1)], fill=(180, 180, 180))

# Color coding for categories
colors = {
    "Input (LQ)":         (200, 200, 210),
    "Paper Baseline":     (255, 200, 200),
    "Optimized Baseline": (200, 255, 200),
    "K=2+sc=0.20":        (180, 230, 180),
    "K=2+sc=0.10":        (255, 255, 200),
    "Pure Diffusion":     (200, 220, 255),
}

for i in range(2):
    for j in range(3):
        idx = i * 3 + j
        name, img, labels = images[idx]

        x0 = j * col_w
        y0 = TITLE_H + i * (LABEL_BAR_H + row_h + METRIC_BAR_H)

        # Color bar (config name)
        color = colors.get(name, (230, 230, 230))
        draw.rectangle([x0, y0, x0 + col_w - 1, y0 + LABEL_BAR_H - 1], fill=color)
        # Border between cells
        if j > 0:
            draw.line([(x0, y0), (x0, y0 + LABEL_BAR_H - 1)], fill=(160, 160, 160))

        # Centered name text
        try:
            bbox = draw.textbbox((0, 0), name, font=font_label)
            tw = bbox[2] - bbox[0]
        except:
            tw = len(name) * 7
        tx = x0 + (col_w - tw) // 2
        ty = y0 + (LABEL_BAR_H - 14) // 2
        draw.text((tx, ty), name, fill=(30, 30, 30), font=font_label)

        # Paste image
        img_y = y0 + LABEL_BAR_H
        # Center horizontally if image is narrower than column
        img_x = x0 + (col_w - img.size[0]) // 2
        canvas.paste(img, (img_x, img_y))

        # Metric label bar
        met_y = img_y + row_h
        draw.rectangle([x0, met_y, x0 + col_w - 1, met_y + METRIC_BAR_H - 1], fill=(238, 238, 238))

        # Row divider
        if j > 0:
            draw.line([(x0, met_y), (x0, met_y + METRIC_BAR_H - 1)], fill=(160, 160, 160))

        for k, label in enumerate(labels):
            tx = x0 + 6
            ty = met_y + 4 + k * 14
            draw.text((tx, ty), label, fill=(50, 50, 50), font=font_label)

# Horizontal dividers between rows of cells
for i in range(1, 2):
    y = TITLE_H + i * (LABEL_BAR_H + row_h + METRIC_BAR_H)
    draw.line([(0, y - 1), (grid_w, y - 1)], fill=(180, 180, 180))

out_path = os.path.join(WORK, "comparison_final.png")
canvas.save(out_path)
print(f"Saved: {out_path}  ({canvas.size[0]}x{canvas.size[1]})")

# Also create a compact crop comparison
crop_h = 200
crop_w = 200
crops = []
for name, relpath, _ in configs:
    img = Image.open(os.path.join(WORK, relpath))
    w0, h0 = img.size
    # Center crop
    x0 = (w0 - 256) // 2
    y0 = (h0 - 256) // 2
    crop = img.crop((x0, y0, x0 + 256, y0 + 256))
    crop = crop.resize((crop_w, crop_h), Image.LANCZOS)

    # Add border color
    color = colors.get(name, (230, 230, 230))
    bordered = Image.new("RGB", (crop_w + 4, crop_h + 4), color)
    bordered.paste(crop, (2, 2))
    crops.append(bordered)

crop_grid = Image.new("RGB", ((crop_w + 4) * 6, crop_h + 4))
for i, c in enumerate(crops):
    crop_grid.paste(c, (i * (crop_w + 4), 0))

out_path2 = os.path.join(WORK, "comparison_crop.png")
crop_grid.save(out_path2)
print(f"Saved crop: {out_path2}  ({crop_grid.size[0]}x{crop_grid.size[1]})")
