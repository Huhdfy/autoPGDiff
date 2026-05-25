"""Side-by-side: Input → Baseline → Cutoff — for image.png."""
import cv2, numpy as np, os

RESULTS = "../results"
paths = [
    "/tmp/pgdiff_test/image.png",              # original input (resized to 512 by model)
    "../results/exp_image_baseline/image.png",  # baseline
    "../results/exp_image_cutoff/image.png",    # cutoff
]
labels = [
    "Input",
    "Baseline (qs=96.9, sharp=74.9)",
    "Cutoff last 20% (qs=96.4, sharp=75.2)",
]

imgs = []
for p in paths:
    im = cv2.imread(p)
    if im is None:
        im = np.zeros((512, 512, 3), dtype=np.uint8)
    else:
        im = cv2.resize(im, (512, 512))
    imgs.append(im)

h, w = 512, 512
pad = 10
n = 3

# ── Top: full, Bottom: center crop 2x ──
crop_sz = 150
cy, cx = h // 2 - 10, w // 2

canvas_w = w * n + pad * (n + 1)
canvas_h = h + 160 + h + pad * 3
canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 245
font = cv2.FONT_HERSHEY_SIMPLEX

for col in range(n):
    x0 = pad + col * (w + pad)
    img = imgs[col]

    canvas[pad:pad + h, x0:x0 + w] = img

    crop = img[cy - crop_sz // 2:cy + crop_sz // 2, cx - crop_sz // 2:cx + crop_sz // 2]
    crop_big = cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)
    y0 = h + pad * 2 + 36
    canvas[y0:y0 + h, x0:x0 + w] = crop_big

    cv2.rectangle(canvas, (x0, pad + h - 28), (x0 + 300, pad + h - 4), (0, 0, 0), -1)
    cv2.putText(canvas, labels[col], (x0 + 6, pad + h - 9), font, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, "  center crop (2x)", (x0 + 6, y0 + 22), font, 0.45, (60, 60, 60), 1)

cv2.rectangle(canvas, (pad + cx - crop_sz // 2, pad + cy - crop_sz // 2),
              (pad + cx + crop_sz // 2, pad + cy + crop_sz // 2), (0, 255, 0), 2)
cv2.putText(canvas, "PGDiff on image.png — Input vs Baseline vs Cutoff (last 20% guidance=0)",
            (pad, canvas_h - 8), font, 0.65, (0, 0, 0), 2)

out = os.path.join(RESULTS, "comparison", "image_png_comparison.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
cv2.imwrite(out, canvas)
print(f"Saved: {out}")
