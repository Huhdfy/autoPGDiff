"""Side-by-side: Input ← Baseline → Cutoff — with close-up crops."""
import cv2, numpy as np, os

RESULTS = "../results"
paths = [
    "../testdata/cropped_faces/0040.png",  # input
    "../results/exp_baseline/0040.png",     # baseline
    "../results/exp_cutoff_last20/0040.png",# cutoff
]
labels = ["Input (Degraded)", "Baseline (DDPM 1000, s=0.1)", "Cutoff (last 20% guidance=0)"]

imgs = [cv2.imread(p) for p in paths]
for i, im in enumerate(imgs):
    if im is None:

        imgs[i] = np.zeros((512, 512, 3), dtype=np.uint8)

h, w = imgs[0].shape[:2]
pad = 12

# ── Top row: full images, Bottom row: center 2x crop ──
crop_sz = 160
cy, cx = h // 2 - 20, w // 2  # slightly above center for face focus

n = len(imgs)
canvas_w = w * n + pad * (n + 1)
canvas_h = h + 160 + h + pad * 3
canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240

font = cv2.FONT_HERSHEY_SIMPLEX

for col in range(n):
    x0 = pad + col * (w + pad)
    img = imgs[col]

    # Full image
    canvas[pad:pad + h, x0:x0 + w] = img

    # Zoomed crop
    crop = img[cy - crop_sz // 2:cy + crop_sz // 2, cx - crop_sz // 2:cx + crop_sz // 2]
    crop_big = cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)
    y0 = h + pad * 2 + 36
    canvas[y0:y0 + h, x0:x0 + w] = crop_big

    # Label
    cv2.rectangle(canvas, (x0, pad + h - 28), (x0 + 310, pad + h - 4), (0, 0, 0), -1)
    cv2.putText(canvas, labels[col], (x0 + 6, pad + h - 9), font, 0.5, (255, 255, 255), 1)

    # Zoom label
    cv2.putText(canvas, f"  center crop (2x)", (x0 + 6, y0 + 22), font, 0.45, (60, 60, 60), 1)

# Crop indicator on top row
cv2.rectangle(canvas, (pad + cx - crop_sz // 2, pad + cy - crop_sz // 2),
              (pad + cx + crop_sz // 2, pad + cy + crop_sz // 2), (0, 255, 0), 2)

# Title
cv2.putText(canvas, "PGDiff: Input vs Baseline vs Guidance Cutoff (last 20%)",
            (pad, canvas_h - 8), font, 0.7, (0, 0, 0), 2)

out_path = os.path.join(RESULTS, "comparison", "baseline_vs_cutoff.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
cv2.imwrite(out_path, canvas)
print(f"Saved: {out_path}")
