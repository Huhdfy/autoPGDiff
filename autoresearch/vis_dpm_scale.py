#!/usr/bin/env python3
"""Visualize DPM-Solver-2 comparison: scale=0.1, 1.0, 5.0 for img 0006_00092."""
import os, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJ, "final_experiments")
os.makedirs(OUT_DIR, exist_ok=True)

IMG_ID = "0006_00092"
DPMSRC = os.path.join(PROJ, "final_experiments", "dpm_scale_compare")

IMGS = {}
# LQ
lq_path = os.path.join(PROJ, "testdata/pairs/LQ", f"{IMG_ID}.png")
IMGS["LQ"] = cv2.cvtColor(cv2.imread(lq_path), cv2.COLOR_BGR2RGB)
# GT
gt_path = os.path.join(PROJ, "testdata/pairs/GT", f"{IMG_ID}.png")
IMGS["GT"] = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
# DPM outputs
for tag, scale in [("s010", 0.10), ("s100", 1.0), ("s500", 5.0)]:
    path = os.path.join(DPMSRC, f"dpm35_{tag}.png")
    IMGS[f"DPM_{scale}"] = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

METRICS = {
    "DPM_0.1":  dict(psnr=8.50,  ssim=0.18, niqe=4908),
    "DPM_1.0":  dict(psnr=10.20, ssim=0.56, niqe=1194),
    "DPM_5.0":  dict(psnr=10.03, ssim=0.50, niqe=2016),
}

NCOLS = 5
fig = plt.figure(figsize=(15, 3.5))
gs = gridspec.GridSpec(1, NCOLS, figure=fig,
                       left=0.02, right=0.98, top=0.88, bottom=0.15,
                       wspace=0.03)

for col, (label, img) in enumerate(IMGS.items()):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(img)
    ax.axis("off")

    if label.startswith("DPM_"):
        scale_key = label.replace("DPM_", "")
        m = METRICS[f"DPM_{scale_key}"]
        anno = f"scale={float(scale_key):.2f}\nPSNR={m['psnr']:.2f}  SSIM={m['ssim']:.2f}  NIQE={m['niqe']:.0f}"
        ax.text(0.5, 0.02, anno, ha="center", va="bottom",
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                color="#222", bbox=dict(boxstyle="round,pad=0.3",
                facecolor="#f0f0f0", edgecolor="#aaaaaa", alpha=0.92))
    elif label == "LQ":
        ax.text(0.5, 0.02, "LQ PSNR=22.21", ha="center", va="bottom",
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                color="#222", bbox=dict(boxstyle="round,pad=0.3",
                facecolor="#ffe8e8", edgecolor="#aaaaaa", alpha=0.92))

    ax.set_title(label.replace("_", " "), fontsize=12, fontweight="bold", pad=6)

fig.suptitle("DPM-Solver-2 (35 steps) — Guidance Scale Comparison  |  Image: 0006_00092",
             fontsize=14, fontweight="bold", y=0.97)

# Bottom: comparison table
table_text = (
    "Config: DPM-Solver-2, 35 steps, LINEAR schedule, MSE loss, EDGE=0.02, BLEND=0.5  |  "
    "GPU: A10, ~12s/img, ~1053MB"
)
fig.text(0.5, 0.03, table_text, ha="center", va="center",
         fontsize=8.5, transform=fig.transFigure, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="#cccccc"))

out_path = os.path.join(OUT_DIR, "dpm_scale_comparison.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print(f"Saved: {out_path}")
