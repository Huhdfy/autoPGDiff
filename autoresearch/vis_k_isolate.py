#!/usr/bin/env python3
"""K isolation — images only, fixed scale=0.10, K=1→2→3→5 on Mild."""
import os, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "final_experiments")
KISO = os.path.join(OUT, "k_isolate")
TP = os.path.join(PROJ, "testdata", "pairs")

PATHS = [
    ("LQ",       os.path.join(TP, "LQ", "0000_00010.png")),
    ("GT",       os.path.join(TP, "GT", "0000_00010.png")),
    ("K=1  s=0.10", os.path.join(OUT, "exp_A0_true_paper", "0000_00010.png")),
    ("K=2  s=0.10", os.path.join(KISO, "K2_s010_mild.png")),
    ("K=3  s=0.10", os.path.join(KISO, "K3_s010_mild.png")),
    ("K=5  s=0.10", os.path.join(KISO, "K5_s010_mild.png")),
]

PSNR_VALS = [None, None, 26.47, 25.04, 23.83, 22.13]
NIQE_VALS = [None, None, 756, 792, 753, 834]

NCOLS = len(PATHS)
fig = plt.figure(figsize=(3.4 * NCOLS, 4.2))
gs = gridspec.GridSpec(1, NCOLS, figure=fig,
                       left=0.02, right=0.98, top=0.88, bottom=0.15,
                       wspace=0.03)

def load(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

for col, (label, path) in enumerate(PATHS):
    ax = fig.add_subplot(gs[0, col])
    img = load(path)
    if img is not None:
        ax.imshow(img)
    ax.axis("off")
    ax.set_title(label, fontsize=13, fontweight="bold", pad=8)

    psnr = PSNR_VALS[col]
    niqe = NIQE_VALS[col]
    if psnr is not None:
        bc = "#fff3cd" if col == 4 else "#f0f0f0"
        anno = f"PSNR={psnr:.2f}  NIQE={niqe:.0f}"
        ax.text(0.5, 0.015, anno, ha="center", va="bottom",
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=bc,
                          edgecolor="#aaa", alpha=0.92))

fig.suptitle("K Isolation: Fixed scale=0.10, K=1→2→3→5  (Mild, 0000_00010)",
             fontsize=16, fontweight="bold", y=0.96)
fig.text(0.5, 0.04, "PSNR monotonic ↓  |  NIQE U-shaped (K=3: 753 = global best)  |  MSE + CONSTANT + BLEND=0 + BLOCK_UNET",
         ha="center", va="center", fontsize=10, fontweight="bold",
         transform=fig.transFigure,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="#ccc"))

out_path = os.path.join(OUT, "k_isolation_comparison.png")
fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print(f"Saved: {out_path}")
