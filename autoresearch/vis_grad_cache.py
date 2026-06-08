#!/usr/bin/env python3
"""Generate gradient-cache experiment comparison visualization."""
import os, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJ, "final_experiments")
GC = os.path.join(OUT_DIR, "grad_cache")
TP = os.path.join(PROJ, "testdata", "pairs")

NROWS = 3
NCOLS = 5
TEST_IMG = "0006_00092"

fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(NROWS + 1, NCOLS, figure=fig,
                       left=0.02, right=0.98, top=0.93, bottom=0.07,
                       hspace=0.25, wspace=0.03,
                       height_ratios=[0.05, 1, 1, 1])

def load(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

# Row 0: column headers
COL_NAMES = ["LQ", "GT", "Baseline\nK=1 s=0.10", "K=2 s=0.15\n(no cache)", "K=2 s=0.15\n(cache)"]
for col, name in enumerate(COL_NAMES):
    ax = fig.add_subplot(gs[0, col])
    ax.set_title(name, fontsize=10, fontweight="bold", pad=4)
    ax.axis("off")

# Row 1: A group (baselines)
group_A = [
    ("LQ", os.path.join(TP, "LQ", f"{TEST_IMG}.png"), "", ""),
    ("GT", os.path.join(TP, "GT", f"{TEST_IMG}.png"), "", ""),
    ("Paper", os.path.join(GC, "A0_paper.png"), "22.68", "778"),
    ("K2_s15", os.path.join(GC, "A1_K2_s15.png"), "22.76", "777"),
    ("", "", "", ""),
]
for col, (label, path, psnr, niqe) in enumerate(group_A):
    ax = fig.add_subplot(gs[1, col])
    img = load(path) if path else None
    if img is not None:
        ax.imshow(img)
    ax.axis("off")
    if label:
        ax.set_title(label, fontsize=9, pad=2)
    if psnr:
        anno = f"PSNR={psnr}  NIQE={niqe}"
        bc = "#e0ffe0" if float(niqe) < 778 else "#ffe0e0"
        ax.text(0.5, 0.015, anno, ha="center", va="bottom", transform=ax.transAxes,
                fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=bc, edgecolor="#aaa", alpha=0.9))

# Row 2: K=2 cache vs no-cache (B group)
B_DATA = [
    ("B1\nK=2 s=0.10\ncache", load(os.path.join(GC, "B1_K2_s10_cache.png")), "22.68", "779"),
    ("A0 paper\n(K=1 s=0.10)", None, "22.68", "778"),
    ("B2\nK=2 s=0.15\ncache", load(os.path.join(GC, "B2_K2_s15_cache.png")), "22.32", "829"),
    ("A1 K2_s15\n(no cache)", None, "22.76", "777"),
    ("B3\nK=2 s=0.20\ncache", load(os.path.join(GC, "B3_K2_s20_cache.png")), "21.95", "902"),
]
for col, (label, img, psnr, niqe) in enumerate(B_DATA):
    ax = fig.add_subplot(gs[2, col])
    if img is not None:
        ax.imshow(img)
    ax.axis("off")
    ax.set_title(label, fontsize=8, pad=2)
    bc = "#ffe0e0"
    ax.text(0.5, 0.015, f"PSNR={psnr}  NIQE={niqe}", ha="center", va="bottom",
            transform=ax.transAxes, fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=bc, edgecolor="#aaa", alpha=0.9))

# Row 3: K=3 and K=5 cache vs no-cache
C_D_DATA = [
    ("C1\nK=3 s=0.15\nno-cache", load(os.path.join(GC, "C1_K3_s15.png")), "22.88", "775"),
    ("C2\nK=3 s=0.15\ncache", load(os.path.join(GC, "C2_K3_s15_cache.png")), "22.32", "828"),
    ("C3\nK=3 s=0.20\ncache", load(os.path.join(GC, "C3_K3_s20_cache.png")), "21.95", "905"),
    ("D1\nK=5 s=0.15\nno-cache", load(os.path.join(GC, "D1_K5_s15.png")), "22.64", "774"),
    ("D2\nK=5 s=0.15\ncache", load(os.path.join(GC, "D2_K5_s15_cache.png")), "22.32", "827"),
]
for col, (label, img, psnr, niqe) in enumerate(C_D_DATA):
    ax = fig.add_subplot(gs[3, col])
    if img is not None:
        ax.imshow(img)
    ax.axis("off")
    ax.set_title(label, fontsize=8, pad=2)
    bc = "#ffe0e0" if "cache" in label and "no-cache" not in label else "#e0ffe0"
    ax.text(0.5, 0.015, f"PSNR={psnr}  NIQE={niqe}", ha="center", va="bottom",
            transform=ax.transAxes, fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=bc, edgecolor="#aaa", alpha=0.9))

# Row labels
for row_idx, row_label in enumerate(["A: Baselines", "B: K=2 Cache", "C-D: K=3/5 Cache"]):
    y = 1.0 - (row_idx + 1) * 0.28
    fig.text(0.002, y, row_label, fontsize=11, fontweight="bold",
             va="center", ha="left", rotation=90,
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#ccc"))

fig.suptitle("Gradient Cache Experiment: Reusing Previous Gradient on Skip-Steps",
             fontsize=15, fontweight="bold", y=0.975)
fig.text(0.5, 0.01, "RED = worse than no-cache counterpart  |  GREEN = better/same  |  "
         "All: MSE+CONSTANT+BLEND=0+BLOCK_UNET, DDPM 1000, img=0006_00092",
         ha="center", va="center", fontsize=8.5, fontweight="bold",
         transform=fig.transFigure,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="#ccc"))

out_path = os.path.join(OUT_DIR, "grad_cache_comparison.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print(f"Saved: {out_path}")
