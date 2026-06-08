#!/usr/bin/env python3
"""Redesigned scale sweep comparison: larger images, larger fonts, 6-col layout."""
import os, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJ, "final_experiments")
FE = os.path.join(OUT_DIR)
SS = os.path.join(FE, "scale_sweep")
GC = os.path.join(FE, "grad_cache")
TP = os.path.join(PROJ, "testdata", "pairs")

PATHS = {
    "Mild": [
        ("LQ\n(PSNR=31.2)",    os.path.join(TP, "LQ", "0000_00010.png")),
        ("GT",                  os.path.join(TP, "GT", "0000_00010.png")),
        ("Paper\nK=1  s=0.10",  os.path.join(FE, "exp_A0_true_paper", "0000_00010.png")),
        ("K=2  s=0.10",        os.path.join(SS, "K2_s10_mild.png")),
        ("K=2  s=0.15",        os.path.join(FE, "NIQE_K2_s15", "0000_00010.png")),
        ("K=2  s=0.20",        os.path.join(SS, "K2_s20_mild.png")),
    ],
    "Severe": [
        ("LQ\n(PSNR=22.2)",    os.path.join(TP, "LQ", "0006_00092.png")),
        ("GT",                  os.path.join(TP, "GT", "0006_00092.png")),
        ("Paper\nK=1  s=0.10",  os.path.join(GC, "A0_paper.png")),
        ("K=2  s=0.10",        os.path.join(SS, "K2_s10_severe.png")),
        ("K=2  s=0.15",        os.path.join(GC, "A1_K2_s15.png")),
        ("K=2  s=0.20",        os.path.join(FE, "NIQE_K2_s20", "0006_00092.png")),
    ],
}

METRICS = {
    ("Mild", "Paper\nK=1  s=0.10"): (26.47, 756, True),
    ("Mild", "K=2  s=0.10"):       (25.04, 792, False),
    ("Mild", "K=2  s=0.15"):       (25.95, 756, True),
    ("Mild", "K=2  s=0.20"):       (26.50, 757, False),
    ("Severe", "Paper\nK=1  s=0.10"): (22.68, 778, True),
    ("Severe", "K=2  s=0.10"):       (22.88, 775, True),
    ("Severe", "K=2  s=0.15"):       (22.76, 777, True),
    ("Severe", "K=2  s=0.20"):       (22.69, 779, False),
}

NCOLS = 6
NROWS = 2

fig = plt.figure(figsize=(3.0 * NCOLS, 8.0))
gs = gridspec.GridSpec(NROWS, NCOLS, figure=fig,
                       left=0.08, right=0.97, top=0.93, bottom=0.06,
                       hspace=0.06, wspace=0.04)

def load_img(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

for i_level, (level, entries) in enumerate(PATHS.items()):
    for col, (label, path) in enumerate(entries):
        ax = fig.add_subplot(gs[i_level, col])
        img = load_img(path)
        if img is not None:
            ax.imshow(img)
        ax.axis("off")

        title = label.replace("\n", " ")
        ax.set_title(title, fontsize=12, fontweight="bold" if col <= 2 else "normal", pad=8)

        key = (level, label)
        if key in METRICS:
            psnr, niqe, is_best = METRICS[key]
            anno = f"PSNR = {psnr:.2f}     NIQE = {niqe:.0f}"
            if is_best:
                anno += "   [BEST]"
            bc = "#d4edda" if is_best else "#f8f9fa"
            ax.text(0.5, 0.015, anno, ha="center", va="bottom",
                    transform=ax.transAxes, fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor=bc,
                              edgecolor="#888", alpha=0.92, linewidth=1.2))

# Side labels
fig.text(0.01, 0.72, "Mild\nDegradation", fontsize=14, fontweight="bold",
         va="center", ha="left", rotation=90,
         bbox=dict(boxstyle="round,pad=0.2", facecolor="#e3f2fd", edgecolor="#90caf9"))

fig.text(0.01, 0.28, "Severe\nDegradation", fontsize=14, fontweight="bold",
         va="center", ha="left", rotation=90,
         bbox=dict(boxstyle="round,pad=0.2", facecolor="#fce4ec", edgecolor="#f48fb1"))

fig.suptitle("PGDiff: K=2 Guidance Scale Sweep  —  Mild vs Severe Degradation",
             fontsize=17, fontweight="bold", y=0.98)

out_path = os.path.join(OUT_DIR, "scale_sweep_comparison.png")
fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
print(f"Saved: {out_path}")
