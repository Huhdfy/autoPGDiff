#!/usr/bin/env python3
"""Main results: LQ / Paper(K=1) / K2_s15 / K3_s225 / GT — Mild + Severe."""
import os, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "final_experiments")
TP = os.path.join(PROJ, "testdata", "pairs")

SOURCES = {
    "Mild": {
        "img_id": "0000_00010",
        "LQ":       os.path.join(TP, "LQ", "0000_00010.png"),
        "GT":       os.path.join(TP, "GT", "0000_00010.png"),
        "Paper\nK=1 s=0.10":   os.path.join(OUT, "exp_A0_true_paper", "0000_00010.png"),
        "K2_s15\nK=2 s=0.15":  os.path.join(OUT, "NIQE_K2_s15", "0000_00010.png"),
        "K3_s225\nK=3 s=0.225": os.path.join(OUT, "k3_s225", "K3_s225_mild.png"),
    },
    "Severe": {
        "img_id": "0006_00092",
        "LQ":       os.path.join(TP, "LQ", "0006_00092.png"),
        "GT":       os.path.join(TP, "GT", "0006_00092.png"),
        "Paper\nK=1 s=0.10":   os.path.join(OUT, "grad_cache", "A0_paper.png"),
        "K2_s15\nK=2 s=0.15":  os.path.join(OUT, "grad_cache", "A1_K2_s15.png"),
        "K3_s225\nK=3 s=0.225": os.path.join(OUT, "k3_s225", "K3_s225_severe.png"),
    },
}

METRICS = {
    ("Mild", "Paper"):    ("26.47 / 756", 136, 1000),
    ("Mild", "K2_s15"):   ("25.95 / 756",  97,  500),
    ("Mild", "K3_s225"):  ("25.95 / 758",  85,  333),
    ("Severe", "Paper"):  ("22.68 / 778", 136, 1000),
    ("Severe", "K2_s15"): ("22.76 / 777",  97,  500),
    ("Severe", "K3_s225"):("22.76 / 776",  85,  333),
}

COL_ORDER = ["LQ", "Paper\nK=1 s=0.10", "K2_s15\nK=2 s=0.15", "K3_s225\nK=3 s=0.225", "GT"]
NCOLS = 5
NROWS = 2

fig = plt.figure(figsize=(18, 8))
gs = gridspec.GridSpec(NROWS, NCOLS, figure=fig,
                       left=0.05, right=0.98, top=0.93, bottom=0.10,
                       hspace=0.06, wspace=0.03)

def load(p): 
    img = cv2.imread(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

for i_level, (level, sources) in enumerate(SOURCES.items()):
    for col, col_key in enumerate(COL_ORDER):
        ax = fig.add_subplot(gs[i_level, col])
        path = sources[col_key]
        img = load(path)
        if img is not None:
            ax.imshow(img)
        ax.axis("off")
        display = col_key.replace("\n", " ")
        ax.set_title(display, fontsize=12, fontweight="bold", pad=6)

        short_key = col_key.split("\n")[0]
        mkey = (level, short_key)
        if mkey in METRICS:
            psnr_niqe, t, rc = METRICS[mkey]
            anno = f"PSNR/NIQE = {psnr_niqe}"
            ax.text(0.5, 0.015, anno, ha="center", va="bottom",
                    transform=ax.transAxes, fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                              edgecolor="#aaa", alpha=0.92))

# Row labels
fig.text(0.008, 0.72, "Mild\n(0000_00010)", fontsize=14, fontweight="bold",
         va="center", ha="left", rotation=90,
         bbox=dict(boxstyle="round,pad=0.2", facecolor="#e3f2fd", edgecolor="#90caf9"))
fig.text(0.008, 0.28, "Severe\n(0006_00092)", fontsize=14, fontweight="bold",
         va="center", ha="left", rotation=90,
         bbox=dict(boxstyle="round,pad=0.2", facecolor="#fce4ec", edgecolor="#f48fb1"))

fig.suptitle("PGDiff: K=1 → K2_s15 → K3_s225  —  Main Results", fontsize=18, fontweight="bold", y=0.98)
fig.text(0.5, 0.035,
         "Mild: three configs visually near-identical, NIQE all ~756  |  "
         "Severe: K2_s15/K3_s225 match Paper quality, −28~38% time, −77% VRAM  |  "
         "MSE + CONSTANT + BLEND=0 + BLOCK_UNET, DDPM 1000",
         ha="center", va="center", fontsize=10, fontweight="bold", transform=fig.transFigure,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#ccc"))

out_path = os.path.join(OUT, "main_results.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print(f"Saved: {out_path}")
