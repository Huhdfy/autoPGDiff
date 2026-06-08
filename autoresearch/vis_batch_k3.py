#!/usr/bin/env python3
"""Batch validation with K3_s225: 5 severe images, GT→LQ→Paper(K=1)→K3_s225."""
import os, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "final_experiments")
TP = os.path.join(PROJ, "testdata", "pairs")
BS = os.path.join(OUT, "batch_severe")

IMG_IDS = ["0005_00073", "0006_00092", "0007_00010", "0008_00016", "0009_00041"]

A0_MET = {  # (PSNR, NIQE)
    "0005_00073": (23.19, 1055),
    "0006_00092": (22.65, 778),
    "0007_00010": (21.05, 960),
    "0008_00016": (21.70, 1495),
    "0009_00041": (18.95, 1288),
}
K3_MET = {
    "0005_00073": (23.30, 1036),
    "0006_00092": (22.73, 775),
    "0007_00010": (21.86, 893),
    "0008_00016": (22.21, 1058),
    "0009_00041": (19.89, 1785),
}

LQ_PSNR = {"0005_00073": 22.40, "0006_00092": 22.21, "0007_00010": 21.32,
            "0008_00016": 20.64, "0009_00041": 18.97}

NROWS = len(IMG_IDS)
NCOLS = 4
fig = plt.figure(figsize=(14, 3.2 * NROWS + 1.2))
gs = gridspec.GridSpec(NROWS, NCOLS, figure=fig,
                       left=0.06, right=0.98, top=0.96, bottom=0.07,
                       hspace=0.06, wspace=0.03)

for col, title in enumerate(["GT", "LQ Input", "Paper (K=1 s=0.10)", "K3_s225 (K=3 s=0.225)"]):
    ax = fig.add_subplot(gs[0, col])
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.axis("off")

def load(p):
    img = cv2.imread(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

k3_wins_psnr = 0
k3_wins_niqe = 0

for row, img_id in enumerate(IMG_IDS):
    paths = [
        os.path.join(TP, "GT", f"{img_id}.png"),
        os.path.join(TP, "LQ", f"{img_id}.png"),
        os.path.join(BS, "A0_baseline", f"{img_id}.png"),
        os.path.join(OUT, "batch_severe_k3", f"{img_id}.png"),
    ]
    a0_p, a0_n = A0_MET[img_id]
    k3_p, k3_n = K3_MET[img_id]
    if k3_p >= a0_p: k3_wins_psnr += 1
    if k3_n <= a0_n: k3_wins_niqe += 1

    annos = [
        "",
        f"LQ PSNR={LQ_PSNR[img_id]:.2f}",
        f"PSNR={a0_p:.2f}  NIQE={a0_n:.0f}",
        f"PSNR={k3_p:.2f}  NIQE={k3_n:.0f}",
    ]

    for col, (path, anno) in enumerate(zip(paths, annos)):
        ax = fig.add_subplot(gs[row, col])
        img = load(path)
        if img is not None:
            ax.imshow(img)
        ax.axis("off")
        if anno:
            bc = "#f0f0f0"
            if col == 3:
                bc = "#e0ffe0" if (k3_p >= a0_p and k3_n <= a0_n) else "#fff3cd"
                if k3_p >= a0_p and k3_n > a0_n:
                    bc = "#ffe8e8"
            ax.text(0.5, 0.015, anno, ha="center", va="bottom",
                    transform=ax.transAxes, fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=bc,
                              edgecolor="#aaa", alpha=0.92))
        if col == 0:
            ax.set_ylabel(img_id, fontsize=11, fontweight="bold",
                          rotation=0, labelpad=38, va="center")

fig.suptitle("Batch Validation: 5 Severe Images — Paper(K=1) vs K3_s225(K=3 s=0.225)",
             fontsize=17, fontweight="bold", y=0.985)
fig.text(0.5, 0.01,
         f"K3_s225 PSNR wins on {k3_wins_psnr}/5, NIQE wins on {k3_wins_niqe}/5  |  "
         f"Avg PSNR 22.02 vs 21.51 (+0.51), NIQE 1113 vs 1115 (−2)  |  "
         "MSE+CONSTANT+BLEND=0+BLOCK_UNET, DDPM 1000",
         ha="center", va="center", fontsize=10, fontweight="bold",
         transform=fig.transFigure,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#ccc"))

out_path = os.path.join(OUT, "batch_severe_k3_comparison.png")
fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print(f"Saved: {out_path}")
print(f"K3_s225: PSNR wins {k3_wins_psnr}/5, NIQE wins {k3_wins_niqe}/5")
