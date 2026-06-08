#!/usr/bin/env python3
"""K=3 s=0.225 vs K=2 s=0.15 vs K=1 s=0.10 — iso-effective-guidance comparison."""
import os, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "final_experiments")

PATHS = {
    "Mild": [
        ("LQ\n(PSNR=31.2)",       os.path.join(PROJ, "testdata/pairs/LQ", "0000_00010.png")),
        ("GT",                     os.path.join(PROJ, "testdata/pairs/GT", "0000_00010.png")),
        ("K=1  s=0.10\n136s  1000 RC", os.path.join(OUT, "exp_A0_true_paper", "0000_00010.png")),
        ("K=2  s=0.15\n 98s   500 RC", os.path.join(OUT, "NIQE_K2_s15", "0000_00010.png")),
        ("K=3  s=0.225\n 85s   333 RC", os.path.join(OUT, "k3_s225", "K3_s225_mild.png")),
    ],
    "Severe": [
        ("LQ\n(PSNR=22.2)",       os.path.join(PROJ, "testdata/pairs/LQ", "0006_00092.png")),
        ("GT",                     os.path.join(PROJ, "testdata/pairs/GT", "0006_00092.png")),
        ("K=1  s=0.10\n136s  1000 RC", os.path.join(OUT, "grad_cache", "A0_paper.png")),
        ("K=2  s=0.15\n 98s   500 RC", os.path.join(OUT, "grad_cache", "A1_K2_s15.png")),
        ("K=3  s=0.225\n 85s   333 RC", os.path.join(OUT, "k3_s225", "K3_s225_severe.png")),
    ],
}

METRICS = {
    ("Mild", "K=1"):    (26.47, 756, 0.10, 136),
    ("Mild", "K=2"):    (25.95, 756, 0.15,  98),
    ("Mild", "K=3"):    (25.95, 758, 0.225, 85),
    ("Severe", "K=1"):  (22.68, 778, 0.10, 136),
    ("Severe", "K=2"):  (22.76, 777, 0.15,  98),
    ("Severe", "K=3"):  (22.76, 776, 0.225, 85),
}

NCOLS = 5
NROWS = 2

fig = plt.figure(figsize=(15, 8.5))
gs = gridspec.GridSpec(NROWS, NCOLS + 1, figure=fig,
                       left=0.10, right=0.97, top=0.92, bottom=0.13,
                       hspace=0.06, wspace=0.03,
                       width_ratios=[1]*NCOLS + [0.6])

def load(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

for i_level, (level, entries) in enumerate(PATHS.items()):
    for col, (label, path) in enumerate(entries):
        ax = fig.add_subplot(gs[i_level, col])
        img = load(path)
        if img is not None:
            ax.imshow(img)
        ax.axis("off")
        title = label.split("\n")[0]
        ax.set_title(label.replace("\n", "\n"), fontsize=11.5, fontweight="bold", pad=6)

        key = (level, title)
        if key in METRICS:
            psnr, niqe, scale, t = METRICS[key]
            eff = scale / ({"K=1":1,"K=2":2,"K=3":3}[title])
            anno = f"PSNR={psnr:.2f}  NIQE={niqe:.0f}\neff={eff:.3f}×"
            bc = "#d4edda" if niqe <= (756 if level == "Mild" else 778) else "#fff3cd"
            ax.text(0.5, 0.015, anno, ha="center", va="bottom",
                    transform=ax.transAxes, fontsize=10.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=bc,
                              edgecolor="#999", alpha=0.92, linewidth=1))

# Side labels
fig.text(0.025, 0.68, "Mild\nDegradation", fontsize=14, fontweight="bold",
         va="center", ha="left", rotation=90,
         bbox=dict(boxstyle="round,pad=0.2", facecolor="#e3f2fd", edgecolor="#90caf9"))
fig.text(0.025, 0.25, "Severe\nDegradation", fontsize=14, fontweight="bold",
         va="center", ha="left", rotation=90,
         bbox=dict(boxstyle="round,pad=0.2", facecolor="#fce4ec", edgecolor="#f48fb1"))

# Summary panel (right side)
ax_s = fig.add_subplot(gs[:, -1])
ax_s.axis("off")
table_lines = [
    " SUMMARY",
    "",
    "Config   K  s      eff",
    "Paper    1  0.10   1.00",
    "K2_s15   2  0.15   0.75",
    "K3_s225  3  0.225  0.75",
    "",
    " Mild     PSNR  NIQE",
    f" Paper   26.47  756",
    f" K2_s15  25.95  756",
    f" K3_s225 25.95  758",
    "",
    " Severe   PSNR  NIQE",
    f" Paper   22.68  778",
    f" K2_s15  22.76  777",
    f" K3_s225 22.76  776",
    "",
    "Time  136/98/85s",
    "RC   1000/500/333",
]
table_text = "\n".join(table_lines)
ax_s.text(0.5, 0.5, table_text, ha="center", va="center",
          fontsize=10, fontfamily="monospace", fontweight="bold",
          transform=ax_s.transAxes,
          bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa",
                    edgecolor="#ccc", linewidth=1))

fig.suptitle("PGDiff: Iso-Effective-Guidance K-Scale Comparison  (MSE + CONSTANT + BLEND=0 + BLOCK_UNET=1)",
             fontsize=16, fontweight="bold", y=0.97)
fig.text(0.5, 0.035, "DDPM 1000 steps, seed=1234  |  K=3@0.225 matches K=2@0.15 on PSNR/NIQE, saves −13% time & −33% restorer calls",
         ha="center", va="center", fontsize=10.5, fontweight="bold", transform=fig.transFigure,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#ccc"))

out_path = os.path.join(OUT, "k3_iso_guidance.png")
fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print(f"Saved: {out_path}")
