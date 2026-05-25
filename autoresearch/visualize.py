"""
Visualize baseline vs best experiment results for PGDiff autoresearch.
"""
import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_srcdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _srcdir not in sys.path:
    sys.path.insert(0, _srcdir)

sys.path.insert(0, os.path.join(_srcdir, "autoresearch"))
from prepare import load_image, save_image

# Data
baseline = {
    "name": "Baseline (MSE + no extras)",
    "quality_score": 5.865,
    "avg_loss": 637606.927,
    "sharpness": 108717.554,
    "loss_per_image": 53133.911,
    "color": "#e74c3c",
}
best = {
    "name": "Best (Huber + Edge + Schedule + Momentum)",
    "quality_score": 1.484,
    "avg_loss": 161604.217,
    "sharpness": 108917.506,
    "loss_per_image": 13467.018,
    "color": "#27ae60",
}

out_dir = "results/comparison"
os.makedirs(out_dir, exist_ok=True)

# ---- 1. Bar charts ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("PGDiff Autoresearch: Baseline vs Best Configuration", fontsize=18, fontweight="bold", y=0.98)

metrics = [
    ("quality_score", "Quality Score (lower = better)", True),
    ("avg_loss", "Avg Guidance Loss (lower = better)", True),
    ("sharpness", "Avg Sharpness (higher = better)", False),
    ("loss_per_image", "Loss per Image (lower = better)", True),
]

for ax, (key, title, lower_better) in zip(axes.flat, metrics):
    vals = [baseline[key], best[key]]
    labels = ["Baseline", "Best"]
    colors = [baseline["color"], best["color"]]
    bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3)

    # Improvement annotation
    if lower_better:
        pct = (baseline[key] - best[key]) / baseline[key] * 100
        sign = "↓"
    else:
        pct = (best[key] - baseline[key]) / baseline[key] * 100
        sign = "↑"
    ax.annotate(f"Improvement: {sign} {pct:.1f}%",
                xy=(0.5, 0.92), xycoords="axes fraction",
                ha="center", fontsize=11, fontweight="bold",
                color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1", edgecolor="#bdc3c7"))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(out_dir, "metrics_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"[1/3] Metrics bar chart saved to {out_dir}/metrics_comparison.png")

# ---- 2. Side-by-side image comparisons ----
img_names = sorted(os.listdir("results/baseline_outputs"))[:6]  # first 6

fig = plt.figure(figsize=(18, 3 * len(img_names)))
gs = gridspec.GridSpec(len(img_names), 3, figure=fig, hspace=0.15, wspace=0.05)
fig.suptitle("Per-Image Comparison: Baseline vs Best", fontsize=16, fontweight="bold", y=0.98)

titles = ["Input (LQ)", "Baseline (MSE)", "Best (Huber + Edge + Schedule + Momentum)"]

for row, img_name in enumerate(img_names):
    # Load input image
    input_path = os.path.join("testdata/cropped_faces", img_name)
    if os.path.exists(input_path):
        img_input = cv2.imread(input_path)
        img_input_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    else:
        img_input_rgb = np.zeros((512, 512, 3), dtype=np.uint8)

    # Load baseline output
    base_path = os.path.join("results/baseline_outputs", img_name)
    if os.path.exists(base_path):
        img_base = cv2.imread(base_path)
        img_base_rgb = cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB)
    else:
        img_base_rgb = np.zeros((512, 512, 3), dtype=np.uint8)

    # Load best output
    best_path = os.path.join("results/best_outputs", img_name)
    if os.path.exists(best_path):
        img_best = cv2.imread(best_path)
        img_best_rgb = cv2.cvtColor(img_best, cv2.COLOR_BGR2RGB)
    else:
        img_best_rgb = np.zeros((512, 512, 3), dtype=np.uint8)

    for col, (img, title) in enumerate(zip(
        [img_input_rgb, img_base_rgb, img_best_rgb],
        titles
    )):
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        ax.set_title(title if row == 0 else "", fontsize=11, fontweight="bold")
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(img_name, fontsize=9, fontweight="bold", rotation=0, labelpad=15, va="center")

plt.savefig(os.path.join(out_dir, "image_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"[2/3] Image comparison saved to {out_dir}/image_comparison.png")

# ---- 3. Summary table as image ----
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis("off")

table_data = [
    ["Metric", "Baseline (MSE)", "Best (Huber + all extras)", "Improvement"],
    ["Quality Score ↓", f"{baseline['quality_score']:.3f}", f"{best['quality_score']:.3f}",
     f"↓ {(baseline['quality_score'] - best['quality_score']) / baseline['quality_score'] * 100:.1f}%"],
    ["Avg Guidance Loss ↓", f"{baseline['avg_loss']:.1f}", f"{best['avg_loss']:.1f}",
     f"↓ {(baseline['avg_loss'] - best['avg_loss']) / baseline['avg_loss'] * 100:.1f}%"],
    ["Avg Sharpness ↑", f"{baseline['sharpness']:.1f}", f"{best['sharpness']:.1f}",
     f"↑ {(best['sharpness'] - baseline['sharpness']) / baseline['sharpness'] * 100:.2f}%"],
    ["Loss per Image ↓", f"{baseline['loss_per_image']:.1f}", f"{best['loss_per_image']:.1f}",
     f"↓ {(baseline['loss_per_image'] - best['loss_per_image']) / baseline['loss_per_image'] * 100:.1f}%"],
]

table = ax.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(13)
table.scale(1, 2.5)

# Style header
for j in range(4):
    cell = table[0, j]
    cell.set_facecolor("#2c3e50")
    cell.set_text_props(color="white", fontweight="bold")

# Color improvement column
for i in range(1, 5):
    cell = table[i, 3]
    txt = table_data[i][3]
    if "↓" in txt:
        pct = float(txt.replace("↓ ", "").replace("%", ""))
        if pct > 20:
            cell.set_facecolor("#d5f5e3")  # green
        elif pct > 0:
            cell.set_facecolor("#fef9e7")  # yellow
        else:
            cell.set_facecolor("#fadbd8")  # red
    elif "↑" in txt:
        pct = float(txt.replace("↑ ", "").replace("%", ""))
        if pct > 0:
            cell.set_facecolor("#d5f5e3")  # green
        else:
            cell.set_facecolor("#fadbd8")  # red

ax.set_title("PGDiff Experiment: Baseline vs Best Configuration Summary",
             fontsize=15, fontweight="bold", pad=20)

plt.savefig(os.path.join(out_dir, "summary_table.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"[3/3] Summary table saved to {out_dir}/summary_table.png")

print(f"\nAll visualizations saved to {out_dir}/")
