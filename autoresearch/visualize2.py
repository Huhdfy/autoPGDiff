"""
Visualize full 1000-step DDPM: baseline vs best (from saved outputs).
"""
import os, cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASELINE_DIR = "results/full_ddpm_outputs"
BEST_DIR = "results/experiment"
INPUT_DIR = "testdata/cropped_faces"
OUT_DIR = "results/comparison_final"
os.makedirs(OUT_DIR, exist_ok=True)

baseline = {"name": "Baseline (Huber+Edge+Sched+Momentum)", "quality_score": 49.191, "avg_loss": 5413.498, "sharpness": 110.051, "color": "#e74c3c"}
best = {"name": "Best (Huber+Schedule only)", "quality_score": 36.988, "avg_loss": 4354.327, "sharpness": 117.722, "color": "#27ae60"}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("PGDiff Autoresearch @ Full 1000-step DDPM", fontsize=16, fontweight="bold", y=0.98)

metrics = [
    ("quality_score", "Quality Score (lower = better)", True),
    ("avg_loss", "Avg Guidance Loss (lower = better)", True),
    ("sharpness", "Avg Sharpness (higher = better)", False),
]
for idx, (ax, (key, title, lower_better)) in enumerate([(axes.flat[i], m) for i, m in enumerate(metrics)]):
    vals = [baseline[key], best[key]]
    labels = ["Baseline", "Best"]
    bars = ax.bar(labels, vals, color=[baseline["color"], best["color"]], width=0.5, edgecolor="black")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.02, f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    pct = (baseline[key]-best[key])/baseline[key]*100 if lower_better else (best[key]-baseline[key])/baseline[key]*100
    sign = "↓" if lower_better else "↑"
    ax.annotate(f"Improvement: {sign} {pct:.1f}%", xy=(0.5,0.92), xycoords="axes fraction", ha="center", fontsize=11, fontweight="bold", color="#2c3e50", bbox=dict(boxstyle="round", facecolor="#ecf0f1", edgecolor="#bdc3c7"))

axes.flat[2].remove()
axes.flat[3].remove()
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig(os.path.join(OUT_DIR, "metrics_final.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"[1/3] Saved to {OUT_DIR}/metrics_final.png")

img_names = sorted(os.listdir(BASELINE_DIR))[:6]
fig = plt.figure(figsize=(18, 3*len(img_names)))
gs = gridspec.GridSpec(len(img_names), 3, figure=fig, hspace=0.15, wspace=0.05)
titles = ["Input (LQ)", "Baseline", "Best"]
for row, name in enumerate(img_names):
    for col, (src, lb) in enumerate(zip([INPUT_DIR, BASELINE_DIR, BEST_DIR], titles)):
        path = os.path.join(src, name)
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) if os.path.exists(path) else np.zeros((512,512,3),np.uint8)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        if row == 0: ax.set_title(lb, fontsize=11, fontweight="bold")
        ax.axis("off")
        if col == 0: ax.set_ylabel(name, fontsize=9, rotation=0, labelpad=15, va="center")
plt.savefig(os.path.join(OUT_DIR, "images_final.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"[2/3] Saved to {OUT_DIR}/images_final.png")

fig, ax = plt.subplots(figsize=(14, 3))
ax.axis("off")
data = [
    ["Metric", "Baseline", "Best", "Improvement"],
    ["Quality Score ↓", f"{baseline['quality_score']:.3f}", f"{best['quality_score']:.3f}", f"↓ {(baseline['quality_score']-best['quality_score'])/baseline['quality_score']*100:.1f}%"],
    ["Guidance Loss ↓", f"{baseline['avg_loss']:.1f}", f"{best['avg_loss']:.1f}", f"↓ {(baseline['avg_loss']-best['avg_loss'])/baseline['avg_loss']*100:.1f}%"],
    ["Sharpness ↑", f"{baseline['sharpness']:.1f}", f"{best['sharpness']:.1f}", f"↑ {(best['sharpness']-baseline['sharpness'])/baseline['sharpness']*100:.2f}%"],
]
tbl = ax.table(cellText=data, loc="center", cellLoc="center", colWidths=[0.2,0.25,0.25,0.3])
tbl.auto_set_font_size(False); tbl.set_fontsize(13); tbl.scale(1, 2.5)
for j in range(4):
    tbl[0,j].set_facecolor("#2c3e50"); tbl[0,j].set_text_props(color="white", fontweight="bold")
for i in range(1,4):
    txt = data[i][3]
    if "↓" in txt and float(txt.replace("↓ ","").replace("%","")) > 0: tbl[i,3].set_facecolor("#d5f5e3")
    elif "↑" in txt and float(txt.replace("↑ ","").replace("%","")) > 0: tbl[i,3].set_facecolor("#d5f5e3")
plt.savefig(os.path.join(OUT_DIR, "summary_final.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"[3/3] Saved to {OUT_DIR}/summary_final.png")
print(f"\nAll vis saved to {OUT_DIR}/")
