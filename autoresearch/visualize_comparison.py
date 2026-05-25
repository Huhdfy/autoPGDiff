"""
Generate visual comparison of output images from 3 experiments.
Produces a 2x3 grid: 1 row showing outputs, 1 row with cropped details.
"""
import os, re, cv2
import numpy as np

RESULTS = "../results"
EXPS = [
    ("exp_baseline",   "Baseline\n(no optimization)",          "#FF6B6B"),
    ("exp_grad_block", "Grad Block Only\n(restorer outside)",  "#4ECDC4"),
    ("exp_dpmsolver",  "DPM-Solver-2 Only\n(50 steps)",       "#45B7D1"),
]
LOG_FILES = ["run_baseline.log", "run_grad_block.log", "run_dpmsolver.log"]

IMG_NAME = "0040.png"

def parse_metrics(log_path):
    m = {}
    if not os.path.exists(log_path):
        return m
    with open(log_path) as f:
        for line in f:
            ma = re.match(r"^(\w[\w_]+):\s+([\d.]+)", line)
            if ma:
                m[ma.group(1)] = ma.group(2)
    return m

# Collect metrics
all_metrics = [parse_metrics(os.path.join(os.path.dirname(__file__), lf)) for lf in LOG_FILES]

# Load images
imgs = []
for exp_dir, _, _ in EXPS:
    p = os.path.join(RESULTS, exp_dir, IMG_NAME)
    im = cv2.imread(p)
    if im is None:
        print(f"WARN: missing {p}")
        im = np.zeros((512, 512, 3), dtype=np.uint8)
    imgs.append(im)

h, w = imgs[0].shape[:2]
pad = 15

# ── Build composite ──
# Top row: output images, Bottom row: center crop zoom
crop_sz = 200
cx, cy = w // 2, h // 2

col_w = w + pad
row_h = h + 180  # extra space below for metrics table

canvas_w = col_w * 3 + pad
canvas_h = row_h * 2 + pad
canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 245

font = cv2.FONT_HERSHEY_SIMPLEX

for col in range(3):
    img = imgs[col]
    x0 = col * col_w + pad

    # ── Top row: full image ──
    canvas[pad:pad + h, x0:x0 + w] = img

    # ── Bottom row: zoomed center crop ──
    crop = img[cy - crop_sz // 2:cy + crop_sz // 2, cx - crop_sz // 2:cx + crop_sz // 2]
    crop_big = cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)
    y0 = row_h + pad
    canvas[y0:y0 + h, x0:x0 + w] = crop_big

    # ── Label above top-left corner ──
    _, label, color_hex = EXPS[col]
    color = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))  # BGR
    cv2.putText(canvas, label.replace('\n', ' - '), (x0 + 8, pad + 28),
                font, 0.55, (0, 0, 0), 2)
    cv2.putText(canvas, label.replace('\n', ' - '), (x0 + 8, pad + 28),
                font, 0.55, color, 1)

    # ── Region indicator ──
    cv2.rectangle(canvas, (x0 + 8, pad + h - 30), (x0 + 180, pad + h - 6),
                  (0, 0, 0), -1)
    cv2.putText(canvas, "  center crop zone", (x0 + 12, pad + h - 10),
                font, 0.45, (255, 255, 255), 1)

    # ── Zoom label ──
    cv2.putText(canvas, "2x zoom", (x0 + 8, y0 + 24),
                font, 0.5, (100, 100, 100), 1)


# ── Title ──
cv2.putText(canvas, "PGDiff: 3-Way Comparison — Output Image Quality",
            (pad, canvas_h - 8), font, 0.7, (0, 0, 0), 2)

# ── Save full-image comparison ──
out_full = os.path.join(RESULTS, "comparison", "image_quality_comparison.png")
os.makedirs(os.path.dirname(out_full), exist_ok=True)
cv2.imwrite(out_full, canvas)
print(f"Saved: {out_full}")

# ── Create separate metrics table image ──
summary_h = 520
summary_w = 850
summary = np.ones((summary_h, summary_w, 3), dtype=np.uint8) * 255
y = 35
cv2.putText(summary, "PGDiff Experiment Comparison — Speed & Quality", (25, y),
            font, 0.75, (0, 0, 100), 2)
y += 15
cv2.line(summary, (20, y), (summary_w - 20, y), (200, 200, 200), 1)
y += 20

headers = ["Metric", "Baseline", "Grad Block", "DPM-Solver-2"]
col_x = [25, 250, 420, 590]
w_max = [200, 150, 150, 200]
for j, hdr in enumerate(headers):
    cv2.putText(summary, hdr, (col_x[j], y), font, 0.6, (0, 0, 0), 2)
y += 10
cv2.line(summary, (20, y), (summary_w - 20, y), (200, 200, 200), 1)
y += 20

rows = [
    ("quality_score",      "{:.3f}",  None),
    ("avg_guidance_loss",  "{:.1f}",  None),
    ("sharpness_avg",      "{:.1f}",  None),
    ("total_seconds",      "{:.1f}s", "lower"),
    ("peak_vram_mb",       "{:.0f} MB","lower"),
    ("guidance_calls",     "{}",      "lower"),
    ("s_per_image",        "{:.2f}s", "lower"),
    ("ms_per_guidance",    "{:.2f}ms","lower"),
]

for rname, rfmt, better in rows:
    is_speed = rname in ("total_seconds", "s_per_image", "ms_per_guidance", "peak_vram_mb", "guidance_calls")
    cv2.putText(summary, rname, (col_x[0], y),
                font, 0.5, (0, 0, 0) if not is_speed else (80, 80, 180), 1)
    vals = []
    for j, m in enumerate(all_metrics):
        val = m.get(rname, "?")
        try:
            val_f = float(val)
            val = rfmt.format(val_f)
        except:
            pass
        vals.append((val, val_f if isinstance(val, str) else float(m.get(rname, 0))))
        cv2.putText(summary, str(val), (col_x[j], y),
                    font, 0.5, (0, 0, 0), 1)

    # Highlight best value
    if better and len(vals) == 3 and all(v[1] != "?" for v in vals):
        num_vals = [v[1] for v in vals]
        if better == "lower":
            best_idx = np.argmin(num_vals)
        else:
            best_idx = np.argmax(num_vals)
        bx, by = col_x[best_idx], y - 15
        cv2.rectangle(summary, (bx - 5, by), (bx + 160, by + 22), (200, 255, 200), 2)
    y += 28

y += 10
cv2.line(summary, (20, y), (summary_w - 20, y), (200, 200, 200), 1)
y += 20

# Observations
notes = [
    "Observations:",
    "  * Grad Block: same quality as baseline, VRAM 4346->1019 MB (-77%)",
    "  * DPM-Solver-2: 8.4x faster (138s -> 16s), sharpness 30->14686",
    "  * DPM-Solver requires more VRAM (7941 MB) due to enable_grad + dual NFE",
    "  * Best combo: Grad Block + DPM-Solver = fast + low VRAM",
]
for note in notes:
    cv2.putText(summary, note, (25, y), font, 0.5, (50, 50, 50) if not note.startswith("Observ") else (0, 0, 0), 1)
    y += 22

out_metrics = os.path.join(RESULTS, "comparison", "metrics_comparison_v2.png")
cv2.imwrite(out_metrics, summary)
print(f"Saved: {out_metrics}")
