#!/usr/bin/env python3
"""Run key configs and generate output images for visual comparison."""
import subprocess, sys, re, os, shutil

WORK = "/mnt/workspace/autoPGDiff/autoresearch"
TRAIN = os.path.join(WORK, "train.py")
BASE = os.path.join(WORK, "..")

# Save original train.py
shutil.copy(TRAIN, TRAIN + ".bak")

CONFIGS = [
    ("paper",    "论文原配 S_END=0.7 MSE",   "paper_baseline",
     "S_END = 0.7\nGUIDANCE_EARLY_STOP = False\nBLOCK_UNET_GRAD = False\nGUIDANCE_EVERY_K = 1\nGUIDANCE_SCALE = 0.10\nEDGE_WEIGHT = 0.0\nGRAD_MOMENTUM = 0.0"),
    ("optim",    "优化基线 S_END=0.5 Huber", "optim_baseline",
     "S_END = 0.5\nGUIDANCE_EARLY_STOP = True\nBLOCK_UNET_GRAD = True\nGUIDANCE_EVERY_K = 1\nGUIDANCE_SCALE = 0.10\nEDGE_WEIGHT = 0.0\nGRAD_MOMENTUM = 0.0"),
    ("k2_s020",  "K=2+scale=0.20",          "k2_sc020",
     "S_END = 0.5\nGUIDANCE_EARLY_STOP = True\nBLOCK_UNET_GRAD = True\nGUIDANCE_EVERY_K = 2\nGUIDANCE_SCALE = 0.20\nEDGE_WEIGHT = 0.0\nGRAD_MOMENTUM = 0.0"),
    ("k2_s010",  "K=2+scale=0.10 (最大细节)","k2_sc010",
     "S_END = 0.5\nGUIDANCE_EARLY_STOP = True\nBLOCK_UNET_GRAD = True\nGUIDANCE_EVERY_K = 2\nGUIDANCE_SCALE = 0.10\nEDGE_WEIGHT = 0.0\nGRAD_MOMENTUM = 0.0"),
    ("pure",     "纯扩散 (无引导上限)",      "pure_diff",
     "S_END = 0.0\nGUIDANCE_EARLY_STOP = False\nBLOCK_UNET_GRAD = True\nGUIDANCE_EVERY_K = 1\nGUIDANCE_SCALE = 0.0\nEDGE_WEIGHT = 0.0\nGRAD_MOMENTUM = 0.0"),
]

# Build fresh train.py from clean state each time
def build_config(overrides_text, is_paper):
    # Start from clean saved copy
    with open(TRAIN + ".bak", "r") as f:
        c = f.read()

    # Apply lines
    for line in overrides_text.strip().split("\n"):
        var, val = line.split(" = ", 1)
        if val in ("True", "False"):
            c = re.sub(rf'{var} = (True|False)', f'{var} = {val}', c)
        elif '"' in val:
            c = re.sub(rf'{var} = "[^"]*"', f'{var} = {val}', c)
        else:
            c = re.sub(rf'{var} = [\d.]+', f'{var} = {val}', c)

    # Set output dir
    outdir_name = overrides_text.strip().split("\n")[-1].split(" = ")[0]  # won't work
    # Just set it via the overrides

    # Loss function: MSE for paper, Huber for others
    if is_paper:
        c = c.replace('F.smooth_l1_loss(', 'F.mse_loss(')
        c = re.sub(r',\s*beta=\d+\.?\d*', '', c)

    # Common resets
    c = re.sub(r'S_START = [.\d]+', 'S_START = 1.0', c)
    c = re.sub(r'TIMESTEP_RESPACING = "[^"]*"', 'TIMESTEP_RESPACING = ""', c)
    c = re.sub(r'USE_DPMSOLVER = (True|False)', 'USE_DPMSOLVER = False', c)
    c = re.sub(r'USE_DDIM = (True|False)', 'USE_DDIM = False', c)
    c = re.sub(r'RESTORER_EMA = [.\d]+', 'RESTORER_EMA = 0.0', c)
    c = re.sub(r'USE_ANALYTICAL_GRAD = (True|False)', 'USE_ANALYTICAL_GRAD = False', c)
    c = re.sub(r'NO_GRAD_CLAMP = (True|False)', 'NO_GRAD_CLAMP = False', c)
    c = re.sub(r'MAX_IMAGES = \d+', 'MAX_IMAGES = 1', c)
    c = re.sub(r'IN_DIR = "[^"]*"', 'IN_DIR = "../testdata/cropped_faces"', c)
    c = re.sub(r'RUN_TAG = "[^"]*"', 'RUN_TAG = ""', c)

    return c

for name, desc, outdir_name, config_text in CONFIGS:
    is_paper = (name == "paper")
    outdir = f'"../autoresearch/results/{outdir_name}"'

    print(f"\n=== {name}: {desc} ===")

    c = build_config(config_text, is_paper)
    # Set output directory for this run
    c = re.sub(r'OUT_DIR = "[^"]*"', f'OUT_DIR = {outdir}', c)

    with open(TRAIN, "w") as f:
        f.write(c)

    # Verify syntax
    import py_compile
    try:
        py_compile.compile(TRAIN, doraise=True)
    except py_compile.PyCompileError as e:
        print(f"  SYNTAX ERROR: {e}")
        continue

    r = subprocess.run([sys.executable, TRAIN], capture_output=True, text=True,
                       cwd=WORK, timeout=600)

    if r.returncode == 0:
        out = r.stdout
        s = lambda p, d='0': re.search(p, out).group(1) if re.search(p, out) else d
        print(f"  V2={float(s(r'quality_score_v2:\s+([\d.]+)')):.4f}  "
              f"Sharp={float(s(r'sharpness_avg:\s+([\d.]+)')):.1f}  "
              f"Edge={float(s(r'edge_density_avg:\s+([\d.]+)')):.4f}  "
              f"SSIM={float(s(r'ssim_input_avg:\s+([\d.]+)')):.4f}  "
              f"GAIN={float(s(r'sharpness_gain_avg:\s+([\d.]+)')):.2f}  "
              f"RC={int(s(r'restorer_calls:\s+(\d+)', '0'))}")
    else:
        print(f"  CRASHED: {r.stderr[-200:]}")
