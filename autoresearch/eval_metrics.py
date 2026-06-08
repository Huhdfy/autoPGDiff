#!/usr/bin/env python3
"""Evaluate key configurations with raw no-reference metrics."""
import subprocess, sys, re, os

WORK = "/mnt/workspace/autoPGDiff/autoresearch"
TRAIN = os.path.join(WORK, "train.py")

CONFIGS = [
    ("baseline",    "K=1 s=0.5 sc=0.10", {"S_END": "0.5", "GUIDANCE_EVERY_K": "1", "GUIDANCE_SCALE": "0.10"}),
    ("k2_sc020",    "K=2 s=0.5 sc=0.20", {"S_END": "0.5", "GUIDANCE_EVERY_K": "2", "GUIDANCE_SCALE": "0.20"}),
    ("k2_sc010",    "K=2 s=0.5 sc=0.10", {"S_END": "0.5", "GUIDANCE_EVERY_K": "2", "GUIDANCE_SCALE": "0.10"}),
    ("sc005",       "K=1 s=0.5 sc=0.05", {"S_END": "0.5", "GUIDANCE_EVERY_K": "1", "GUIDANCE_SCALE": "0.05"}),
    ("sc020",       "K=1 s=0.5 sc=0.20", {"S_END": "0.5", "GUIDANCE_EVERY_K": "1", "GUIDANCE_SCALE": "0.20"}),
    ("s03",         "K=1 s=0.3 sc=0.10", {"S_END": "0.3", "GUIDANCE_EVERY_K": "1", "GUIDANCE_SCALE": "0.10"}),
    ("edge002",     "K=1 s=0.5 sc=0.10 edge=0.02", {"S_END": "0.5", "GUIDANCE_EVERY_K": "1", "GUIDANCE_SCALE": "0.10", "EDGE_WEIGHT": "0.02"}),
    ("no_guide",    "scale=0 pure diffusion", {"S_END": "0.0", "GUIDANCE_SCALE": "0.0", "GUIDANCE_EVERY_K": "1", "GUIDANCE_EARLY_STOP": "False"}),
    ("dpm100",      "DPM 100 steps sc=0.10", {"USE_DPMSOLVER": "True", "GUIDANCE_EARLY_STOP": "False", "S_END": "0.0"}),
]

def set_config(overrides):
    with open(TRAIN, "r") as f:
        c = f.read()

    for var, val in overrides.items():
        if val in ("True", "False"):
            c = re.sub(rf'{var} = (True|False)', f'{var} = {val}', c)
        elif val.startswith('"'):
            c = re.sub(rf'{var} = "[^"]*"', f'{var} = {val}', c)
        else:
            c = re.sub(rf'{var} = [\d.]+', f'{var} = {val}', c)

    # Reset commonly modified
    c = re.sub(r'USE_DPMSOLVER = (True|False)', 'USE_DPMSOLVER = False', c)
    c = re.sub(r'USE_DDIM = (True|False)', 'USE_DDIM = False', c)
    c = re.sub(r'TIMESTEP_RESPACING = "[^"]*"', 'TIMESTEP_RESPACING = ""', c)
    c = re.sub(r'EDGE_WEIGHT = [.\d]+', 'EDGE_WEIGHT = 0.0', c)
    c = re.sub(r'GRAD_MOMENTUM = [.\d]+', 'GRAD_MOMENTUM = 0.0', c)
    c = re.sub(r'S_START = [.\d]+', 'S_START = 1.0', c)
    c = re.sub(r'GUIDANCE_EARLY_STOP = (True|False)', 'GUIDANCE_EARLY_STOP = True', c)
    c = re.sub(r'GUIDANCE_EVERY_K = \d+', 'GUIDANCE_EVERY_K = 1', c)
    c = re.sub(r'GUIDANCE_SCALE = [.\d]+', 'GUIDANCE_SCALE = 0.10', c)
    c = re.sub(r'S_END = [.\d]+', 'S_END = 0.5', c)
    c = re.sub(r'MAX_IMAGES = \d+', 'MAX_IMAGES = 1', c)
    c = re.sub(r'RUN_TAG = "[^"]*"', 'RUN_TAG = ""', c)
    c = re.sub(r'IN_DIR = "[^"]*"', 'IN_DIR = "../testdata/cropped_faces"', c)
    c = re.sub(r'OUT_DIR = "[^"]*"', 'OUT_DIR = "../results/experiment"', c)
    c = re.sub(r'RESTORER_EMA = [.\d]+', 'RESTORER_EMA = 0.0', c)
    c = re.sub(r'USE_ANALYTICAL_GRAD = (True|False)', 'USE_ANALYTICAL_GRAD = False', c)

    # Now apply the specific overrides
    for var, val in overrides.items():
        if val in ("True", "False"):
            c = re.sub(rf'{var} = (True|False)', f'{var} = {val}', c)
        elif val.startswith('"'):
            c = re.sub(rf'{var} = "[^"]*"', f'{var} = {val}', c)
        else:
            c = re.sub(rf'{var} = [\d.]+', f'{var} = {val}', c)

    with open(TRAIN, "w") as f:
        f.write(c)

def run(name, desc):
    log = os.path.join(WORK, f"eval_{name}.log")
    print(f"\n=== {name}: {desc} ===")
    r = subprocess.run([sys.executable, TRAIN], capture_output=True, text=True, cwd=WORK, timeout=600)
    with open(log, "w") as f:
        f.write(r.stdout + r.stderr)

    output = r.stdout
    def g(pat, d="0"):
        m = re.search(pat, output)
        return m.group(1) if m else d

    return {
        "name": name,
        "v2": float(g(r"quality_score_v2:\s+([\d.]+)")),
        "ssim": float(g(r"ssim_input_avg:\s+([\d.]+)")),
        "nat": float(g(r"naturalness_avg:\s+([\d.]+)")),
        "gain": float(g(r"sharpness_gain_avg:\s+([\d.]+)")),
        "sharp": float(g(r"sharpness_avg:\s+([\d.]+)")),
        "edge_d": float(g(r"edge_density_avg:\s+([\d.]+)")),
        "lvarme": float(g(r"local_var_mean_avg:\s+([\d.]+)")),
        "spi": float(g(r"s_per_image:\s+([\d.]+)")),
        "gc": int(g(r"guidance_calls:\s+(\d+)", "0")),
        "rc": int(g(r"restorer_calls:\s+(\d+)", "0")),
    }

results = []
for name, desc, overrides in CONFIGS:
    set_config(overrides)
    r = run(name, desc)
    results.append(r)
    print(f"  V2={r['v2']:.4f}  SSIM={r['ssim']:.4f}  NAT={r['nat']:.4f}  "
          f"SHARP={r['sharp']:.1f}  GAIN={r['gain']:.1f}  "
          f"ED={r['edge_d']:.4f}  LV={r['lvarme']:.0f}  "
          f"Time={r['spi']:.1f}s  RC={r['rc']}")

print(f"\n\n{'='*110}")
print("  RAW METRICS COMPARISON  (no composite score, pure measurements)")
print(f"{'='*110}")
print(f"{'Config':<18s} {'V2(old)':>8s} {'Sharp':>8s} {'GAIN':>7s} {'Edge_d':>8s} {'Var':>7s} {'NAT':>7s} {'SSIM2LQ':>8s} {'Time':>7s} {'RC':>5s}")
print("-"*110)

for r in results:
    print(f"{r['name']:<18s} {r['v2']:8.4f} {r['sharp']:8.1f} {r['gain']:7.2f} "
          f"{r['edge_d']:8.4f} {r['lvarme']:7.0f} {r['nat']:7.4f} {r['ssim']:8.4f} "
          f"{r['spi']:7.1f} {r['rc']:5d}")

# Which metrics actually vary meaningfully?
sharps = [r['sharp'] for r in results]
gains = [r['gain'] for r in results]
eds = [r['edge_d'] for r in results]
lvs = [r['lvarme'] for r in results]
nats = [r['nat'] for r in results]
ssims = [r['ssim'] for r in results]

print(f"\n  Metric variation (coefficient of variation = std/mean):")
for name, vals in [("sharpness", sharps), ("gain", gains), ("edge_density", eds),
                     ("local_var", lvs), ("naturalness", nats), ("ssim_to_LQ", ssims)]:
    import numpy as np
    vals = np.array(vals)
    cv = np.std(vals) / (np.mean(vals) + 1e-8)
    print(f"    {name:<18s}:  mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  CV={cv:.3f}  "
          f"range=[{np.min(vals):.3f}, {np.max(vals):.3f}]")
