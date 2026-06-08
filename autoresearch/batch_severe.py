#!/usr/bin/env python3
"""Batch test A0 (baseline) and K2_s15 (best) on 5 severe images."""
import os, sys, re, time, subprocess, numpy as np

WORK = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(WORK, "train.py")
OUT_ROOT = os.path.join(os.path.dirname(WORK), "final_experiments", "batch_severe")
os.makedirs(OUT_ROOT, exist_ok=True)

# ── Common configs ──
A0_CONFIG = {
    "IN_DIR": '"../testdata/temp_multi/LQ"', "GT_DIR": '"../testdata/temp_multi/GT"',
    "MAX_IMAGES": "0", "SEED": "1234",
    "USE_HUBER_LOSS": "False", "USE_L1_LOSS": "False",
    "CONSTANT_SCHEDULE": "True", "GUIDANCE_EARLY_STOP": "False",
    "BLOCK_UNET_GRAD": "False", "USE_DPMSOLVER": "False", "USE_DDIM": "False",
    "HYBRID_MODE": "False", "TIMESTEP_RESPACING": '""',
    "S_END": "1.0", "GUIDANCE_EVERY_K": "1",
    "GUIDANCE_SCALE": "0.10", "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0",
    "RESIDUAL_BLEND": "0.0",
}

K2S15_CONFIG = {
    "IN_DIR": '"../testdata/temp_multi/LQ"', "GT_DIR": '"../testdata/temp_multi/GT"',
    "MAX_IMAGES": "0", "SEED": "1234",
    "USE_HUBER_LOSS": "False", "USE_L1_LOSS": "False",
    "CONSTANT_SCHEDULE": "True", "GUIDANCE_EARLY_STOP": "False",
    "BLOCK_UNET_GRAD": "True", "USE_DPMSOLVER": "False", "USE_DDIM": "False",
    "HYBRID_MODE": "False", "TIMESTEP_RESPACING": '""',
    "S_END": "1.0", "GUIDANCE_EVERY_K": "2",
    "GUIDANCE_SCALE": "0.15", "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0",
    "RESIDUAL_BLEND": "0.0",
}


def modify_trainpy(overrides):
    with open(TRAIN_PY, "r") as f:
        content = f.read()
    for var, val in overrides.items():
        if val.startswith('"') and val.endswith('"'):
            content = re.sub(rf'{var} = "[^"]*"', f'{var} = {val}', content)
        elif val in ("True", "False"):
            content = re.sub(rf'{var} = (True|False)', f'{var} = {val}', content)
        else:
            content = re.sub(rf'{var} = [\d.]+', f'{var} = {val}', content)
    with open(TRAIN_PY, "w") as f:
        f.write(content)


def extract_metrics(output):
    """Extract per-image metrics from train.py stdout."""
    imgs = []
    # Find per-image data block
    block = re.search(r'image quality metrics \(per-image\):(.*?)(?:\n={60}|\n---|\Z)', output, re.DOTALL)
    if not block:
        return [], float('inf'), 0
    # Parse per-image lines  
    for line in block.group(1).split('\n'):
        m = re.match(r'\s*img_\d+:\s*(.*)', line)
        if m:
            d = {}
            for pair in m.group(1).split(','):
                kv = pair.split('=')
                if len(kv) == 2:
                    try:
                        d[kv[0].strip()] = float(kv[1].strip())
                    except:
                        pass
            imgs.append(d)
    # Extract total time and other globals
    sec = float(re.search(r'total_seconds:\s+([\d.]+)', output).group(1)) if re.search(r'total_seconds:\s+([\d.]+)', output) else 0
    vram = float(re.search(r'peak_vram_mb:\s+([\d.]+)', output).group(1)) if re.search(r'peak_vram_mb:\s+([\d.]+)', output) else 0
    return imgs, sec, vram


def run_config(name, config, out_subdir):
    out_dir = os.path.join(OUT_ROOT, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    full = dict(config)
    full["OUT_DIR"] = f'"../final_experiments/batch_severe/{out_subdir}"'
    modify_trainpy(full)
    print(f"  Running {name} on 5 images...")
    t0 = time.time()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    result = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True, text=True, timeout=1800, cwd=WORK, env=env,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  CRASHED: {result.stderr[-200:] if result.stderr else 'unknown'}")
        return None
    imgs, total_sec, vram = extract_metrics(result.stdout)
    print(f"  Done in {elapsed:.0f}s (total_sec={total_sec:.1f}), {len(imgs)} images")
    return imgs, total_sec, vram


def stats(imgs):
    """Compute mean ± std across images."""
    if not imgs:
        return {}
    keys = ['psnr', 'ssim_gt', 'niqe']
    result = {}
    for k in keys:
        vals = [d[k] for d in imgs if k in d]
        if vals:
            result[k] = (np.mean(vals), np.std(vals))
    return result


# ── Run ──
print("=" * 70)
print("  BATCH TEST: A0 (baseline) vs K2_s15 (best) on 5 Severe images")
print("=" * 70)

# A0 (baseline)
print("\n[1/2] A0_true_paper (MSE, CONSTANT, K=1, scale=0.10, blend=0)")
a0_imgs, a0_sec, a0_vram = run_config("A0", A0_CONFIG, "A0_baseline")
a0_stats = stats(a0_imgs) if a0_imgs else {}

# K2_s15 (best)
print("\n[2/2] K2_s15 (MSE, CONSTANT, K=2, scale=0.15, blend=0)")
k2_imgs, k2_sec, k2_vram = run_config("K2_s15", K2S15_CONFIG, "K2_s15_best")
k2_stats = stats(k2_imgs) if k2_imgs else {}

# ── Print results ──
print(f"\n{'=' * 70}")
print(f"  RESULTS (mean ± std, 5 severe images)")
print(f"{'=' * 70}")
print(f"  {'Metric':<15s} {'A0 (baseline)':>22s} {'K2_s15 (best)':>22s} {'Delta':>10s}")
print(f"  {'-' * 70}")

for k, label in [('psnr', 'PSNR (dB)'), ('ssim_gt', 'SSIM_gt'), ('niqe', 'NIQE')]:
    if k in a0_stats and k in k2_stats:
        a0_m, a0_s = a0_stats[k]
        k2_m, k2_s = k2_stats[k]
        delta = k2_m - a0_m
        direction = '↑' if k != 'niqe' else '↓'
        print(f"  {label:<15s} {a0_m:>7.2f} ± {a0_s:.2f}     {k2_m:>7.2f} ± {k2_s:.2f}     {delta:>+8.2f} {direction}")

print(f"  {'Time (s)':<15s} {a0_sec:>21.1f}      {k2_sec:>21.1f}     {k2_sec-a0_sec:>+8.1f} ↓")
print(f"  {'VRAM (MB)':<15s} {a0_vram:>21.0f}      {k2_vram:>21.0f}     {k2_vram-a0_vram:>+8.0f}")

# Per-image breakdown
print(f"\n  Per-image breakdown:")
print(f"  {'Image':<18s} {'A0 PSNR':>9s} {'A0 NIQE':>9s} {'K2 PSNR':>9s} {'K2 NIQE':>9s}")
print(f"  {'-' * 56}")
if a0_imgs and k2_imgs:
    for i in range(min(len(a0_imgs), len(k2_imgs))):
        a0p = a0_imgs[i].get('psnr', 0)
        a0n = a0_imgs[i].get('niqe', 0)
        k2p = k2_imgs[i].get('psnr', 0)
        k2n = k2_imgs[i].get('niqe', 0)
        # Get image name from output dir
        print(f"  img_{i:02d}              {a0p:>7.2f} {a0n:>9.0f} {k2p:>9.2f} {k2n:>9.0f}")
