#!/usr/bin/env python3
"""Run gradient-cache experiments: A0/A1 baseline, B/C/D groups, 9 new experiments."""
import os, sys, re, time, subprocess, shutil, tempfile, json

WORK = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(WORK)
TRAIN_PY = os.path.join(WORK, "train.py")
OUT_ROOT = os.path.join(PROJ, "final_experiments", "grad_cache")
os.makedirs(OUT_ROOT, exist_ok=True)

TEST_IMG = "0006_00092"
SRC_LQ = os.path.join(PROJ, "testdata/pairs/LQ", f"{TEST_IMG}.png")
SRC_GT = os.path.join(PROJ, "testdata/pairs/GT", f"{TEST_IMG}.png")

# ── Experiment matrix ──
# (exp_id, label, K, scale, use_cache)
EXPERIMENTS = [
    # A group — baselines
    ("A0", "A0_paper",         1, 0.10, False),
    ("A1", "A1_K2_s15",        2, 0.15, False),

    # B group — K=2 with cache
    ("B1", "B1_K2_s10_cache",  2, 0.10, True),
    ("B2", "B2_K2_s15_cache",  2, 0.15, True),
    ("B3", "B3_K2_s20_cache",  2, 0.20, True),

    # C group — K=3
    ("C1", "C1_K3_s15",        3, 0.15, False),
    ("C2", "C2_K3_s15_cache",  3, 0.15, True),
    ("C3", "C3_K3_s20_cache",  3, 0.20, True),

    # D group — K=5
    ("D1", "D1_K5_s15",        5, 0.15, False),
    ("D2", "D2_K5_s15_cache",  5, 0.15, True),
]

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
    psnr = float(re.search(r'psnr:\s+([\d.]+)', output).group(1)) if re.search(r'psnr:\s+([\d.]+)', output) else 0
    ssim = float(re.search(r'ssim_gt:\s+([\d.]+)', output).group(1)) if re.search(r'ssim_gt:\s+([\d.]+)', output) else 0
    niqe = float(re.search(r'niqe:\s+([\d.]+)', output).group(1)) if re.search(r'niqe:\s+([\d.]+)', output) else 0
    tsec = float(re.search(r'total_seconds:\s+([\d.]+)', output).group(1)) if re.search(r'total_seconds:\s+([\d.]+)', output) else 0
    vram = float(re.search(r'peak_vram_mb:\s+([\d.]+)', output).group(1)) if re.search(r'peak_vram_mb:\s+([\d.]+)', output) else 0
    return psnr, ssim, niqe, tsec, vram

results = {}
N = len(EXPERIMENTS)

for idx, (exp_id, label, K, scale, use_cache) in enumerate(EXPERIMENTS):
    print(f"\n{'='*60}")
    print(f"  [{idx+1}/{N}] {exp_id} : {label}")
    print(f"       K={K}, scale={scale}, cache={use_cache}")
    print(f"{'='*60}")

    out_subdir = os.path.join(OUT_ROOT, label)
    os.makedirs(out_subdir, exist_ok=True)

    # temp dir with single image
    tmp_dir = tempfile.mkdtemp(prefix="gcache_")
    os.makedirs(os.path.join(tmp_dir, "LQ"))
    os.makedirs(os.path.join(tmp_dir, "GT"))
    shutil.copy2(SRC_LQ, os.path.join(tmp_dir, "LQ", f"{TEST_IMG}.png"))
    shutil.copy2(SRC_GT, os.path.join(tmp_dir, "GT", f"{TEST_IMG}.png"))

    cfg = {
        "IN_DIR": f'"{tmp_dir}/LQ"',
        "GT_DIR": f'"{tmp_dir}/GT"',
        "OUT_DIR": f'"../final_experiments/grad_cache/{label}"',
        "MAX_IMAGES": "1",
        "SEED": "1234",
        "USE_HUBER_LOSS": "False",
        "USE_L1_LOSS": "False",
        "CONSTANT_SCHEDULE": "True",
        "GUIDANCE_EARLY_STOP": "False",
        "BLOCK_UNET_GRAD": "True",
        "USE_DPMSOLVER": "False",
        "USE_DDIM": "False",
        "HYBRID_MODE": "False",
        "TIMESTEP_RESPACING": '""',
        "S_END": "1.0",
        "GUIDANCE_EVERY_K": str(K),
        "GUIDANCE_SCALE": str(scale),
        "USE_CACHED_GRADIENT": str(use_cache),
        "EDGE_WEIGHT": "0.0",
        "GRAD_MOMENTUM": "0.0",
        "RESIDUAL_BLEND": "0.0",
    }
    modify_trainpy(cfg)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True, text=True, timeout=1800, cwd=WORK, env=env,
    )
    elapsed = time.time() - t0
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-300:] if result.stderr else 'unknown'}")
        results[exp_id] = None
        continue

    psnr, ssim, niqe, tsec, vram = extract_metrics(result.stdout)
    results[exp_id] = {"label": label, "K": K, "scale": scale, "cache": use_cache,
                        "psnr": psnr, "ssim": ssim, "niqe": niqe, "time": tsec, "vram": vram}
    print(f"  OK in {elapsed:.0f}s | PSNR={psnr:.2f}, SSIM={ssim:.4f}, NIQE={niqe:.0f}, Time={tsec:.1f}s, VRAM={vram:.0f}MB")

    # copy output image
    expected_img = os.path.join(out_subdir, f"{TEST_IMG}.png")
    if os.path.exists(expected_img):
        shutil.copy2(expected_img, os.path.join(OUT_ROOT, f"{label}.png"))

# ── Summary ──
print(f"\n{'='*70}")
print(f"  GRADIENT CACHE EXPERIMENT RESULTS  (Image: {TEST_IMG}.png)")
print(f"{'='*70}")
hdr = f"  {'ID':<6s} {'K':>3s} {'scale':>6s} {'cache':>6s} {'PSNR':>8s} {'SSIM':>8s} {'NIQE':>8s} {'Time(s)':>8s} {'VRAM(MB)':>8s}"
print(hdr)
print(f"  {'-'*68}")
for exp_id, r in sorted(results.items()):
    c = "Y" if r["cache"] else "N"
    print(f"  {exp_id:<6s} {r['K']:>3d} {r['scale']:>6.2f} {c:>6s} {r['psnr']:>8.2f} {r['ssim']:>8.4f} {r['niqe']:>8.0f} {r['time']:>8.1f} {r['vram']:>8.0f}")

# ── Save JSON ──
with open(os.path.join(OUT_ROOT, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to {OUT_ROOT}/results.json")
