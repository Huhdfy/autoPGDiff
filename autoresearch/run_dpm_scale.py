#!/usr/bin/env python3
"""Run DPM-Solver-2 experiments at scale=0.1, 1.0, 5.0 and collect results.
Uses 0006_00092.png (Severe) consistently for all three runs."""
import os, sys, re, time, subprocess, shutil, tempfile

WORK = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(WORK)
TRAIN_PY = os.path.join(WORK, "train.py")
OUT_ROOT = os.path.join(PROJ, "final_experiments", "dpm_scale_compare")
os.makedirs(OUT_ROOT, exist_ok=True)

SCALES = [0.10, 1.0, 5.0]
TEST_IMG = "0006_00092"

SRC_LQ = os.path.join(PROJ, "testdata/pairs/LQ", f"{TEST_IMG}.png")
SRC_GT = os.path.join(PROJ, "testdata/pairs/GT", f"{TEST_IMG}.png")

BASE_CONFIG = {
    "MAX_IMAGES": "1",
    "SEED": "1234",
    "USE_HUBER_LOSS": "False",
    "USE_L1_LOSS": "False",
    "CONSTANT_SCHEDULE": "False",
    "GUIDANCE_EARLY_STOP": "False",
    "BLOCK_UNET_GRAD": "True",
    "USE_DPMSOLVER": "True",
    "USE_DDIM": "False",
    "HYBRID_MODE": "False",
    "TIMESTEP_RESPACING": '""',
    "DPM_SOLVER_STEPS": "35",
    "S_END": "1.0",
    "GUIDANCE_EVERY_K": "1",
    "EDGE_WEIGHT": "0.02",
    "GRAD_MOMENTUM": "0.0",
    "RESIDUAL_BLEND": "0.5",
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
    psnr = float(re.search(r'psnr:\s+([\d.]+)', output).group(1)) if re.search(r'psnr:\s+([\d.]+)', output) else 0
    ssim = float(re.search(r'ssim_gt:\s+([\d.]+)', output).group(1)) if re.search(r'ssim_gt:\s+([\d.]+)', output) else 0
    niqe = float(re.search(r'niqe:\s+([\d.]+)', output).group(1)) if re.search(r'niqe:\s+([\d.]+)', output) else 0
    tsec = float(re.search(r'total_seconds:\s+([\d.]+)', output).group(1)) if re.search(r'total_seconds:\s+([\d.]+)', output) else 0
    vram = float(re.search(r'peak_vram_mb:\s+([\d.]+)', output).group(1)) if re.search(r'peak_vram_mb:\s+([\d.]+)', output) else 0
    return psnr, ssim, niqe, tsec, vram

results = {}

for scale in SCALES:
    tag = f"s{int(scale*100):03d}" if scale < 1 else f"s{int(scale*100):03d}"
    label = f"DPM35_{tag}"
    print(f"\n{'='*60}")
    print(f"  Running: DPM-Solver-2, 35 steps, scale={scale}")
    print(f"{'='*60}")

    out_subdir = os.path.join(OUT_ROOT, label)
    os.makedirs(out_subdir, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="dpm35_")
    os.makedirs(os.path.join(tmp_dir, "LQ"))
    os.makedirs(os.path.join(tmp_dir, "GT"))
    shutil.copy2(SRC_LQ, os.path.join(tmp_dir, "LQ", f"{TEST_IMG}.png"))
    shutil.copy2(SRC_GT, os.path.join(tmp_dir, "GT", f"{TEST_IMG}.png"))

    cfg = dict(BASE_CONFIG)
    cfg["GUIDANCE_SCALE"] = str(scale)
    cfg["OUT_DIR"] = f'"../final_experiments/dpm_scale_compare/{label}"'
    cfg["IN_DIR"] = f'"{tmp_dir}/LQ"'
    cfg["GT_DIR"] = f'"{tmp_dir}/GT"'

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
        results[scale] = None
        continue

    psnr, ssim, niqe, tsec, vram = extract_metrics(result.stdout)
    results[scale] = {"psnr": psnr, "ssim": ssim, "niqe": niqe, "time": tsec, "vram": vram}
    print(f"  Done in {elapsed:.0f}s")
    print(f"  PSNR={psnr:.2f}, SSIM={ssim:.4f}, NIQE={niqe:.0f}, Time={tsec:.1f}s, VRAM={vram:.0f}MB")

    expected_img = os.path.join(out_subdir, f"{TEST_IMG}.png")
    if os.path.exists(expected_img):
        shutil.copy2(expected_img, os.path.join(OUT_ROOT, f"dpm35_{tag}.png"))

print(f"\n{'='*60}")
print(f"  RESULTS SUMMARY (Image: {TEST_IMG}.png)")
print(f"{'='*60}")
print(f"  {'Scale':<8s} {'PSNR':>8s} {'SSIM':>8s} {'NIQE':>8s} {'Time(s)':>8s} {'VRAM(MB)':>8s}")
for scale in SCALES:
    if results[scale]:
        r = results[scale]
        print(f"  {scale:<8.2f} {r['psnr']:>8.2f} {r['ssim']:>8.4f} {r['niqe']:>8.0f} {r['time']:>8.1f} {r['vram']:>8.0f}")
    else:
        print(f"  {scale:<8.2f} {'FAILED':>8s}")
