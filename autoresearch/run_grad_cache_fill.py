#!/usr/bin/env python3
"""Quickly run A0-A1-B1-B2-B3 to capture metrics."""
import os, sys, re, time, subprocess, shutil, tempfile, json

WORK = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(WORK)
TRAIN_PY = os.path.join(WORK, "train.py")
OUT_ROOT = os.path.join(PROJ, "final_experiments", "grad_cache")
TEST_IMG = "0006_00092"
SRC_LQ = os.path.join(PROJ, "testdata/pairs/LQ", f"{TEST_IMG}.png")
SRC_GT = os.path.join(PROJ, "testdata/pairs/GT", f"{TEST_IMG}.png")

EXPERIMENTS = [
    ("A0", "A0_paper",          1, 0.10, False),
    ("A1", "A1_K2_s15",         2, 0.15, False),
    ("B1", "B1_K2_s10_cache",   2, 0.10, True),
    ("B2", "B2_K2_s15_cache",   2, 0.15, True),
    ("B3", "B3_K2_s20_cache",   2, 0.20, True),
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

existing = json.load(open(os.path.join(OUT_ROOT, "results.json"))) if os.path.exists(os.path.join(OUT_ROOT, "results.json")) else {}

for exp_id, label, K, scale, use_cache in EXPERIMENTS:
    if exp_id in existing:
        continue
    print(f"  {exp_id} — K={K}, scale={scale}, cache={use_cache}")
    out_subdir = os.path.join(OUT_ROOT, label)
    os.makedirs(out_subdir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="gcache_")
    os.makedirs(os.path.join(tmp_dir, "LQ"))
    os.makedirs(os.path.join(tmp_dir, "GT"))
    shutil.copy2(SRC_LQ, os.path.join(tmp_dir, "LQ", f"{TEST_IMG}.png"))
    shutil.copy2(SRC_GT, os.path.join(tmp_dir, "GT", f"{TEST_IMG}.png"))
    cfg = {
        "IN_DIR": f'"{tmp_dir}/LQ"', "GT_DIR": f'"{tmp_dir}/GT"',
        "OUT_DIR": f'"../final_experiments/grad_cache/{label}"',
        "MAX_IMAGES": "1", "SEED": "1234",
        "USE_HUBER_LOSS": "False", "USE_L1_LOSS": "False",
        "CONSTANT_SCHEDULE": "True", "GUIDANCE_EARLY_STOP": "False",
        "BLOCK_UNET_GRAD": "True", "USE_DPMSOLVER": "False",
        "USE_DDIM": "False", "HYBRID_MODE": "False",
        "TIMESTEP_RESPACING": '""', "S_END": "1.0",
        "GUIDANCE_EVERY_K": str(K), "GUIDANCE_SCALE": str(scale),
        "USE_CACHED_GRADIENT": str(use_cache),
        "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0", "RESIDUAL_BLEND": "0.0",
    }
    modify_trainpy(cfg)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    result = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True, timeout=1800, cwd=WORK, env=env)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if result.returncode != 0:
        print(f"    FAILED")
        continue
    psnr, ssim, niqe, tsec, vram = extract_metrics(result.stdout)
    existing[exp_id] = {"label": label, "K": K, "scale": scale, "cache": use_cache,
                        "psnr": psnr, "ssim": ssim, "niqe": niqe, "time": tsec, "vram": vram}
    print(f"    PSNR={psnr:.2f}, SSIM={ssim:.4f}, NIQE={niqe:.0f}, Time={tsec:.1f}s, VRAM={vram:.0f}MB")
    expected_img = os.path.join(out_subdir, f"{TEST_IMG}.png")
    if os.path.exists(expected_img):
        shutil.copy2(expected_img, os.path.join(OUT_ROOT, f"{label}.png"))

with open(os.path.join(OUT_ROOT, "results.json"), "w") as f:
    json.dump(existing, f, indent=2)

print(f"\nComplete:")
for exp_id in sorted(existing.keys()):
    r = existing[exp_id]
    c = "Y" if r["cache"] else "N"
    print(f"  {exp_id:<6s} K={r['K']}  s={r['scale']:.2f}  cache={c}  PSNR={r['psnr']:.2f}  NIQE={r['niqe']:.0f}  Time={r['time']:.0f}s")
