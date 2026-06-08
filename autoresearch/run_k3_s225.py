#!/usr/bin/env python3
"""Run K=3, scale=0.225 for Mild and Severe — matching effective guidance (0.75×)."""
import os, sys, re, time, subprocess, shutil, tempfile, json

WORK = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(WORK)
TRAIN_PY = os.path.join(WORK, "train.py")
OUT_ROOT = os.path.join(PROJ, "final_experiments", "k3_s225")
os.makedirs(OUT_ROOT, exist_ok=True)

EXPERIMENTS = [
    ("K3_s225_mild",   "0000_00010"),
    ("K3_s225_severe", "0006_00092"),
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
    rc = int(re.search(r'total_restorer_calls:\s+(\d+)', output).group(1)) if re.search(r'total_restorer_calls:\s+(\d+)', output) else 0
    return psnr, ssim, niqe, tsec, vram, rc

results = {}
for label, img_id in EXPERIMENTS:
    print(f"  {label} — img={img_id}")
    out_subdir = os.path.join(OUT_ROOT, label)
    os.makedirs(out_subdir, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="k3_")
    os.makedirs(os.path.join(tmp_dir, "LQ"))
    os.makedirs(os.path.join(tmp_dir, "GT"))
    src_lq = os.path.join(PROJ, "testdata/pairs/LQ", f"{img_id}.png")
    src_gt = os.path.join(PROJ, "testdata/pairs/GT", f"{img_id}.png")
    shutil.copy2(src_lq, os.path.join(tmp_dir, "LQ", f"{img_id}.png"))
    shutil.copy2(src_gt, os.path.join(tmp_dir, "GT", f"{img_id}.png"))

    cfg = {
        "IN_DIR": f'"{tmp_dir}/LQ"', "GT_DIR": f'"{tmp_dir}/GT"',
        "OUT_DIR": f'"../final_experiments/k3_s225/{label}"',
        "MAX_IMAGES": "1", "SEED": "1234",
        "USE_HUBER_LOSS": "False", "USE_L1_LOSS": "False",
        "CONSTANT_SCHEDULE": "True", "GUIDANCE_EARLY_STOP": "False",
        "BLOCK_UNET_GRAD": "True", "USE_DPMSOLVER": "False",
        "USE_DDIM": "False", "HYBRID_MODE": "False",
        "TIMESTEP_RESPACING": '""', "S_END": "1.0",
        "GUIDANCE_EVERY_K": "3", "GUIDANCE_SCALE": "0.225",
        "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0", "RESIDUAL_BLEND": "0.0",
    }
    modify_trainpy(cfg)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    result = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True,
                           timeout=1800, cwd=WORK, env=env)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if result.returncode != 0:
        print(f"    FAILED: {result.stderr[-200:]}")
        continue
    psnr, ssim, niqe, tsec, vram, rc = extract_metrics(result.stdout)
    results[label] = {"psnr": psnr, "ssim": ssim, "niqe": niqe, "time": tsec, "vram": vram, "rc": rc}
    print(f"    PSNR={psnr:.2f}, SSIM={ssim:.4f}, NIQE={niqe:.0f}, Time={tsec:.1f}s, RC={rc}")

    expected_img = os.path.join(out_subdir, f"{img_id}.png")
    if os.path.exists(expected_img):
        shutil.copy2(expected_img, os.path.join(OUT_ROOT, f"{label}.png"))

print(f"\n  {'Level':<10s} {'PSNR':>8s} {'NIQE':>8s} {'Time':>8s} {'RC':>6s}")
for label in ["K3_s225_mild", "K3_s225_severe"]:
    if label in results:
        r = results[label]
        print(f"  {label:<10s} {r['psnr']:>8.2f} {r['niqe']:>8.0f} {r['time']:>8.1f} {r['rc']:>6d}")

with open(os.path.join(OUT_ROOT, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
