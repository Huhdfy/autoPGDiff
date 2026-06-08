#!/usr/bin/env python3
"""Fill data gaps: K2_s10 Mild, K2_s10 Severe, K2_s20 Mild."""
import os, sys, re, time, subprocess, shutil, tempfile, json

WORK = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(WORK)
TRAIN_PY = os.path.join(WORK, "train.py")
OUT_ROOT = os.path.join(PROJ, "final_experiments", "scale_sweep")
os.makedirs(OUT_ROOT, exist_ok=True)

EXPERIMENTS = [
    ("K2_s10_mild",  2, 0.10, "0000_00010"),
    ("K2_s10_severe", 2, 0.10, "0006_00092"),
    ("K2_s20_mild",  2, 0.20, "0000_00010"),
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
for label, K, scale, img_id in EXPERIMENTS:
    print(f"  {label} — K={K}, scale={scale}, img={img_id}")
    out_subdir = os.path.join(OUT_ROOT, label)
    os.makedirs(out_subdir, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="sweep_")
    os.makedirs(os.path.join(tmp_dir, "LQ"))
    os.makedirs(os.path.join(tmp_dir, "GT"))
    src_lq = os.path.join(PROJ, "testdata/pairs/LQ", f"{img_id}.png")
    src_gt = os.path.join(PROJ, "testdata/pairs/GT", f"{img_id}.png")
    shutil.copy2(src_lq, os.path.join(tmp_dir, "LQ", f"{img_id}.png"))
    shutil.copy2(src_gt, os.path.join(tmp_dir, "GT", f"{img_id}.png"))

    cfg = {
        "IN_DIR": f'"{tmp_dir}/LQ"', "GT_DIR": f'"{tmp_dir}/GT"',
        "OUT_DIR": f'"../final_experiments/scale_sweep/{label}"',
        "MAX_IMAGES": "1", "SEED": "1234",
        "USE_HUBER_LOSS": "False", "USE_L1_LOSS": "False",
        "CONSTANT_SCHEDULE": "True", "GUIDANCE_EARLY_STOP": "False",
        "BLOCK_UNET_GRAD": "True", "USE_DPMSOLVER": "False",
        "USE_DDIM": "False", "HYBRID_MODE": "False",
        "TIMESTEP_RESPACING": '""', "S_END": "1.0",
        "GUIDANCE_EVERY_K": str(K), "GUIDANCE_SCALE": str(scale),
        "USE_CACHED_GRADIENT": "False",
        "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0", "RESIDUAL_BLEND": "0.0",
    }
    modify_trainpy(cfg)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    t0 = time.time()
    result = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True,
                           timeout=1800, cwd=WORK, env=env)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if result.returncode != 0:
        print(f"    FAILED")
        continue
    psnr, ssim, niqe, tsec, vram = extract_metrics(result.stdout)
    results[label] = {"K": K, "scale": scale, "img": img_id,
                      "psnr": psnr, "ssim": ssim, "niqe": niqe, "time": tsec, "vram": vram}
    print(f"    PSNR={psnr:.2f}, SSIM={ssim:.4f}, NIQE={niqe:.0f}, Time={tsec:.1f}s")
    expected_img = os.path.join(out_subdir, f"{img_id}.png")
    if os.path.exists(expected_img):
        shutil.copy2(expected_img, os.path.join(OUT_ROOT, f"{label}.png"))

with open(os.path.join(OUT_ROOT, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nDone:")
for label, r in sorted(results.items()):
    print(f"  {label:<20s} PSNR={r['psnr']:.2f}  NIQE={r['niqe']:.0f}  Time={r['time']:.0f}s")
