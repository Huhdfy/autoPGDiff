#!/usr/bin/env python3
"""Run K=2/3/5 with fixed scale=0.10 on Mild (0000_00010) — isolate K effect."""
import os, sys, re, time, subprocess, shutil, tempfile, json

WORK = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(WORK)
TRAIN_PY = os.path.join(WORK, "train.py")
OUT_ROOT = os.path.join(PROJ, "final_experiments", "k_isolate")
os.makedirs(OUT_ROOT, exist_ok=True)

IMG_ID = "0000_00010"
SRC_LQ = os.path.join(PROJ, "testdata/pairs/LQ", f"{IMG_ID}.png")
SRC_GT = os.path.join(PROJ, "testdata/pairs/GT", f"{IMG_ID}.png")

EXPERIMENTS = [
    ("K2_s010_mild", 2, 0.10),
    ("K3_s010_mild", 3, 0.10),
    ("K5_s010_mild", 5, 0.10),
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

cp = f"cp {TRAIN_PY} {TRAIN_PY}.bak"
subprocess.run(cp, shell=True)

results = {}
for label, K, scale in EXPERIMENTS:
    print(f"  {label} — K={K}, scale={scale}")
    out_subdir = os.path.join(OUT_ROOT, label)
    os.makedirs(out_subdir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="kiso_")
    os.makedirs(os.path.join(tmp_dir, "LQ"))
    os.makedirs(os.path.join(tmp_dir, "GT"))
    shutil.copy2(SRC_LQ, os.path.join(tmp_dir, "LQ", f"{IMG_ID}.png"))
    shutil.copy2(SRC_GT, os.path.join(tmp_dir, "GT", f"{IMG_ID}.png"))

    cfg = {
        "IN_DIR": f'"{tmp_dir}/LQ"', "GT_DIR": f'"{tmp_dir}/GT"',
        "OUT_DIR": f'"../final_experiments/k_isolate/{label}"',
        "MAX_IMAGES": "1", "SEED": "1234",
        "USE_HUBER_LOSS": "False", "USE_L1_LOSS": "False",
        "CONSTANT_SCHEDULE": "True", "GUIDANCE_EARLY_STOP": "False",
        "BLOCK_UNET_GRAD": "True", "USE_DPMSOLVER": "False",
        "USE_DDIM": "False", "HYBRID_MODE": "False",
        "TIMESTEP_RESPACING": '""', "S_END": "1.0",
        "GUIDANCE_EVERY_K": str(K), "GUIDANCE_SCALE": str(scale),
        "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0", "RESIDUAL_BLEND": "0.0",
    }
    modify_trainpy(cfg)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    result = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True,
                           timeout=1800, cwd=WORK, env=env)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if result.returncode != 0:
        print(f"    FAILED")
        continue
    psnr, ssim, niqe, tsec, vram = extract_metrics(result.stdout)
    results[label] = {"K": K, "scale": scale, "psnr": psnr, "ssim": ssim, "niqe": niqe, "time": tsec, "vram": vram}
    print(f"    PSNR={psnr:.2f}, SSIM={ssim:.4f}, NIQE={niqe:.0f}, Time={tsec:.1f}s")
    expected = os.path.join(out_subdir, f"{IMG_ID}.png")
    if os.path.exists(expected):
        shutil.copy2(expected, os.path.join(OUT_ROOT, f"{label}.png"))

subprocess.run(f"cp {TRAIN_PY}.bak {TRAIN_PY} && rm -f {TRAIN_PY}.bak", shell=True)

print(f"\n  {'ID':<18s} {'K':>3s} {'PSNR':>8s} {'NIQE':>8s} {'Time':>8s}")
for label in sorted(results.keys()):
    r = results[label]
    print(f"  {label:<18s} {r['K']:>3d} {r['psnr']:>8.2f} {r['niqe']:>8.0f} {r['time']:>8.1f}")

with open(os.path.join(OUT_ROOT, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
