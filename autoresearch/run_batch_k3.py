#!/usr/bin/env python3
"""Batch K3_s225 on 5 severe images: MSE+CONSTANT+BLEND=0+BLOCK_UNET, K=3, s=0.225."""
import os, sys, re, time, subprocess, shutil, tempfile, json

WORK = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(WORK)
TRAIN_PY = os.path.join(WORK, "train.py")
OUT_ROOT = os.path.join(PROJ, "final_experiments", "batch_severe_k3")
os.makedirs(OUT_ROOT, exist_ok=True)

IMG_IDS = ["0005_00073", "0006_00092", "0007_00010", "0008_00016", "0009_00041"]
LQ_SRC = os.path.join(PROJ, "testdata", "temp_multi", "LQ")
GT_SRC = os.path.join(PROJ, "testdata", "temp_multi", "GT")

subprocess.run(f"cp {TRAIN_PY} {TRAIN_PY}.bak", shell=True)

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
    imgs = []
    block = re.search(r'image quality metrics \(per-image\):(.*?)(?:\n={60}|\n---|\Z)', output, re.DOTALL)
    if block:
        for line in block.group(1).split('\n'):
            m = re.match(r'\s*img_\d+:\s*(.*)', line)
            if m:
                d = {}
                for pair in m.group(1).split(','):
                    kv = pair.split('=')
                    if len(kv) == 2:
                        try: d[kv[0].strip()] = float(kv[1].strip())
                        except: pass
                imgs.append(d)
    tsec = float(re.search(r'total_seconds:\s+([\d.]+)', output).group(1)) if re.search(r'total_seconds:\s+([\d.]+)', output) else 0
    vram = float(re.search(r'peak_vram_mb:\s+([\d.]+)', output).group(1)) if re.search(r'peak_vram_mb:\s+([\d.]+)', output) else 0
    psnr = float(re.search(r'psnr:\s+([\d.]+)', output).group(1)) if re.search(r'psnr:\s+([\d.]+)', output) else 0
    ssim = float(re.search(r'ssim_gt:\s+([\d.]+)', output).group(1)) if re.search(r'ssim_gt:\s+([\d.]+)', output) else 0
    niqe = float(re.search(r'niqe:\s+([\d.]+)', output).group(1)) if re.search(r'niqe:\s+([\d.]+)', output) else 0
    return imgs, tsec, vram, psnr, ssim, niqe

cfg = {
    "IN_DIR": f'"{LQ_SRC}"', "GT_DIR": f'"{GT_SRC}"',
    "OUT_DIR": f'"../final_experiments/batch_severe_k3"',
    "MAX_IMAGES": "0", "SEED": "1234",
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
print("Running K3_s225 on 5 severe images...")
result = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True,
                       timeout=3600, cwd=WORK, env=env)

subprocess.run(f"cp {TRAIN_PY}.bak {TRAIN_PY} && rm -f {TRAIN_PY}.bak", shell=True)

if result.returncode != 0:
    print(f"FAILED: {result.stderr[-300:]}")
    exit(1)

imgs, tsec, vram, apsnr, assim, aniqe = extract_metrics(result.stdout)
print(f"\nDone: total_seconds={tsec:.1f}, avg PSNR={apsnr:.2f}, avg SSIM={assim:.4f}, avg NIQE={aniqe:.0f}")
print(f"\n  {'Image':<18s} {'PSNR':>8s} {'SSIM':>8s} {'NIQE':>8s}")
for i, img_id in enumerate(IMG_IDS):
    if i < len(imgs):
        d = imgs[i]
        print(f"  {img_id:<18s} {d.get('psnr',0):>8.2f} {d.get('ssim_gt',0):>8.4f} {d.get('niqe',0):>8.0f}")

with open(os.path.join(OUT_ROOT, "results.json"), "w") as f:
    json.dump({"imgs": imgs, "time": tsec, "vram": vram, "avg_psnr": apsnr, "avg_ssim": assim, "avg_niqe": aniqe}, f, indent=2)
