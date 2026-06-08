#!/usr/bin/env python3
"""Run 4 configs on severe degradation image: paper + 3 best approaches."""
import os, sys, re, time, subprocess

WORK = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(WORK, "train.py")
OUT_ROOT = os.path.join(os.path.dirname(WORK), "final_experiments", "severe")
os.makedirs(OUT_ROOT, exist_ok=True)

BASE = {
    "TASK": '"restoration"', "SEED": "1234",
    "LIGHTNESS_WEIGHT": "1.0", "COLOR_WEIGHT": "0.05",
    "UNMASKED_WEIGHT": "1.0", "SS_WEIGHT": "1.0",
    "REF_WEIGHT": "25.0", "OP_LIGHTNESS_WEIGHT": "1.0", "OP_COLOR_WEIGHT": "0.5",
    "N": "1", "S_START": "1.0",
    "CLIP_DENOISED": "True", "BATCH_SIZE": "1",
    "IMAGE_SIZE": "512", "DIFFUSION_STEPS": "1000",
    "IN_DIR": '"../testdata/temp_severe/LQ"',
    "GT_DIR": '"../testdata/temp_severe/GT"',
    "MAX_IMAGES": "0",
    "USE_FP16": "False", "RUN_TAG": '""',
}

experiments = [
    # 1: Paper config (MSE, CONSTANT, K=1, RESIDUAL_BLEND=0, BLOCK_UNET=False)
    ("paper", {
        "USE_HUBER_LOSS": "False", "USE_L1_LOSS": "False",
        "CONSTANT_SCHEDULE": "True", "GUIDANCE_EARLY_STOP": "False",
        "BLOCK_UNET_GRAD": "False", "USE_DPMSOLVER": "False", "USE_DDIM": "False",
        "HYBRID_MODE": "False", "TIMESTEP_RESPACING": '""',
        "S_END": "1.0", "GUIDANCE_EVERY_K": "1",
        "GUIDANCE_SCALE": "0.10", "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0",
        "RESIDUAL_BLEND": "0.0",
    }),
    # 2: R2_E2 — balanced optimum (MSE, CONSTANT, K=2, s=0.25)
    ("R2E2_balanced", {
        "USE_HUBER_LOSS": "False", "USE_L1_LOSS": "False",
        "CONSTANT_SCHEDULE": "True", "GUIDANCE_EARLY_STOP": "False",
        "BLOCK_UNET_GRAD": "True", "USE_DPMSOLVER": "False", "USE_DDIM": "False",
        "HYBRID_MODE": "False", "TIMESTEP_RESPACING": '""',
        "S_END": "1.0", "GUIDANCE_EVERY_K": "2",
        "GUIDANCE_SCALE": "0.25", "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0",
        "RESIDUAL_BLEND": "0.5",
    }),
    # 3: N3 — extreme scale (scale=1.0, K=2)
    ("N3_extreme", {
        "USE_HUBER_LOSS": "False", "USE_L1_LOSS": "False",
        "CONSTANT_SCHEDULE": "True", "GUIDANCE_EARLY_STOP": "False",
        "BLOCK_UNET_GRAD": "True", "USE_DPMSOLVER": "False", "USE_DDIM": "False",
        "HYBRID_MODE": "False", "TIMESTEP_RESPACING": '""',
        "S_END": "1.0", "GUIDANCE_EVERY_K": "2",
        "GUIDANCE_SCALE": "1.00", "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0",
        "RESIDUAL_BLEND": "0.5",
    }),
    # 4: R3_E2 — speed champion (K=3, s=0.35, LINEAR)
    ("R3E2_speed", {
        "USE_HUBER_LOSS": "False", "USE_L1_LOSS": "False",
        "CONSTANT_SCHEDULE": "False", "GUIDANCE_EARLY_STOP": "False",
        "BLOCK_UNET_GRAD": "True", "USE_DPMSOLVER": "False", "USE_DDIM": "False",
        "HYBRID_MODE": "False", "TIMESTEP_RESPACING": '""',
        "S_END": "1.0", "GUIDANCE_EVERY_K": "3",
        "GUIDANCE_SCALE": "0.35", "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0",
        "RESIDUAL_BLEND": "0.5",
    }),
]


def modify_trainpy(overrides):
    with open(TRAIN_PY, "r") as f:
        content = f.read()
    full = dict(BASE)
    full.update(overrides)
    for var, val in full.items():
        if val.startswith('"') and val.endswith('"'):
            content = re.sub(rf'{var} = "[^"]*"', f'{var} = {val}', content)
    for var, val in full.items():
        if val in ("True", "False"):
            content = re.sub(rf'{var} = (True|False)', f'{var} = {val}', content)
    for var, val in full.items():
        if not val.startswith('"') and val not in ("True", "False"):
            content = re.sub(rf'{var} = [\d.]+', f'{var} = {val}', content)
    with open(TRAIN_PY, "w") as f:
        f.write(content)


def extract_metrics(output):
    def g(pat, default="0"):
        m = re.search(pat, output)
        return m.group(1) if m else default
    return {
        "psnr": float(g(r"psnr:\s+([\d.]+)")),
        "ssim_gt": float(g(r"ssim_gt:\s+([\d.]+)")),
        "niqe": float(g(r"niqe:\s+([\d.]+)")),
        "sec": float(g(r"total_seconds:\s+([\d.]+)")),
    }


def run_one(name, overrides):
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    full = dict(overrides)
    full["OUT_DIR"] = f'"../final_experiments/severe/{name}"'
    modify_trainpy(full)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    result = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True, text=True, timeout=900, cwd=WORK, env=env,
    )
    if result.returncode != 0:
        print(f"  CRASHED: {result.stderr[-200:] if result.stderr else 'unknown'}")
        return None
    return extract_metrics(result.stdout)


def main():
    results = {}
    labels = {
        "paper": "Paper (MSE,CONSTANT,K=1,blend=0)",
        "R2E2_balanced": "R2_E2 (MSE,CONSTANT,K=2,s=0.25)",
        "N3_extreme": "N3 (scale=1.0,K=2)",
        "R3E2_speed": "R3_E2 (K=3,s=0.35,LINEAR)",
    }
    for idx, (name, overrides) in enumerate(experiments):
        print(f"[{idx+1}/4] {name}: {labels[name]}")
        t0 = time.time()
        r = run_one(name, overrides)
        elapsed = time.time() - t0
        if r:
            results[name] = r
            print(f"  PSNR={r['psnr']:.2f} SSIM={r['ssim_gt']:.4f} NIQE={r['niqe']:.0f} Time={r['sec']:.1f}s  [{elapsed:.0f}s]")
        else:
            print(f"  FAIL [{elapsed:.0f}s]")

    print(f"\n{'='*65}")
    print(f"  SEVERE DEGRADATION RESULTS (0006_00092.png)")
    print(f"  {'Config':<20s} {'PSNR':>7s} {'SSIM':>7s} {'NIQE':>7s} {'Time':>7s}")
    print(f"  {'-'*50}")
    for name, _ in experiments:
        r = results.get(name)
        if r:
            print(f"  {name:<20s} {r['psnr']:7.2f} {r['ssim_gt']:7.4f} {r['niqe']:7.0f} {r['sec']:7.1f}")

if __name__ == "__main__":
    main()
