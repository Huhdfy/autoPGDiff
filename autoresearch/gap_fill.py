#!/usr/bin/env python3
"""Run all missing paper-metric experiments (N1-N7 + Phase1 reps)."""
import os, sys, re, time, subprocess

WORK = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(WORK, "train.py")
OUT_ROOT = os.path.join(os.path.dirname(WORK), "final_experiments")
LOG_DIR = os.path.join(WORK, "exp_logs")
os.makedirs(LOG_DIR, exist_ok=True)

BASE = {
    "TASK": '"restoration"', "SEED": "1234",
    "LIGHTNESS_WEIGHT": "1.0", "COLOR_WEIGHT": "0.05",
    "UNMASKED_WEIGHT": "1.0", "SS_WEIGHT": "1.0",
    "REF_WEIGHT": "25.0", "OP_LIGHTNESS_WEIGHT": "1.0", "OP_COLOR_WEIGHT": "0.5",
    "N": "1", "S_START": "1.0",
    "CLIP_DENOISED": "True", "BATCH_SIZE": "1",
    "IMAGE_SIZE": "512", "DIFFUSION_STEPS": "1000",
    "IN_DIR": '"../testdata/pairs/LQ"', "GT_DIR": '"../testdata/pairs/GT"',
    "MAX_IMAGES": "1",
    "USE_FP16": "False", "RUN_TAG": '""',
}

experiments = [
    # N1: DDPM 100 steps (extreme reduction)
    ("exp_N1_ddpm100", {
        "TIMESTEP_RESPACING": '"ddpm100"',
        "USE_HUBER_LOSS": "False", "USE_L1_LOSS": "False",
        "CONSTANT_SCHEDULE": "True", "GUIDANCE_EARLY_STOP": "False",
        "BLOCK_UNET_GRAD": "True", "USE_DPMSOLVER": "False", "USE_DDIM": "False",
        "HYBRID_MODE": "False",
        "S_END": "1.0", "GUIDANCE_EVERY_K": "1",
        "GUIDANCE_SCALE": "0.10", "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0",
        "RESIDUAL_BLEND": "0.5",
    }),

    # N2: L1 loss
    ("exp_N2_l1loss", {
        "TIMESTEP_RESPACING": '""',
        "USE_HUBER_LOSS": "False", "USE_L1_LOSS": "True",
        "CONSTANT_SCHEDULE": "True", "GUIDANCE_EARLY_STOP": "False",
        "BLOCK_UNET_GRAD": "True", "USE_DPMSOLVER": "False", "USE_DDIM": "False",
        "HYBRID_MODE": "False",
        "S_END": "1.0", "GUIDANCE_EVERY_K": "2",
        "GUIDANCE_SCALE": "0.25", "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0",
        "RESIDUAL_BLEND": "0.5",
    }),

    # N3: scale=1.0 extreme
    ("exp_N3_scale10", {
        "TIMESTEP_RESPACING": '""',
        "USE_HUBER_LOSS": "False", "USE_L1_LOSS": "False",
        "CONSTANT_SCHEDULE": "True", "GUIDANCE_EARLY_STOP": "False",
        "BLOCK_UNET_GRAD": "True", "USE_DPMSOLVER": "False", "USE_DDIM": "False",
        "HYBRID_MODE": "False",
        "S_END": "1.0", "GUIDANCE_EVERY_K": "2",
        "GUIDANCE_SCALE": "1.00", "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0",
        "RESIDUAL_BLEND": "0.5",
    }),

    # N4: DDIM 100 steps
    ("exp_N4_ddim", {
        "TIMESTEP_RESPACING": '"ddim100"',
        "USE_HUBER_LOSS": "False", "USE_L1_LOSS": "False",
        "CONSTANT_SCHEDULE": "True", "GUIDANCE_EARLY_STOP": "False",
        "BLOCK_UNET_GRAD": "True", "USE_DPMSOLVER": "False", "USE_DDIM": "True",
        "HYBRID_MODE": "False",
        "S_END": "1.0", "GUIDANCE_EVERY_K": "1",
        "GUIDANCE_SCALE": "0.10", "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0",
        "RESIDUAL_BLEND": "0.5",
    }),

    # N5: DPM 35 + scale=5.0 (scale amplification test)
    ("exp_N5_dpm35_s5", {
        "TIMESTEP_RESPACING": '""',
        "USE_HUBER_LOSS": "False", "USE_L1_LOSS": "False",
        "CONSTANT_SCHEDULE": "False", "GUIDANCE_EARLY_STOP": "False",
        "BLOCK_UNET_GRAD": "True", "USE_DPMSOLVER": "True", "USE_DDIM": "False",
        "HYBRID_MODE": "False",
        "DPM_SOLVER_STEPS": "35",
        "S_END": "1.0", "GUIDANCE_EVERY_K": "1",
        "GUIDANCE_SCALE": "5.00", "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0",
        "RESIDUAL_BLEND": "0.5",
    }),

    # N6: Pure DDPM 50 steps (no guidance — step bottleneck proof)
    ("exp_N6_ddpm50_noguide", {
        "TIMESTEP_RESPACING": '"ddpm50"',
        "USE_HUBER_LOSS": "False", "USE_L1_LOSS": "False",
        "CONSTANT_SCHEDULE": "True", "GUIDANCE_EARLY_STOP": "False",
        "BLOCK_UNET_GRAD": "True", "USE_DPMSOLVER": "False", "USE_DDIM": "False",
        "HYBRID_MODE": "False",
        "S_END": "1.0", "GUIDANCE_EVERY_K": "1",
        "GUIDANCE_SCALE": "0.00", "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0",
        "RESIDUAL_BLEND": "0.5",
    }),

    # N7: Hybrid D15→DDPM50 T200
    ("exp_N7_hybrid", {
        "TIMESTEP_RESPACING": '""',
        "USE_HUBER_LOSS": "False", "USE_L1_LOSS": "False",
        "CONSTANT_SCHEDULE": "True", "GUIDANCE_EARLY_STOP": "False",
        "BLOCK_UNET_GRAD": "True", "USE_DPMSOLVER": "False", "USE_DDIM": "False",
        "HYBRID_MODE": "True",
        "DPM_SOLVER_STEPS": "15", "HYBRID_SWITCH_T": "200", "REFINE_STEPS": "50",
        "S_END": "1.0", "GUIDANCE_EVERY_K": "1",
        "GUIDANCE_SCALE": "0.10", "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0",
        "RESIDUAL_BLEND": "0.5",
    }),

    # P1_edge: Sobel edge in K=1 context
    ("exp_P1_edge_k1", {
        "TIMESTEP_RESPACING": '""',
        "USE_HUBER_LOSS": "False", "USE_L1_LOSS": "False",
        "CONSTANT_SCHEDULE": "True", "GUIDANCE_EARLY_STOP": "False",
        "BLOCK_UNET_GRAD": "True", "USE_DPMSOLVER": "False", "USE_DDIM": "False",
        "HYBRID_MODE": "False",
        "S_END": "1.0", "GUIDANCE_EVERY_K": "1",
        "GUIDANCE_SCALE": "0.10", "EDGE_WEIGHT": "0.02", "GRAD_MOMENTUM": "0.0",
        "RESIDUAL_BLEND": "0.5",
    }),

    # P1_momentum: Momentum in K=1 context
    ("exp_P1_momentum_k1", {
        "TIMESTEP_RESPACING": '""',
        "USE_HUBER_LOSS": "False", "USE_L1_LOSS": "False",
        "CONSTANT_SCHEDULE": "True", "GUIDANCE_EARLY_STOP": "False",
        "BLOCK_UNET_GRAD": "True", "USE_DPMSOLVER": "False", "USE_DDIM": "False",
        "HYBRID_MODE": "False",
        "S_END": "1.0", "GUIDANCE_EVERY_K": "1",
        "GUIDANCE_SCALE": "0.10", "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.9",
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
        "vram": float(g(r"peak_vram_mb:\s+([\d.]+)")),
        "rc": int(g(r"restorer_calls:\s+(\d+)", "0")),
    }


def run_one(name, overrides):
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    logfile = os.path.join(LOG_DIR, f"gap_{name}.log")

    full = dict(overrides)
    full["OUT_DIR"] = f'"../final_experiments/{name}"'
    modify_trainpy(full)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    result = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True, text=True, timeout=900, cwd=WORK, env=env,
    )
    with open(logfile, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n" + result.stderr)

    if result.returncode != 0:
        return None
    return extract_metrics(result.stdout)


def main():
    print("Filling paper metric gaps: 9 experiments\n")
    results = {}
    for idx, (name, overrides) in enumerate(experiments):
        short = name.replace("exp_", "")
        print(f"[{idx+1}/9] {short}")
        t0 = time.time()
        r = run_one(name, overrides)
        elapsed = time.time() - t0
        if r:
            results[name] = r
            print(f"  PSNR={r['psnr']:.2f}  SSIM={r['ssim_gt']:.4f}  "
                  f"Time={r['sec']:.1f}s  VRAM={r['vram']:.0f}MB  RC={r['rc']}  [{elapsed:.0f}s]")
        else:
            results[name] = None
            print(f"  CRASHED  [{elapsed:.0f}s]")

    print(f"\n{'='*75}")
    print(f"  RESULTS")
    print(f"  {'ID':<25s} {'PSNR':>7s} {'SSIM':>7s} {'Time':>7s} {'VRAM':>7s} {'RC':>5s}")
    print(f"  {'-'*60}")
    for name, _ in experiments:
        r = results.get(name)
        if r:
            print(f"  {name:<25s} {r['psnr']:7.2f} {r['ssim_gt']:7.4f} {r['sec']:7.1f} {r['vram']:7.0f} {r['rc']:5d}")
        else:
            print(f"  {name:<25s} {'CRASH':>7s}")

    # Key comparisons
    print(f"\n  Key comparisons:")
    ref_r2e2 = 30.65  # R2_E2
    ref_a1 = 30.32    # A1 (K=1)
    if results.get("exp_N1_ddpm100"):
        print(f"  - DDPM100 vs A1(30.32): {results['exp_N1_ddpm100']['psnr']:.2f}")
    if results.get("exp_N2_l1loss"):
        print(f"  - L1 loss vs R2_E2(30.65): {results['exp_N2_l1loss']['psnr']:.2f}")
    if results.get("exp_N3_scale10"):
        print(f"  - Scale=1.0 vs R2_E2(30.65): {results['exp_N3_scale10']['psnr']:.2f}")
    if results.get("exp_P1_edge_k1"):
        print(f"  - Edge(K=1) vs A1(30.32): {results['exp_P1_edge_k1']['psnr']:.2f}")
    if results.get("exp_P1_momentum_k1"):
        print(f"  - Momentum(K=1) vs A1(30.32): {results['exp_P1_momentum_k1']['psnr']:.2f}")


if __name__ == "__main__":
    main()
