#!/usr/bin/env python3
"""Phase 9: 7 experiments filling the final gaps in the experiment record."""
import os, sys, re, time, subprocess, shutil

WORK = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(WORK, "train.py")
OUT_ROOT = os.path.join(os.path.dirname(WORK), "final_experiments")
LOG_DIR = os.path.join(WORK, "exp_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ── BASE for experiments 2-7 (shared optimized baseline) ──
BASE = {
    "TASK": '"restoration"', "SEED": "1234",
    "LIGHTNESS_WEIGHT": "1.0", "COLOR_WEIGHT": "0.05",
    "UNMASKED_WEIGHT": "1.0", "SS_WEIGHT": "1.0",
    "REF_WEIGHT": "25.0", "OP_LIGHTNESS_WEIGHT": "1.0", "OP_COLOR_WEIGHT": "0.5",
    "N": "1", "S_START": "1.0",
    "TIMESTEP_RESPACING": '""', "USE_DDIM": "False",
    "CLIP_DENOISED": "True", "BATCH_SIZE": "1",
    "IMAGE_SIZE": "512", "DIFFUSION_STEPS": "1000",
    "IN_DIR": '"../testdata/pairs/LQ"', "GT_DIR": '"../testdata/pairs/GT"',
    "MAX_IMAGES": "1",
    "USE_DPMSOLVER": "False", "USE_FP16": "False", "HYBRID_MODE": "False",
    "RUN_TAG": '""',
}

# ── Experiment list ──
experiments = [
    # A0: True paper config (NO residual blend)
    ("exp_A0_true_paper", {
        "USE_HUBER_LOSS": "False", "CONSTANT_SCHEDULE": "True",
        "GUIDANCE_EARLY_STOP": "False", "BLOCK_UNET_GRAD": "False",
        "S_END": "0.7", "GUIDANCE_EVERY_K": "1",
        "GUIDANCE_SCALE": "0.10", "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0",
        "RESIDUAL_BLEND": "0.0",
    }),

    # A1: K=1 full guidance + BLOCK_UNET=True (VRAM saving verification)
    ("exp_A1_full_unetTrue", {
        "USE_HUBER_LOSS": "False", "CONSTANT_SCHEDULE": "True",
        "GUIDANCE_EARLY_STOP": "False", "BLOCK_UNET_GRAD": "True",
        "S_END": "0.7", "GUIDANCE_EVERY_K": "1",
        "GUIDANCE_SCALE": "0.10", "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0",
        "RESIDUAL_BLEND": "0.5",
    }),

    # B1: Sobel edge preservation
    ("exp_B1_edge", {
        "USE_HUBER_LOSS": "False", "CONSTANT_SCHEDULE": "True",
        "GUIDANCE_EARLY_STOP": "False", "BLOCK_UNET_GRAD": "True",
        "S_END": "1.0", "GUIDANCE_EVERY_K": "2",
        "GUIDANCE_SCALE": "0.25", "EDGE_WEIGHT": "0.02", "GRAD_MOMENTUM": "0.0",
        "RESIDUAL_BLEND": "0.5",
    }),

    # B2: Gradient momentum
    ("exp_B2_momentum", {
        "USE_HUBER_LOSS": "False", "CONSTANT_SCHEDULE": "True",
        "GUIDANCE_EARLY_STOP": "False", "BLOCK_UNET_GRAD": "True",
        "S_END": "1.0", "GUIDANCE_EVERY_K": "2",
        "GUIDANCE_SCALE": "0.25", "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.9",
        "RESIDUAL_BLEND": "0.5",
    }),

    # D1: DDIM
    ("exp_D1_ddim", {
        "USE_HUBER_LOSS": "False", "USE_DDIM": "True",
        "CONSTANT_SCHEDULE": "False",
        "GUIDANCE_EARLY_STOP": "False", "BLOCK_UNET_GRAD": "True",
        "S_END": "1.0", "GUIDANCE_EVERY_K": "1",
        "GUIDANCE_SCALE": "0.10", "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0",
        "RESIDUAL_BLEND": "0.5",
    }),

    # D2: K=1 + early stop (old V2 SOTA test)
    ("exp_D2_k1_earlystop", {
        "USE_HUBER_LOSS": "False", "CONSTANT_SCHEDULE": "True",
        "GUIDANCE_EARLY_STOP": "True", "BLOCK_UNET_GRAD": "True",
        "S_END": "0.5", "GUIDANCE_EVERY_K": "1",
        "GUIDANCE_SCALE": "0.10", "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0",
        "RESIDUAL_BLEND": "0.5",
    }),

    # D3: K=2 + early stop (MSE+CONSTANT framework)
    ("exp_D3_k2_earlystop", {
        "USE_HUBER_LOSS": "False", "CONSTANT_SCHEDULE": "True",
        "GUIDANCE_EARLY_STOP": "True", "BLOCK_UNET_GRAD": "True",
        "S_END": "0.5", "GUIDANCE_EVERY_K": "2",
        "GUIDANCE_SCALE": "0.25", "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0",
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
    logfile = os.path.join(LOG_DIR, f"phase9_{name}.log")

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
        print(f"  CRASHED")
        return None

    r = extract_metrics(result.stdout)
    print(f"  PSNR={r['psnr']:.2f}  SSIM={r['ssim_gt']:.4f}  NIQE={r['niqe']:.0f}  "
          f"Time={r['sec']:.1f}s  VRAM={r['vram']:.0f}MB  RC={r['rc']}")

    # Copy output image to experiment dir (train.py saved to OUT_DIR directly)
    src = os.path.join(out_dir, "0000_00010.png")
    if not os.path.isfile(src):
        project_root = os.path.dirname(WORK)
        alt_src = os.path.join(project_root, "final_experiments", name, "0000_00010.png")
        if os.path.isfile(alt_src):
            r["img_saved"] = True
    return r


def main():
    print("Phase 9: 7 experiments for final gap-filling")
    print(f"Output dir: {OUT_ROOT}")
    print()

    results = {}
    for idx, (name, overrides) in enumerate(experiments):
        short = name.replace("exp_", "")
        print(f"[{idx+1}/7] {short}")
        t0 = time.time()
        r = run_one(name, overrides)
        elapsed = time.time() - t0
        if r:
            results[name] = r
            print(f"      [{elapsed:.0f}s]")
        else:
            results[name] = None
            print(f"      [FAIL] {elapsed:.0f}s")

    print(f"\n{'='*80}")
    print("  PHASE 9 COMPLETE — All Paper Metrics")
    print(f"{'='*80}")
    header = f"  {'ID':<25s} {'PSNR':>7s} {'SSIM':>7s} {'NIQE':>7s} {'Time':>7s} {'VRAM':>7s} {'RC':>5s}"
    print(header)
    print("  " + "-" * 73)
    for name, _ in experiments:
        r = results.get(name)
        if r:
            print(f"  {name:<25s} {r['psnr']:7.2f} {r['ssim_gt']:7.4f} {r['niqe']:7.0f} "
                  f"{r['sec']:7.1f} {r['vram']:7.0f} {r['rc']:5d}")
        else:
            print(f"  {name:<25s} {'CRASH':>7s}")

    # Summary: compare key pairs
    print(f"\n  Key comparisons:")
    if results.get("exp_A0_true_paper") and results.get("exp_A1_full_unetTrue"):
        a0 = results["exp_A0_true_paper"]
        a1 = results["exp_A1_full_unetTrue"]
        print(f"  - BLOCK_UNET effect: PSNR {a0['psnr']:.2f}→{a1['psnr']:.2f} ({a1['psnr']-a0['psnr']:+.2f}), "
              f"VRAM {a0['vram']:.0f}→{a1['vram']:.0f}MB ({a1['vram']-a0['vram']:+.0f})")
    if results.get("exp_A0_true_paper") and results.get("exp_P0_paper"):
        # we need P0 data from earlier benchmark
        pass
    if results.get("exp_B1_edge"):
        b1 = results["exp_B1_edge"]
        print(f"  - Edge vs R2_E2(30.65): PSNR {b1['psnr']:.2f} ({b1['psnr']-30.65:+.2f})")
    if results.get("exp_B2_momentum"):
        b2 = results["exp_B2_momentum"]
        print(f"  - Momentum vs R2_E2(30.65): PSNR {b2['psnr']:.2f} ({b2['psnr']-30.65:+.2f})")


if __name__ == "__main__":
    main()
