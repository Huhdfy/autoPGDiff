#!/usr/bin/env python3
"""Auto-experiment: Find config Pareto-better than C4 (PSNR>=28.04, Time<=98s)."""
import os, sys, re, time, subprocess

WORK = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(WORK, "train.py")
LOG_DIR = os.path.join(WORK, "exp_logs")

C4_PSNR = 28.04
C4_TIME = 98.4

BASE = {
    "TASK": '"restoration"', "SEED": "1234",
    "LIGHTNESS_WEIGHT": "1.0", "COLOR_WEIGHT": "0.05",
    "UNMASKED_WEIGHT": "1.0", "SS_WEIGHT": "1.0",
    "EDGE_WEIGHT": "0.0", "GRAD_MOMENTUM": "0.0",
    "N": "1", "S_START": "1.0", "S_END": "1.0",
    "TIMESTEP_RESPACING": '""', "USE_DDIM": "False",
    "CLIP_DENOISED": "True", "BATCH_SIZE": "1",
    "IMAGE_SIZE": "512", "DIFFUSION_STEPS": "1000",
    "IN_DIR": '"../testdata/pairs/LQ"', "GT_DIR": '"../testdata/pairs/GT"',
    "MAX_IMAGES": "1",
    "BLOCK_UNET_GRAD": "True",
    "USE_DPMSOLVER": "False", "USE_FP16": "False",
    "HYBRID_MODE": "False", "USE_DDIM": "False",
    "OUT_DIR": '"../results/exp"',
    "RUN_TAG": '""',
}

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
        "ssim": float(g(r"ssim_gt:\s+([\d.]+)")),
        "niqe": float(g(r"niqe:\s+([\d.]+)")),
        "v2": float(g(r"quality_score_v2:\s+([\d.]+)")),
        "sec": float(g(r"total_seconds:\s+([\d.]+)")),
        "vram": float(g(r"peak_vram_mb:\s+([\d.]+)")),
        "gc": int(g(r"guidance_calls:\s+(\d+)", "0")),
        "rc": int(g(r"restorer_calls:\s+(\d+)", "0")),
    }


def run_one(name, desc, overrides, out_dir):
    os.makedirs(LOG_DIR, exist_ok=True)
    logfile = os.path.join(LOG_DIR, f"{name}.log")
    full_overrides = dict(overrides)
    full_overrides["OUT_DIR"] = f'"{out_dir}"'
    modify_trainpy(full_overrides)
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


def compare(name, r, baseline_psnr=C4_PSNR, baseline_time=C4_TIME):
    if r is None:
        print(f"  [{name}] CRASHED")
        return False
    dp = r["psnr"] - baseline_psnr
    dt = baseline_time - r["sec"]
    psnr_ok = r["psnr"] >= baseline_psnr
    time_ok = r["sec"] <= baseline_time
    status = "WIN" if (psnr_ok and time_ok) else ("QUALITY+" if psnr_ok else ("SPEED+" if time_ok else "LOSS"))
    print(f"  [{name}] PSNR={r['psnr']:.2f} ({dp:+.2f})  Time={r['sec']:.1f}s ({dt:+.1f})  "
          f"SSIM={r['ssim']:.4f}  RC={r['rc']}  VRAM={r['vram']:.0f}MB  -> {status}")
    return psnr_ok and time_ok


# ============================================================
experiments = [
    # Round 1: MSE + K=2 + scale sweep
    ("R1_E1_MSE_K2_s20", "MSE, K=2, scale=0.20, no early-stop [*** WINNER ***]",
     {"GUIDANCE_SCALE": "0.20", "GUIDANCE_EVERY_K": "2",
      "USE_HUBER_LOSS": "False", "CONSTANT_SCHEDULE": "False",
      "GUIDANCE_EARLY_STOP": "False", "S_END": "1.0"}),

    ("R1_E2_MSE_K2_s25", "MSE, K=2, scale=0.25, no early-stop",
     {"GUIDANCE_SCALE": "0.25", "GUIDANCE_EVERY_K": "2",
      "USE_HUBER_LOSS": "False", "CONSTANT_SCHEDULE": "False",
      "GUIDANCE_EARLY_STOP": "False", "S_END": "1.0"}),

    ("R1_E3_MSE_K2_s15", "MSE, K=2, scale=0.15, no early-stop",
     {"GUIDANCE_SCALE": "0.15", "GUIDANCE_EVERY_K": "2",
      "USE_HUBER_LOSS": "False", "CONSTANT_SCHEDULE": "False",
      "GUIDANCE_EARLY_STOP": "False", "S_END": "1.0"}),

    ("R1_E4_MSE_K2_s30", "MSE, K=2, scale=0.30, no early-stop",
     {"GUIDANCE_SCALE": "0.30", "GUIDANCE_EVERY_K": "2",
      "USE_HUBER_LOSS": "False", "CONSTANT_SCHEDULE": "False",
      "GUIDANCE_EARLY_STOP": "False", "S_END": "1.0"}),

    # Round 2: CONSTANT schedule (P0's setting) + MSE + K=2
    ("R2_E1_constant_s20", "MSE, K=2, scale=0.20, CONSTANT schedule",
     {"GUIDANCE_SCALE": "0.20", "GUIDANCE_EVERY_K": "2",
      "USE_HUBER_LOSS": "False", "CONSTANT_SCHEDULE": "True",
      "GUIDANCE_EARLY_STOP": "False", "S_END": "1.0"}),

    ("R2_E2_constant_s25", "MSE, K=2, scale=0.25, CONSTANT schedule",
     {"GUIDANCE_SCALE": "0.25", "GUIDANCE_EVERY_K": "2",
      "USE_HUBER_LOSS": "False", "CONSTANT_SCHEDULE": "True",
      "GUIDANCE_EARLY_STOP": "False", "S_END": "1.0"}),

    # Round 3: Push speed - K=3 with higher scale
    ("R3_E1_MSE_K3_s30", "MSE, K=3, scale=0.30, LINEAR schedule",
     {"GUIDANCE_SCALE": "0.30", "GUIDANCE_EVERY_K": "3",
      "USE_HUBER_LOSS": "False", "CONSTANT_SCHEDULE": "False",
      "GUIDANCE_EARLY_STOP": "False", "S_END": "1.0"}),

    ("R3_E2_MSE_K3_s35", "MSE, K=3, scale=0.35, LINEAR schedule",
     {"GUIDANCE_SCALE": "0.35", "GUIDANCE_EVERY_K": "3",
      "USE_HUBER_LOSS": "False", "CONSTANT_SCHEDULE": "False",
      "GUIDANCE_EARLY_STOP": "False", "S_END": "1.0"}),

    # Round 4: K=2 + early-stop combos with MSE
    ("R4_E1_MSE_K2_s25_e08", "MSE, K=2, scale=0.25, s_end=0.8",
     {"GUIDANCE_SCALE": "0.25", "GUIDANCE_EVERY_K": "2",
      "USE_HUBER_LOSS": "False", "CONSTANT_SCHEDULE": "False",
      "GUIDANCE_EARLY_STOP": "True", "S_END": "0.8"}),

    ("R4_E2_MSE_K2_s25_e07", "MSE, K=2, scale=0.25, s_end=0.7",
     {"GUIDANCE_SCALE": "0.25", "GUIDANCE_EVERY_K": "2",
      "USE_HUBER_LOSS": "False", "CONSTANT_SCHEDULE": "False",
      "GUIDANCE_EARLY_STOP": "True", "S_END": "0.7"}),
]

# ============================================================
if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(WORK), "results"), exist_ok=True)

    best = None
    for exp_id, (name, desc, overrides) in enumerate(experiments):
        out_dir = f"../results/exp_{name}"
        print(f"\n[{exp_id+1}/{len(experiments)}] {name}: {desc}")
        t0 = time.time()
        r = run_one(name, desc, overrides, out_dir)
        elapsed = time.time() - t0

        if r:
            win = r["psnr"] >= C4_PSNR and r["sec"] <= C4_TIME
            if win:
                print(f"  >>> WIN: {name} beats C4 (PSNR +{r['psnr']-C4_PSNR:.2f}, Time -{C4_TIME-r['sec']:.1f}s) <<<")
                if best is None or r["psnr"] > best["psnr"]:
                    best = r
                    import shutil
                    shutil.copy(TRAIN_PY, os.path.join(LOG_DIR, f"best_{name}_train.py"))
            compare(name, r)
            print(f"      [{elapsed:.0f}s]")

    if best:
        print(f"\n{'='*70}")
        print(f"  BEST WINNER: PSNR={best['psnr']:.2f} dB  Time={best['sec']:.1f}s  "
              f"SSIM={best['ssim']:.4f}  RC={best['rc']}")
        print(f"  vs C4:       PSNR +{best['psnr']-C4_PSNR:.2f} dB  Time -{C4_TIME-best['sec']:.1f}s")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print(f"  No Pareto winner found. C4 remains unbeaten.")
        print(f"{'='*70}")
