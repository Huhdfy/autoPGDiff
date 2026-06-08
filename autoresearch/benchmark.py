#!/usr/bin/env python3
"""
Benchmark: 12 experiment configs tested with paper-standard metrics (PSNR, SSIM_gt, NIQE).
Each config runs on a single LQ+GT pair: testdata/pairs/LQ/0000_00010.png / GT/0000_00010.png.
"""
import os, sys, re, time, subprocess

WORK = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(WORK, "train.py")
RESULTS_TSV = os.path.join(WORK, "benchmark_results.tsv")

# ── Common optimized baseline params ──
# (applied as base; each config overrides specific keys)
BASE = {
    "TASK": '"restoration"',
    "SEED": "1234",
    "GUIDANCE_SCALE": "0.10",
    "LIGHTNESS_WEIGHT": "1.0",
    "COLOR_WEIGHT": "0.05",
    "UNMASKED_WEIGHT": "1.0",
    "SS_WEIGHT": "1.0",
    "EDGE_WEIGHT": "0.0",
    "GRAD_MOMENTUM": "0.0",
    "REF_WEIGHT": "25.0",
    "OP_LIGHTNESS_WEIGHT": "1.0",
    "OP_COLOR_WEIGHT": "0.5",
    "RESIDUAL_BLEND": "0.5",
    "USE_HUBER_LOSS": "True",
    "N": "1",
    "S_START": "1.0",
    "S_END": "0.5",
    "TIMESTEP_RESPACING": '""',
    "USE_DDIM": "False",
    "CLIP_DENOISED": "True",
    "BATCH_SIZE": "1",
    "IMAGE_SIZE": "512",
    "DIFFUSION_STEPS": "1000",
    "IN_DIR": '"../testdata/pairs/LQ"',
    "GT_DIR": '"../testdata/pairs/GT"',
    "MAX_IMAGES": "1",
    "BLOCK_UNET_GRAD": "True",
    "USE_DPMSOLVER": "False",
    "DPM_SOLVER_STEPS": "35",
    "RESTORER_T_ZERO": "False",
    "DPM_FIRST_ORDER": "False",
    "CONSTANT_SCHEDULE": "False",
    "HYBRID_MODE": "False",
    "HYBRID_SWITCH_T": "400",
    "REFINE_STEPS": "50",
    "GUIDANCE_EVERY_K": "1",
    "SKIP_ZERO_SCALE": "True",
    "GUIDANCE_EARLY_STOP": "True",
    "USE_FP16": "False",
    "OUT_DIR": '"../results/experiment"',
    "RUN_TAG": '""',
    "RUNS_DIR": '"../runs"',
}

EXPERIMENTS = [
    # ── P0: Original paper config ──
    ("P0_paper", "Original PGDiff: MSE, CONSTANT schedule, no early-stop, BLOCK_UNET=False",
     {"USE_HUBER_LOSS": "False", "CONSTANT_SCHEDULE": "True",
      "GUIDANCE_EARLY_STOP": "False", "BLOCK_UNET_GRAD": "False",
      "S_END": "0.7", "GUIDANCE_EVERY_K": "1"}),

    # ── 1: Optimized baseline (loss + schedule + early-stop + block_unet) ──
    ("C1_opt_base", "Optimized: Huber, LinearSched, s=0.5, BLOCK_UNET=True, K=1",
     {"USE_HUBER_LOSS": "True", "CONSTANT_SCHEDULE": "False",
      "GUIDANCE_EARLY_STOP": "True", "BLOCK_UNET_GRAD": "True",
      "S_END": "0.5", "GUIDANCE_EVERY_K": "1"}),

    # ── 2: Sparse guidance K=2 ──
    ("C2_K2", "Sparse: K=2, full-step guidance",
     {"USE_HUBER_LOSS": "True", "CONSTANT_SCHEDULE": "False",
      "GUIDANCE_EARLY_STOP": "False", "BLOCK_UNET_GRAD": "True",
      "S_END": "1.0", "GUIDANCE_EVERY_K": "2"}),

    # ── 3: Combo: early-stop + sparse ──
    ("C3_s05K3", "Combo best: s=0.5, K=3",
     {"USE_HUBER_LOSS": "True", "CONSTANT_SCHEDULE": "False",
      "GUIDANCE_EARLY_STOP": "True", "BLOCK_UNET_GRAD": "True",
      "S_END": "0.5", "GUIDANCE_EVERY_K": "3"}),

    # ── 4: Large scale ──
    ("C4_bigscale", "Scale=0.20: K=2, stronger guidance",
     {"GUIDANCE_SCALE": "0.20",  # override only scale
      "USE_HUBER_LOSS": "True", "CONSTANT_SCHEDULE": "False",
      "GUIDANCE_EARLY_STOP": "False", "BLOCK_UNET_GRAD": "True",
      "S_END": "1.0", "GUIDANCE_EVERY_K": "2"}),

    # ── 5: Pure diffusion (no guidance) ──
    ("C5_noguide", "Pure diffusion: scale=0, no restorer calls",
     {"GUIDANCE_SCALE": "0.0",
      "USE_HUBER_LOSS": "True", "CONSTANT_SCHEDULE": "False",
      "GUIDANCE_EARLY_STOP": "True", "BLOCK_UNET_GRAD": "True",
      "S_END": "0.5", "GUIDANCE_EVERY_K": "1"}),

    # ── 6: DPM-Solver ──
    ("C6_dpm100", "DPM-Solver-2: 100 steps",
     {"USE_DPMSOLVER": "True", "DPM_SOLVER_STEPS": "100",
      "GUIDANCE_EARLY_STOP": "False", "S_END": "1.0",
      "GUIDANCE_EVERY_K": "1", "USE_HUBER_LOSS": "True",
      "CONSTANT_SCHEDULE": "False", "BLOCK_UNET_GRAD": "True"}),

    # ── 7: DDPM 200 steps ──
    ("C7_ddpm200", "DDPM 200: 5x speedup, s=0.5, K=1",
     {"TIMESTEP_RESPACING": '"ddpm200"',
      "USE_HUBER_LOSS": "True", "CONSTANT_SCHEDULE": "False",
      "GUIDANCE_EARLY_STOP": "True", "BLOCK_UNET_GRAD": "True",
      "S_END": "0.5", "GUIDANCE_EVERY_K": "1"}),

    # ── 8: Low scale ──
    ("C8_lowscale", "Scale=0.05: minimal guidance",
     {"GUIDANCE_SCALE": "0.05",
      "USE_HUBER_LOSS": "True", "CONSTANT_SCHEDULE": "False",
      "GUIDANCE_EARLY_STOP": "True", "BLOCK_UNET_GRAD": "True",
      "S_END": "0.5", "GUIDANCE_EVERY_K": "1"}),

    # ── 9: s/K search: K=5 ──
    ("C9_K5", "s/K search: K=5, full-step",
     {"GUIDANCE_EARLY_STOP": "False", "S_END": "1.0",
      "GUIDANCE_EVERY_K": "5",
      "USE_HUBER_LOSS": "True", "CONSTANT_SCHEDULE": "False",
      "BLOCK_UNET_GRAD": "True"}),

    # ── 10: s/K search: s=0.7 ──
    ("C10_s07", "s/K search: s=0.7, K=1",
     {"S_END": "0.7", "GUIDANCE_EVERY_K": "1",
      "GUIDANCE_EARLY_STOP": "True",
      "USE_HUBER_LOSS": "True", "CONSTANT_SCHEDULE": "False",
      "BLOCK_UNET_GRAD": "True"}),

    # ── 11: s/K search: s=0.5, K=2 ──
    ("C11_s05K2", "s/K search: s=0.5, K=2",
     {"S_END": "0.5", "GUIDANCE_EVERY_K": "2",
      "GUIDANCE_EARLY_STOP": "True",
      "USE_HUBER_LOSS": "True", "CONSTANT_SCHEDULE": "False",
      "BLOCK_UNET_GRAD": "True"}),
]


def modify_trainpy(overrides):
    """Apply BASE + overrides to train.py via regex."""
    with open(TRAIN_PY, "r") as f:
        content = f.read()

    full = dict(BASE)
    full.update(overrides)

    # String variables
    for var, val in full.items():
        if val.startswith('"') and val.endswith('"'):
            content = re.sub(rf'{var} = "[^"]*"', f'{var} = {val}', content)

    # Boolean variables
    for var, val in full.items():
        if val in ("True", "False"):
            content = re.sub(rf'{var} = (True|False)', f'{var} = {val}', content)

    # Numeric / float variables
    for var, val in full.items():
        if not val.startswith('"') and val not in ("True", "False"):
            content = re.sub(rf'{var} = [\d.]+', f'{var} = {val}', content)

    with open(TRAIN_PY, "w") as f:
        f.write(content)


def run_one(name, desc):
    """Run train.py with current config, extract paper metrics + speed."""
    logfile = os.path.join(WORK, f"bench_{name}.log")
    t0 = time.time()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    result = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True, text=True, timeout=900, cwd=WORK, env=env,
    )
    elapsed = time.time() - t0

    with open(logfile, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n" + result.stderr)

    if result.returncode != 0:
        print(f"  CRASHED (exit={result.returncode})")
        stderr_tail = result.stderr[-300:] if result.stderr else "(none)"
        print(f"  stderr: {stderr_tail}")
        return None

    output = result.stdout

    def g(pat, default="0"):
        m = re.search(pat, output)
        return m.group(1) if m else default

    r = {
        "name": name,
        "v2": float(g(r"quality_score_v2:\s+([\d.]+)")),
        "psnr": float(g(r"psnr:\s+([\d.]+)")),
        "ssim_gt": float(g(r"ssim_gt:\s+([\d.]+)")),
        "niqe": float(g(r"niqe:\s+([\d.]+)")),
        "sec": float(g(r"total_seconds:\s+([\d.]+)")),
        "vram": float(g(r"peak_vram_mb:\s+([\d.]+)")),
        "gcalls": int(g(r"guidance_calls:\s+(\d+)", "0")),
        "rcalls": int(g(r"restorer_calls:\s+(\d+)", "0")),
    }

    print(f"  {name:<14s} PSNR={r['psnr']:.2f}  SSIM_gt={r['ssim_gt']:.4f}  "
          f"NIQE={r['niqe']:.1f}  V2={r['v2']:.4f}  "
          f"Time={r['sec']:.1f}s  GC={r['gcalls']}  RC={r['rcalls']}")
    return r


def print_final_summary(results):
    """Print sorted summary tables."""
    print(f"\n\n{'='*110}")
    print("  FINAL BENCHMARK RESULTS  (PSNR, SSIM_gt: higher=better | NIQE: lower=better)")
    print(f"{'='*110}")

    if not results:
        print("\n  No successful runs. All experiments crashed.")
        return

    # ── Sorted by PSNR ──
    print(f"\n  Ranked by PSNR (paper primary):")
    print(f"  {'Rank':<5s} {'Config':<16s} {'PSNR↑':>7s} {'SSIM_gt↑':>9s} "
          f"{'NIQE↓':>8s} {'V2':>7s} {'Time(s)':>8s} {'GCalls':>7s} {'RCalls':>7s} {'VRAM(MB)':>9s}")
    print("  " + "-" * 105)
    sorted_psnr = sorted(results, key=lambda x: x["psnr"], reverse=True)
    for i, r in enumerate(sorted_psnr):
        print(f"  {i+1:<5d} {r['name']:<16s} {r['psnr']:7.2f} {r['ssim_gt']:9.4f} "
              f"{r['niqe']:8.0f} {r['v2']:7.4f} {r['sec']:8.1f} "
              f"{r['gcalls']:7d} {r['rcalls']:7d} {r['vram']:9.0f}")

    # ── Sorted by SSIM_gt ──
    print(f"\n  Ranked by SSIM_gt:")
    print(f"  {'Rank':<5s} {'Config':<16s} {'SSIM_gt↑':>9s} {'PSNR↑':>7s} "
          f"{'NIQE↓':>8s} {'V2':>7s} {'Time(s)':>8s} {'GCalls':>7s} {'RCalls':>7s} {'VRAM(MB)':>9s}")
    print("  " + "-" * 105)
    sorted_ssim = sorted(results, key=lambda x: x["ssim_gt"], reverse=True)
    for i, r in enumerate(sorted_ssim):
        print(f"  {i+1:<5d} {r['name']:<16s} {r['ssim_gt']:9.4f} {r['psnr']:7.2f} "
              f"{r['niqe']:8.0f} {r['v2']:7.4f} {r['sec']:8.1f} "
              f"{r['gcalls']:7d} {r['rcalls']:7d} {r['vram']:9.0f}")

    # ── Best per direction ──
    print(f"\n  Best per direction summary:")
    print(f"  {'Direction':<30s} {'Config':<16s} {'PSNR':>7s} {'SSIM_gt':>9s} {'NIQE':>8s} {'Time':>8s}")
    print("  " + "-" * 80)

    # P0 = paper baseline
    paper = next(r for r in results if r["name"] == "P0_paper")
    # Best of optimized (C1-C11)
    best = max([r for r in results if r["name"] != "P0_paper"], key=lambda x: x["psnr"])
    # Best s×K combo (C3, C9, C10, C11)
    combos = [r for r in results if r["name"] in ["C3_s05K3", "C9_K5", "C10_s07", "C11_s05K2"]]
    best_combo = max(combos, key=lambda x: x["psnr"]) if combos else None
    # Speed champion
    speed_champ = min(results, key=lambda x: x["sec"])
    # Quality/speed Pareto
    pareto = [r for r in results if r["psnr"] > 20 and r["sec"] < 80]
    pareto_best = max(pareto, key=lambda x: x["psnr"] / x["sec"]) if pareto else None

    print(f"  {'Paper baseline':<30s} {paper['name']:<16s} {paper['psnr']:7.2f} "
          f"{paper['ssim_gt']:9.4f} {paper['niqe']:8.0f} {paper['sec']:8.1f}")
    print(f"  {'Best PSNR (optimized)':<30s} {best['name']:<16s} {best['psnr']:7.2f} "
          f"{best['ssim_gt']:9.4f} {best['niqe']:8.0f} {best['sec']:8.1f}")
    if best_combo:
        print(f"  {'Best s x K combo':<30s} {best_combo['name']:<16s} {best_combo['psnr']:7.2f} "
              f"{best_combo['ssim_gt']:9.4f} {best_combo['niqe']:8.0f} {best_combo['sec']:8.1f}")
    print(f"  {'Fastest':<30s} {speed_champ['name']:<16s} {speed_champ['psnr']:7.2f} "
          f"{speed_champ['ssim_gt']:9.4f} {speed_champ['niqe']:8.0f} {speed_champ['sec']:8.1f}")
    if pareto_best:
        print(f"  {'Pareto (quality/speed)':<30s} {pareto_best['name']:<16s} {pareto_best['psnr']:7.2f} "
              f"{pareto_best['ssim_gt']:9.4f} {pareto_best['niqe']:8.0f} {pareto_best['sec']:8.1f}")


def main():
    results = []

    # Verify configs compile
    for name, desc, overrides in EXPERIMENTS:
        modify_trainpy(overrides)
        compiled = subprocess.run(
            [sys.executable, "-c",
             "import py_compile; py_compile.compile('/mnt/workspace/autoPGDiff/autoresearch/train.py', doraise=True)"],
            capture_output=True, text=True,
        )
        if compiled.returncode != 0:
            print(f"  SYNTAX ERROR in {name}: {compiled.stderr[-300:]}")
            return

    print(f"  All {len(EXPERIMENTS)} configs syntax OK. Starting benchmark...")
    print(f"  Estimated total: ~19 min\n")

    for name, desc, overrides in EXPERIMENTS:
        print(f"[{len(results)+1}/{len(EXPERIMENTS)}] {name}: {desc}")
        t_start = time.time()
        modify_trainpy(overrides)
        r = run_one(name, desc)
        elapsed = time.time() - t_start
        if r:
            results.append(r)
            print(f"      [OK] {elapsed:.0f}s")
        else:
            print(f"      [FAIL] {elapsed:.0f}s")

    # Restore optimized baseline
    last = EXPERIMENTS[1][2]
    modify_trainpy(last)

    # ── Write TSV ──
    with open(RESULTS_TSV, "w") as f:
        f.write("name\tv2\tpsnr\tssim_gt\tniqe\tsec\tvram\tgcalls\trcalls\tdesc\n")
        for r in results:
            name = r["name"]
            idx = [e[0] for e in EXPERIMENTS].index(name)
            desc = EXPERIMENTS[idx][1]
            f.write(f"{r['name']}\t{r['v2']:.4f}\t{r['psnr']:.2f}\t{r['ssim_gt']:.4f}\t"
                    f"{r['niqe']:.0f}\t{r['sec']:.1f}\t{r['vram']:.0f}\t"
                    f"{r['gcalls']}\t{r['rcalls']}\t{desc}\n")

    print_final_summary(results)


if __name__ == "__main__":
    main()
