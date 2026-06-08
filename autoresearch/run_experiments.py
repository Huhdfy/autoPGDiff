#!/usr/bin/env python3
"""Experiment runner: modifies train.py via regex, runs, extracts results."""
import os, sys, re, time, subprocess

WORK = "/mnt/workspace/autoPGDiff/autoresearch"
TRAIN_PY = os.path.join(WORK, "train.py")
RESULTS_TSV = os.path.join(WORK, "results_v2.tsv")

EXPERIMENTS = [
    # === Performance: Quality optimization ===
    ("baseline",    "Baseline: DDPM1000, s_end=0.5, K=1, scale=0.1",  # config tuple below
     {"S_END": "0.5", "GUIDANCE_EVERY_K": "1", "GUIDANCE_SCALE": "0.1",
      "GRAD_MOMENTUM": "0.0", "EDGE_WEIGHT": "0.0",
      "TIMESTEP_RESPACING": '""', "USE_DPMSOLVER": "False", "BLOCK_UNET_GRAD": "True",
      "MAX_IMAGES": "1", "OUT_DIR": '"../results/experiment"'}),

    ("E1_k2",       "DDPM1000 s_end=0.5 K=2: sparse guidance every 2 steps",
     {"S_END": "0.5", "GUIDANCE_EVERY_K": "2", "GUIDANCE_SCALE": "0.1",
      "GRAD_MOMENTUM": "0.0", "EDGE_WEIGHT": "0.0",
      "TIMESTEP_RESPACING": '""', "USE_DPMSOLVER": "False", "BLOCK_UNET_GRAD": "True",
      "MAX_IMAGES": "1", "OUT_DIR": '"../results/experiment"'}),

    ("E2_mom03",    "DDPM1000 s_end=0.5 K=1 momentum=0.3: light smoothing",
     {"S_END": "0.5", "GUIDANCE_EVERY_K": "1", "GUIDANCE_SCALE": "0.1",
      "GRAD_MOMENTUM": "0.3", "EDGE_WEIGHT": "0.0",
      "TIMESTEP_RESPACING": '""', "USE_DPMSOLVER": "False", "BLOCK_UNET_GRAD": "True",
      "MAX_IMAGES": "1", "OUT_DIR": '"../results/experiment"'}),

    ("E3_mom01",    "DDPM1000 s_end=0.5 K=1 momentum=0.1: minimal smoothing",
     {"S_END": "0.5", "GUIDANCE_EVERY_K": "1", "GUIDANCE_SCALE": "0.1",
      "GRAD_MOMENTUM": "0.1", "EDGE_WEIGHT": "0.0",
      "TIMESTEP_RESPACING": '""', "USE_DPMSOLVER": "False", "BLOCK_UNET_GRAD": "True",
      "MAX_IMAGES": "1", "OUT_DIR": '"../results/experiment"'}),

    ("E4_s04",      "DDPM1000 s_end=0.4 K=1: earlier guidance stop",
     {"S_END": "0.4", "GUIDANCE_EVERY_K": "1", "GUIDANCE_SCALE": "0.1",
      "GRAD_MOMENTUM": "0.0", "EDGE_WEIGHT": "0.0",
      "TIMESTEP_RESPACING": '""', "USE_DPMSOLVER": "False", "BLOCK_UNET_GRAD": "True",
      "MAX_IMAGES": "1", "OUT_DIR": '"../results/experiment"'}),

    ("E5_s06",      "DDPM1000 s_end=0.6 K=1: later guidance stop",
     {"S_END": "0.6", "GUIDANCE_EVERY_K": "1", "GUIDANCE_SCALE": "0.1",
      "GRAD_MOMENTUM": "0.0", "EDGE_WEIGHT": "0.0",
      "TIMESTEP_RESPACING": '""', "USE_DPMSOLVER": "False", "BLOCK_UNET_GRAD": "True",
      "MAX_IMAGES": "1", "OUT_DIR": '"../results/experiment"'}),

    # === Acceleration: speed optimization ===
    ("E6_ddpm500",  "DDPM 500 steps s_end=0.5 K=1: 2x speedup",
     {"S_END": "0.5", "GUIDANCE_EVERY_K": "1", "GUIDANCE_SCALE": "0.1",
      "GRAD_MOMENTUM": "0.0", "EDGE_WEIGHT": "0.0",
      "TIMESTEP_RESPACING": '"ddpm500"', "USE_DPMSOLVER": "False", "BLOCK_UNET_GRAD": "True",
      "MAX_IMAGES": "1", "OUT_DIR": '"../results/experiment"'}),

    ("E7_ddpm300",  "DDPM 300 steps s_end=0.5 K=1: 3.3x speedup",
     {"S_END": "0.5", "GUIDANCE_EVERY_K": "1", "GUIDANCE_SCALE": "0.1",
      "GRAD_MOMENTUM": "0.0", "EDGE_WEIGHT": "0.0",
      "TIMESTEP_RESPACING": '"ddpm300"', "USE_DPMSOLVER": "False", "BLOCK_UNET_GRAD": "True",
      "MAX_IMAGES": "1", "OUT_DIR": '"../results/experiment"'}),

    ("E8_ddpm300k2","DDPM 300 steps s_end=0.5 K=2: dual 3.3x+2x speedup",
     {"S_END": "0.5", "GUIDANCE_EVERY_K": "2", "GUIDANCE_SCALE": "0.1",
      "GRAD_MOMENTUM": "0.0", "EDGE_WEIGHT": "0.0",
      "TIMESTEP_RESPACING": '"ddpm300"', "USE_DPMSOLVER": "False", "BLOCK_UNET_GRAD": "True",
      "MAX_IMAGES": "1", "OUT_DIR": '"../results/experiment"'}),

    ("E9_ddpm500k2","DDPM 500 steps s_end=0.5 K=2: 2x+2x speedup",
     {"S_END": "0.5", "GUIDANCE_EVERY_K": "2", "GUIDANCE_SCALE": "0.1",
      "GRAD_MOMENTUM": "0.0", "EDGE_WEIGHT": "0.0",
      "TIMESTEP_RESPACING": '"ddpm500"', "USE_DPMSOLVER": "False", "BLOCK_UNET_GRAD": "True",
      "MAX_IMAGES": "1", "OUT_DIR": '"../results/experiment"'}),

    ("E10_ddpm200", "DDPM 200 steps s_end=0.5 K=1: 5x speedup",
     {"S_END": "0.5", "GUIDANCE_EVERY_K": "1", "GUIDANCE_SCALE": "0.1",
      "GRAD_MOMENTUM": "0.0", "EDGE_WEIGHT": "0.0",
      "TIMESTEP_RESPACING": '"ddpm200"', "USE_DPMSOLVER": "False", "BLOCK_UNET_GRAD": "True",
      "MAX_IMAGES": "1", "OUT_DIR": '"../results/experiment"'}),

    ("E11_dpm100",  "DPM-Solver 100 steps K=1: ~10x speedup (replication)",
     {"S_END": "0.0", "S_START": "1.0", "GUIDANCE_EVERY_K": "1",
      "GUIDANCE_SCALE": "0.1", "GUIDANCE_EARLY_STOP": "False",
      "GRAD_MOMENTUM": "0.0", "EDGE_WEIGHT": "0.0",
      "TIMESTEP_RESPACING": '""', "USE_DPMSOLVER": "True", "DPM_SOLVER_STEPS": "100",
      "BLOCK_UNET_GRAD": "True",
      "MAX_IMAGES": "1", "OUT_DIR": '"../results/experiment"'}),

    ("E12_ddpm300k3","DDPM 300 steps s_end=0.5 K=3: aggressive accel",
     {"S_END": "0.5", "GUIDANCE_EVERY_K": "3", "GUIDANCE_SCALE": "0.1",
      "GRAD_MOMENTUM": "0.0", "EDGE_WEIGHT": "0.0",
      "TIMESTEP_RESPACING": '"ddpm300"', "USE_DPMSOLVER": "False", "BLOCK_UNET_GRAD": "True",
      "MAX_IMAGES": "1", "OUT_DIR": '"../results/experiment"'}),
]


def modify_trainpy(config):
    with open(TRAIN_PY, "r") as f:
        content = f.read()

    # String variables (need quote handling)
    for var, val in config.items():
        if val.startswith('"') and val.endswith('"'):
            content = re.sub(rf'{var} = "[^"]*"', f'{var} = {val}', content)

    # Boolean variables
    for var, val in config.items():
        if val in ("True", "False"):
            content = re.sub(rf'{var} = (True|False)', f'{var} = {val}', content)

    # Numeric variables
    for var, val in config.items():
        if not val.startswith('"') and val not in ("True", "False"):
            content = re.sub(rf'{var} = [\d.]+', f'{var} = {val}', content)

    with open(TRAIN_PY, "w") as f:
        f.write(content)


def run_one(name, desc):
    logfile = os.path.join(WORK, f"run_{name}.log")
    print(f"\n{'='*60}")
    print(f">>> {name}: {desc}")
    print(f"{'='*60}")

    t0 = time.time()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    result = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True, text=True, timeout=600, cwd=WORK, env=env,
    )
    elapsed = time.time() - t0

    with open(logfile, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n" + result.stderr)

    if result.returncode != 0:
        print(f"  CRASHED (exit={result.returncode})")
        print(f"  stderr: {result.stderr[-500:]}")
        return None

    output = result.stdout

    def g(pat, default="0"):
        m = re.search(pat, output)
        return m.group(1) if m else default

    r = {
        "name": name,
        "v2": float(g(r"quality_score_v2:\s+([\d.]+)")),
        "old": float(g(r"quality_score_old:\s+([\d.]+)")),
        "sec": float(g(r"total_seconds:\s+([\d.]+)")),
        "vram": float(g(r"peak_vram_mb:\s+([\d.]+)")),
        "spi": float(g(r"s_per_image:\s+([\d.]+)")),
        "ssim": float(g(r"ssim_input_avg:\s+([\d.]+)")),
        "nat": float(g(r"naturalness_avg:\s+([\d.]+)")),
        "gain": float(g(r"sharpness_gain_avg:\s+([\d.]+)")),
        "gcalls": int(g(r"guidance_calls:\s+(\d+)", "0")),
        "rcalls": int(g(r"restorer_calls:\s+(\d+)", "0")),
    }

    print(f"  V2={r['v2']:.4f}  SSIM={r['ssim']:.4f}  NAT={r['nat']:.4f}  "
          f"GAIN={r['gain']:.2f}  Time={r['spi']:.1f}s  "
          f"GC={r['gcalls']}  RC={r['rcalls']}  VRAM={r['vram']:.0f}MB")
    return r


def main():
    results = []

    if not os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, "w") as f:
            f.write("name\tv2\tssim\tnat\tgain\tspi_s\tvram_mb\tgcalls\trcalls\tdescription\n")

    for name, desc, config in EXPERIMENTS:
        modify_trainpy(config)
        try:
            compiled = subprocess.run(
                [sys.executable, "-c", "import py_compile; py_compile.compile('/mnt/workspace/autoPGDiff/autoresearch/train.py', doraise=True)"],
                capture_output=True, text=True,
            )
            if compiled.returncode != 0:
                print(f"  SYNTAX ERROR in {name}: {compiled.stderr[-200:]}")
                results.append({"name": name, "v2": 0, "ssim": 0, "nat": 0, "gain": 0,
                               "spi": 0, "vram": 0, "gcalls": 0, "rcalls": 0})
                with open(RESULTS_TSV, "a") as f:
                    f.write(f"{name}\t0\t0\t0\t0\t0\t0\t0\t0\tSYNTAX_ERROR\t{desc}\n")
                continue
        except Exception as e:
            print(f"  SYNTAX CHECK FAILED in {name}: {e}")
            continue

        r = run_one(name, desc)
        if r:
            results.append(r)
            with open(RESULTS_TSV, "a") as f:
                f.write(f"{r['name']}\t{r['v2']}\t{r['ssim']}\t{r['nat']}\t{r['gain']}\t"
                        f"{r['spi']}\t{r['vram']}\t{r['gcalls']}\t{r['rcalls']}\t{desc}\n")
        else:
            results.append({"name": name, "v2": 0, "ssim": 0, "nat": 0, "gain": 0,
                           "spi": 0, "vram": 0, "gcalls": 0, "rcalls": 0})
            with open(RESULTS_TSV, "a") as f:
                f.write(f"{name}\t0\t0\t0\t0\t0\t0\t0\t0\tCRASHED\t{desc}\n")

    # Summary
    baseline = next((r for r in results if r["name"] == "baseline" and r["v2"] > 0), None)
    bv2 = baseline["v2"] if baseline else 1.0

    print(f"\n\n{'='*90}")
    print("  FINAL SUMMARY" + (f"  (Baseline V2 = {bv2:.4f})" if baseline else ""))
    print(f"{'='*90}")
    print(f"{'Name':<16s} {'V2':>7s} {'SSIM':>7s} {'NAT':>7s} {'GAIN':>7s} "
          f"{'Time':>7s} {'GC':>5s} {'RC':>5s} {'VRAM':>6s} {'ΔV2%':>7s}  {'Category'}")
    print("-" * 105)

    for r in results:
        v2 = r.get("v2", 0)
        if v2 > 0:
            delta = (v2 / bv2 - 1) * 100
            cat = "*PERF*" if v2 > bv2 else "*ACCEL*" if r.get("spi", 999) < 50 else "both"
            print(f"{r['name']:<16s} {v2:7.4f} {r.get('ssim',0):7.4f} {r.get('nat',0):7.4f} "
                  f"{r.get('gain',0):7.2f} {r.get('spi',0):7.1f} {r.get('gcalls',0):5d} "
                  f"{r.get('rcalls',0):5d} {r.get('vram',0):6.0f} {delta:+6.1f}%  {cat}")
        else:
            cat = "CRASH"
            print(f"{r['name']:<16s} {'CRASH':>7s} {'':>7s} {'':>7s} {'':>7s} "
                  f"{'':>7s} {'':>5s} {'':>5s} {'':>6s} {'':>7s}  {cat}")

    # Key conclusions
    print(f"\n{'='*90}")
    print("  KEY FINDINGS")
    print(f"{'='*90}")
    valid = [r for r in results if r.get("v2", 0) > 0]
    valid.sort(key=lambda x: x["v2"], reverse=True)

    print("\n  Top 3 by quality (V2):")
    for r in valid[:3]:
        print(f"    {r['name']:<16s} V2={r['v2']:.4f}  Time={r['spi']:.1f}s")

    valid.sort(key=lambda x: x["spi"])
    print("\n  Top 3 by speed (s/image):")
    for r in valid[:3]:
        print(f"    {r['name']:<16s} V2={r['v2']:.4f}  Time={r['spi']:.1f}s")


if __name__ == "__main__":
    main()
