"""
Command-line analysis of PGDiff autoresearch results.
Reads results.tsv and prints summary + per-experiment details.

Usage: python analysis.py [--tsv results.tsv]
"""

import argparse
import os


def parse_loss_breakdown(raw):
    if not raw or raw == "N/A":
        return {}
    result = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            k, v = pair.split(":", 1)
            try:
                result[k.strip()] = float(v.strip())
            except ValueError:
                result[k.strip()] = v.strip()
    return result


def analyze(tsv_path="results.tsv"):
    if not os.path.exists(tsv_path):
        print(f"ERROR: {tsv_path} not found.")
        print("Expected 10 columns (tab-separated):")
        print("  commit  quality_score  avg_loss  avg_sharpness  loss_per_image  task  status  description  key_params  loss_breakdown")
        return

    rows = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            while len(parts) < 10:
                parts.append("")
            rows.append(parts)

    if not rows:
        print("No experiment results yet.")
        return

    print(f"Total experiments: {len(rows)}")
    print()

    # Count outcomes
    statuses = {}
    for r in rows:
        s = r[6].strip().upper() if len(r) > 6 else "UNKNOWN"
        statuses[s] = statuses.get(s, 0) + 1

    print("Experiment outcomes:")
    for s in ("KEEP", "DISCARD", "CRASH"):
        c = statuses.get(s, 0)
        if c > 0 or s in ("KEEP", "DISCARD"):
            print(f"  {s:10s}: {c}")
    n_keep = statuses.get("KEEP", 0)
    n_discard = statuses.get("DISCARD", 0)
    if n_keep + n_discard > 0:
        print(f"  Keep rate: {n_keep}/{n_keep + n_discard} = {n_keep / (n_keep + n_discard) * 100:.1f}%")
    print()

    # Baseline
    baseline = rows[0]
    safe_float = lambda r, i: float(r[i]) if len(r) > i and r[i] and r[i] != "N/A" else 0.0
    baseline_qs = safe_float(baseline, 1)

    # Kept experiments
    kept = [r for r in rows if len(r) > 6 and r[6].strip().upper() == "KEEP"]
    if kept:
        print(f"KEPT experiments ({len(kept)} total):")
        print(f"{'#':>3}  {'quality':>8}  {'loss':>10}  {'sharp':>8}  {'task':<22}  {'params':<28}  description")
        print("-" * 115)
        for i, r in enumerate(kept):
            qs = safe_float(r, 1)
            loss = safe_float(r, 2)
            sharp = safe_float(r, 3)
            task = r[5] if len(r) > 5 else "?"
            params = r[8][:26] if len(r) > 8 else ""
            desc = r[7][:45] if len(r) > 7 else "?"
            print(f"{i:3d}  {qs:8.2f}  {loss:10.1f}  {sharp:8.1f}  {task:<22s}  {params:<28s}  {desc}")
        print()

    # Summary
    best = min(kept, key=lambda r: safe_float(r, 1)) if kept else None
    if best:
        best_qs = safe_float(best, 1)
        improvement = baseline_qs - best_qs
        print("=" * 60)
        print(f"  Baseline quality_score:  {baseline_qs:.2f}")
        print(f"  Best quality_score:      {best_qs:.2f}")
        if baseline_qs > 0:
            print(f"  Improvement:             {improvement:.2f} ({improvement / baseline_qs * 100:.2f}%)")
        print(f"  Best config:             {best[7] if len(best) > 7 else '?'}")
        print("=" * 60)
        print()

    # Per-task breakdown
    tasks = {}
    for r in rows:
        if len(r) > 5:
            task = r[5]
            if task not in tasks:
                tasks[task] = {"total": 0, "kept": 0, "best_qs": float("inf")}
            tasks[task]["total"] += 1
            if len(r) > 6 and r[6].strip().upper() == "KEEP":
                tasks[task]["kept"] += 1
                qs = safe_float(r, 1)
                if qs > 0 and qs < tasks[task]["best_qs"]:
                    tasks[task]["best_qs"] = qs

    if tasks:
        print("By task:")
        for task, stats in sorted(tasks.items()):
            best_str = f"{stats['best_qs']:.2f}" if stats["best_qs"] < float("inf") else "N/A"
            print(f"  {task:22s}: {stats['total']:3d} total, {stats['kept']:3d} kept, best_qs={best_str}")

    # Top hits
    if len(kept) > 1:
        print()
        print("Top improvements (kept experiments by delta):")
        print(f"{'Rank':>4}  {'Delta':>10}  {'QS':>8}  {'Loss':>10}  {'Sharp':>8}  Description")
        print("-" * 85)
        prev_qs = baseline_qs
        hits = []
        for i, r in enumerate(kept):
            qs = safe_float(r, 1)
            if i == 0:
                prev_qs = qs
                continue
            delta = prev_qs - qs
            prev_qs = qs
            desc = r[7] if len(r) > 7 else "?"
            loss = safe_float(r, 2)
            sharp = safe_float(r, 3)
            hits.append((delta, qs, loss, sharp, desc))
        hits.sort(key=lambda x: x[0], reverse=True)
        for rank, (delta, qs, loss, sharp, desc) in enumerate(hits, 1):
            print(f"{rank:4d}  {delta:+10.4f}  {qs:8.2f}  {loss:10.1f}  {sharp:8.1f}  {desc}")

    # runs/ directory
    runs_dir = "runs"
    if os.path.isdir(runs_dir):
        run_dirs = sorted(
            [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))],
            key=lambda x: int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else 0
        )
        if run_dirs:
            print(f"\nDetailed run logs: {len(run_dirs)} experiments in {runs_dir}/")
            print(f"  Latest: {run_dirs[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze PGDiff autoresearch results")
    parser.add_argument("--tsv", default="results.tsv", help="Path to results.tsv")
    args = parser.parse_args()
    analyze(args.tsv)
