"""
Command-line analysis of PGDiff autoresearch results.
Reads results.tsv and prints summary + per-experiment details.

Usage: python analysis.py [--tsv results.tsv]
"""

import argparse
import os
import json


def parse_loss_breakdown(raw):
    """Parse 'key1:val1,key2:val2' into dict."""
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
        print(f"ERROR: {tsv_path} not found. Run some experiments first.")
        print(f"Expected format (tab-separated):")
        print(f"  commit  avg_loss  loss_per_image  task  status  description  key_params  loss_breakdown")
        return

    rows = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            # Pad to 8 columns if needed
            while len(parts) < 8:
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
        s = r[4].strip().upper() if len(r) > 4 else "UNKNOWN"
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
    baseline_loss = float(baseline[1]) if len(baseline) > 1 and baseline[1] else 0

    # Show kept experiments
    kept = [r for r in rows if len(r) > 4 and r[4].strip().upper() == "KEEP"]
    if kept:
        print(f"KEPT experiments ({len(kept)} total):")
        print(f"{'#':>3}  {'loss':>10}  {'per_img':>8}  {'task':<22}  {'params':<30}  description")
        print("-" * 110)
        for i, r in enumerate(kept):
            loss = float(r[1]) if len(r) > 1 else 0
            per_img = float(r[2]) if len(r) > 2 else 0
            task = r[3] if len(r) > 3 else "?"
            params = r[6][:28] if len(r) > 6 else ""
            desc = r[5][:50] if len(r) > 5 else "?"
            print(f"{i:3d}  {loss:10.1f}  {per_img:8.1f}  {task:<22s}  {params:<30s}  {desc}")
        print()

    # Summary
    best = min(kept, key=lambda r: float(r[1])) if kept else None
    if baseline_loss > 0 and best:
        best_loss = float(best[1])
        improvement = baseline_loss - best_loss
        print("=" * 60)
        print(f"  Baseline loss:  {baseline_loss:.1f}")
        print(f"  Best loss:      {best_loss:.1f}")
        print(f"  Improvement:    {improvement:.1f} ({improvement / baseline_loss * 100:.2f}%)")
        print(f"  Best config:    {best[5] if len(best) > 5 else '?'}")
        print("=" * 60)
        print()

    # Per-task breakdown
    tasks = {}
    for r in rows:
        if len(r) > 3:
            task = r[3]
            if task not in tasks:
                tasks[task] = {"total": 0, "kept": 0, "best": float("inf"), "best_losses": {}}
            tasks[task]["total"] += 1
            if len(r) > 4 and r[4].strip().upper() == "KEEP":
                tasks[task]["kept"] += 1
                loss = float(r[1]) if len(r) > 1 else 0
                if 0 < loss < tasks[task]["best"]:
                    tasks[task]["best"] = loss
                # Track loss breakdown terms
                breakdown = parse_loss_breakdown(r[7]) if len(r) > 7 else {}
                for k, v in breakdown.items():
                    if k not in tasks[task]["best_losses"]:
                        tasks[task]["best_losses"][k] = v

    if tasks:
        print("By task:")
        for task, stats in sorted(tasks.items()):
            best_str = f"{stats['best']:.1f}" if stats["best"] < float("inf") else "N/A"
            print(f"  {task:22s}: {stats['total']:3d} total, {stats['kept']:3d} kept, best={best_str}")
            if stats["best_losses"]:
                print(f"    Best loss breakdown: {', '.join(f'{k}={v:.1f}' for k, v in sorted(stats['best_losses'].items()))}")

    # Top hits
    if len(kept) > 1:
        print()
        print("Top improvements (kept experiments by delta):")
        print(f"{'Rank':>4}  {'Delta':>10}  {'Loss':>10}  {'Task':<22}  Description")
        print("-" * 90)
        prev_loss = baseline_loss
        hits = []
        for i, r in enumerate(kept):
            loss = float(r[1]) if len(r) > 1 else 0
            if i == 0:
                prev_loss = loss
                continue
            delta = prev_loss - loss
            prev_loss = loss
            desc = r[5] if len(r) > 5 else "?"
            task = r[3] if len(r) > 3 else "?"
            hits.append((delta, loss, task, desc))
        hits.sort(key=lambda x: x[0], reverse=True)
        for rank, (delta, loss, task, desc) in enumerate(hits, 1):
            print(f"{rank:4d}  {delta:+10.1f}  {loss:10.1f}  {task:<22s}  {desc}")

    # Show runs/ directory content
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
