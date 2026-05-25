#!/bin/bash
set -e

# Run all three comparison experiments and generate visualization
cd /mnt/workspace/autoPGDiff/autoresearch

# Backup original train.py
cp train.py train.py.bak

run_exp() {
    local tag=$1
    local out_dir=$2
    local block_grad=$3
    local use_dpm=$4
    local dpm_steps=${5:-50}

    echo "=========================================="
    echo "  Running: $tag"
    echo "  BLOCK_UNET_GRAD=$block_grad  USE_DPMSOLVER=$use_dpm"
    echo "=========================================="

    # Create output directory
    mkdir -p "$out_dir"

    # Update flags
    sed -i "s/^BLOCK_UNET_GRAD = .*/BLOCK_UNET_GRAD = $block_grad/" train.py
    sed -i "s/^USE_DPMSOLVER = .*/USE_DPMSOLVER = $use_dpm/" train.py
    sed -i "s|^OUT_DIR = \"../results/.*\"|OUT_DIR = \"$out_dir\"|" train.py

    # Run
    python train.py 2>&1 | tee "run_${tag}.log"

    echo "--- Done: $tag ---"
}

# 1) Baseline (no optimizations)
run_exp "baseline"     "../results/exp_baseline"     "False" "False"

# 2) Only gradient blocking
run_exp "grad_block"   "../results/exp_grad_block"   "True"  "False"

# 3) Only DPM-Solver
run_exp "dpmsolver"    "../results/exp_dpmsolver"    "False" "True"

# Restore original
cp train.py.bak train.py
rm train.py.bak

echo "=========================================="
echo "  All experiments complete! Generating comparison..."
echo "=========================================="

# Generate comparison visualization
python visualize_comparison.py

echo "Done! Comparison saved to ../results/comparison/"
