#!/bin/bash
#SBATCH --job-name=llm-bench-seq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out

set -e
cd $SLURM_SUBMIT_DIR

CONFIGS=(small medium large xl 2.7B)

for CONFIG in "${CONFIGS[@]}"; do
  echo "Running config: $CONFIG"
  uv run cs336_systems/benchmark_llm.py \
    --warmup_steps 5 \
    --benchmark_steps 10 \
    --config "$CONFIG" \
    --context_length 256 \
    --device cuda
done
