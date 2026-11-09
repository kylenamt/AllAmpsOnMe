#!/bin/bash
#SBATCH --job-name=unfold_k10_so
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
# Ten learned enrollment steps, full (second-order) gradients, c1's guess as the start,
# on v2_n10. Submit from the repo root on Triton:
#
#   sbatch scripts/triton_unfold_k10_second_order.sbatch
#
# Submitting it again continues from the run's last.pt. Extra arguments go to
# `openamp unfold`, e.g. a one-epoch smoke test:
#
#   sbatch --time=12:00:00 scripts/triton_unfold_k10_second_order.sbatch --epochs 1
#
# torch 2.9.1 (cu128) has no Volta kernels, hence A100 / H100 only.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
BIN=${BIN:-/scratch/work/$USER/envs/openamp/bin/openamp}
CONFIG=configs/unfold/k10_second_order_v2n10_c1.yaml
OUT=results/unfold/$(basename "$CONFIG" .yaml)
mkdir -p "$OUT"
echo "[sbatch] job ${SLURM_JOB_ID:-local} on $(hostname): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)" >> "$OUT/train_stdout.log"
OPENAMP_DATA_DIR=data PYTHONUNBUFFERED=1 "$BIN" unfold --config "$CONFIG" --resume \
    "$@" >> "$OUT/train_stdout.log" 2>&1
