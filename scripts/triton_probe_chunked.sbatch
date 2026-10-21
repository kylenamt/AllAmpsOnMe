#!/bin/bash
#SBATCH --job-name=chunked_p3
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g,gpu-h100-80g-short,gpu-h200-141g-short
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
# probe_chunked, phase 1 with made-up amps: 32 real + 32 made-up per step, shape 0.5.
# Submit from the repo root on Triton:
#
#   sbatch scripts/triton_probe_chunked.sbatch
#
# Submitting it again continues from $OUT/last.pt. Extra arguments go to the probe
# (the last of a repeated flag wins), e.g. a 2-step smoke test:
#
#   OUT=results/encoder/smoke sbatch --time=00:30:00 scripts/triton_probe_chunked.sbatch \
#       --steps 2 --eval-every 1
#
# torch 2.9.1 (cu128) has no Volta kernels, hence A100 / H100 / H200 only.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
module load mamba
source activate $WRKDIR/envs/openamp
set -u
export OPENAMP_DATA_DIR=data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python}
OUT=${OUT:-results/encoder/encoder_probe_chunked_p3}
mkdir -p "$OUT"
RESUME=()
if [ -f "$OUT/last.pt" ]; then
    RESUME=(--resume "$OUT/last.pt")
fi
echo "[sbatch] job ${SLURM_JOB_ID:-local} on $(hostname): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)" >> "$OUT/train_stdout.log"
PYTHONUNBUFFERED=1 "$PY" -m openamp.probes.probe_chunked --out "$OUT" \
    --steps 1000 --batch 32 --synth-per-batch 32 --synth-shape 0.5 \
    "${RESUME[@]}" "$@" >> "$OUT/train_stdout.log" 2>&1
