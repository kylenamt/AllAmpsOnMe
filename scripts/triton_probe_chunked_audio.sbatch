#!/bin/bash
#SBATCH --job-name=chunked_audio
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
# probe_chunked_audio, phase 2: the chunked encoder from b128, trained by audio loss
# (pre-emphasized ESR, no STFT) through the frozen generator; 64 real + 64 made-up
# per step, shape 0.5; phase 1's learning-rate curve over 1,000 steps.
# Submit from the repo root on Triton:
#
#   sbatch scripts/triton_probe_chunked_audio.sbatch
#
# Submitting it again continues from $OUT/last.pt. Extra arguments go to the probe
# (the last of a repeated flag wins), e.g. a 2-step smoke test:
#
#   OUT=results/encoder/smoke_audio sbatch --time=01:00:00 \
#       scripts/triton_probe_chunked_audio.sbatch --steps 2 --eval-every 1
#
# torch 2.9.1 (cu128) has no Volta kernels, hence A100 / H100 only.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
PY=${PY:-/scratch/work/$USER/envs/openamp/bin/python}
OUT=${OUT:-results/encoder/encoder_probe_chunked_audio_a1}
INIT=${INIT:-results/encoder/encoder_probe_chunked_b128/probe_chunked.pt}
mkdir -p "$OUT"
RESUME=()
if [ -f "$OUT/last.pt" ]; then
    RESUME=(--resume "$OUT/last.pt")
fi
echo "[sbatch] job ${SLURM_JOB_ID:-local} on $(hostname): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)" >> "$OUT/train_stdout.log"
OPENAMP_DATA_DIR=data PYTHONUNBUFFERED=1 "$PY" -m openamp.probes.probe_chunked_audio \
    --init "$INIT" --out "$OUT" \
    --steps 1000 --batch 64 --synth-per-batch 64 --synth-shape 0.5 --stft-weight 0 \
    "${RESUME[@]}" "$@" >> "$OUT/train_stdout.log" 2>&1
