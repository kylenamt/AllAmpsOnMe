#!/bin/bash
#SBATCH --job-name=feedback_f1
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g,gpu-h100-80g-short,gpu-h200-141g-short
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out
# probe_feedback, run f1: c1's guess corrected by 3 forward passes that hear their own
# error. c1's recipe (16 real + 16 made-up amps, 84.4 s reference crop, lr 1e-3 cosine
# to 1e-5, wd 0.01) with a 50-step warmup, over 6,000 steps -- the probe's defaults.
# Submit from the repo root on Triton:
#
#   sbatch scripts/triton_probe_feedback.sbatch
#
# Submitting it again continues from $OUT/last.pt; the run is longer than one job's
# time limit, so expect to. Extra arguments go to the probe (the last of a repeated
# flag wins), e.g. a 2-step smoke test:
#
#   OUT=results/encoder/smoke_feedback sbatch --time=01:00:00 \
#       scripts/triton_probe_feedback.sbatch --steps 2 --eval-every 1
#
# Reads what c1 read, plus c1 itself ($INIT). The corpus renders are not on Triton, so
# the report amps are not scored here: copy $OUT back and run, on the local box,
#
#   OPENAMP_DATA_DIR=data python -m openamp.probes.probe_feedback --score $OUT
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
OUT=${OUT:-results/encoder/encoder_probe_feedback_f1}
INIT=${INIT:-results/encoder/encoder_probe_conformer_c1/probe_conformer.pt}
mkdir -p "$OUT"
RESUME=()
if [ -f "$OUT/last.pt" ]; then
    RESUME=(--resume "$OUT/last.pt")
fi
echo "[sbatch] job ${SLURM_JOB_ID:-local} on $(hostname): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)" >> "$OUT/train_stdout.log"
PYTHONUNBUFFERED=1 "$PY" -m openamp.probes.probe_feedback --init "$INIT" --out "$OUT" \
    "${RESUME[@]}" "$@" >> "$OUT/train_stdout.log" 2>&1
