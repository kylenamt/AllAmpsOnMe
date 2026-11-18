#!/bin/bash
#SBATCH --job-name=refine_r1
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g,gpu-h100-80g-short,gpu-h200-141g-short
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out
# probe_refine, run r1: the sweep-aware refinement network. RefineNet proposes 8
# directions per pass; the frozen generator measures each (nudge 0.02, fp32) and the
# best mix is solved. One pass per step from starts at every distance; 16 real + 16
# made-up amps, 84 s crop, lr 3e-4 (500-step warmup) cosine to 1e-5, 6,000 steps --
# the probe's defaults. Submit from the repo root on Triton:
#
#   sbatch scripts/triton_probe_refine.sbatch
#
# Submitting it again continues from $OUT/last.pt; expect to, the run is longer than
# one job. Extra arguments go to the probe, e.g. a smoke test that times one step
# and one validation:
#
#   OUT=results/encoder/smoke_refine sbatch --time=02:00:00 \
#       scripts/triton_probe_refine.sbatch --steps 4 --eval-every 4
#
# Reads what probe_feedback read (c1 = $INIT, the generator run, the sweep, the
# captures, the renders manifest). The corpus renders are not on Triton, so the
# report amps are scored back on the local box:
#
#   OPENAMP_DATA_DIR=data python -m openamp.probes.probe_refine --score $OUT
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
OUT=${OUT:-results/encoder/encoder_probe_refine_r1}
INIT=${INIT:-results/encoder/encoder_probe_conformer_c1/probe_conformer.pt}
mkdir -p "$OUT"
RESUME=()
if [ -f "$OUT/last.pt" ]; then
    RESUME=(--resume "$OUT/last.pt")
fi
echo "[sbatch] job ${SLURM_JOB_ID:-local} on $(hostname): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)" >> "$OUT/train_stdout.log"
PYTHONUNBUFFERED=1 "$PY" -m openamp.probes.probe_refine --init "$INIT" --out "$OUT" \
    "${RESUME[@]}" "$@" >> "$OUT/train_stdout.log" 2>&1
