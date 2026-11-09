#!/usr/bin/env bash
# Copy what the second-order unfold run reads to Triton. All of it is git-ignored, so
# a clone or pull there does not bring it (~204 GB, nearly all data/renders).
#
#   scripts/triton_stage_unfold.sh            # run on the local box, from anywhere
#   TRITON=user@triton.aalto.fi DEST=/scratch/work/user/AllAmpsOnMe scripts/triton_stage_unfold.sh
set -euo pipefail
TRITON=${TRITON:-triton.aalto.fi}
DEST=${DEST:-/scratch/work/$USER/AllAmpsOnMe}
cd "$(dirname "$0")/.."
rsync -aR --info=progress2 \
    ./data/captures \
    ./data/clean \
    ./data/manifests \
    ./data/renders \
    ./data_sweep_a1/clean/train/sweep_train.flac \
    ./results/emulate/delta_a2_tanh_r32_512_v2_n10 \
    ./results/encoder/encoder_probe_conformer_c1/probe_conformer.pt \
    "$TRITON:$DEST/"
