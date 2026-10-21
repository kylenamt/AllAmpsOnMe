#!/usr/bin/env bash
# Copy what probe_chunked reads to Triton. All of it is git-ignored, so a clone
# or pull there does not bring it (~2.2 GB, mostly data/captures).
#
#   scripts/triton_stage_chunked.sh            # run on the local box, from anywhere
#   TRITON=user@triton.aalto.fi DEST=/scratch/work/user/AllAmpsOnMe scripts/triton_stage_chunked.sh
set -euo pipefail
TRITON=${TRITON:-triton.aalto.fi}
DEST=${DEST:-/scratch/work/$USER/AllAmpsOnMe}
cd "$(dirname "$0")/.."
rsync -aR --info=progress2 \
    ./data/captures \
    ./data/manifests/renders.parquet \
    ./data_sweep_a1/clean/train/sweep_train.flac \
    ./results/emulate/delta_a2_tanh_r32_512_v2_n10 \
    "$TRITON:$DEST/"
