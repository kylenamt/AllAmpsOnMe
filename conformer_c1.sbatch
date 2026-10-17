#!/bin/bash
#SBATCH --job-name=conformer_c1
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g,gpu-h100-80g-short,gpu-h200-141g-short
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=results/encoder/encoder_probe_conformer_c1/slurm-%j.log

cd /scratch/work/thieun1/AllAmpsOnMe
module load mamba
source activate $WRKDIR/envs/openamp
export OPENAMP_DATA_DIR=data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=results/encoder/encoder_probe_conformer_c1

# c1 vs c0: E1's 16 real + 16 made-up split, and a 50% crop of the reference
# (84.4 of its 168.9 s after the skip) as the encoder input, instead of 2 s.
# --raw-chunk 4: the raw branch on 32 x 84 s does not fit in 80 GB otherwise.
# From step 3420 (job 20992271 ran 1e-3 constant, wd 0.1 until then): cosine decay of
# the same 1e-3 over all 17,100 steps, no warmup (none was run), wd 0.01.
RESUME=""
[ -f $OUT/last.pt ] && RESUME="--resume $OUT/last.pt"

python -u -m openamp.probes.probe_conformer --synth-per-batch 16 --ref-seconds 84.4 \
    --raw-chunk 4 --lr-schedule cosine --warmup 0 --min-lr 1e-5 --weight-decay 0.01 \
    --out $OUT $RESUME
