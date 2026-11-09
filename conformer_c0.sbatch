#!/bin/bash
#SBATCH --job-name=conformer_c0
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g,gpu-h100-80g-short,gpu-h200-141g-short
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=results/encoder/encoder_probe_conformer_c0/slurm-%j.log

cd /scratch/work/thieun1/AllAmpsOnMe
module load mamba
source activate $WRKDIR/envs/openamp
export OPENAMP_DATA_DIR=data
OUT=results/encoder/encoder_probe_conformer_c0

# If an earlier job stopped, continue from its last checkpoint.
RESUME=""
[ -f $OUT/last.pt ] && RESUME="--resume $OUT/last.pt"

python -u -m openamp.probes.probe_conformer --out $OUT $RESUME
