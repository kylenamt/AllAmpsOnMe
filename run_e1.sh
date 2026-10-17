#!/bin/bash
#SBATCH --job-name=synth_e1
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g,gpu-h200-141g-short
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=results/encoder/encoder_probe_synth_e1/slurm-%j.log

cd $WRKDIR/AllAmpsOnMe
module load mamba
source activate $WRKDIR/envs/openamp
export OPENAMP_DATA_DIR=data

OUT=results/encoder/encoder_probe_synth_e1
RESUME=""
[ -f $OUT/last.pt ] && RESUME="--resume $OUT/last.pt"

python -u -m openamp.probes.probe_synth --synth-per-batch 16 --out $OUT $RESUME
