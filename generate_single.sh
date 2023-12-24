#!/bin/bash
#SBATCH --mem=64GB
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00

set -e

echo $MODEL $DATA $METHOD $PROMPT_ID

nvidia-smi

module load Anaconda3/2022.05
conda activate hl

python generate_and_save_summary.py \
  --model $MODEL \
  --data $DATA \
  --prune_method $METHOD \
  --prompt_id $PROMPT_ID