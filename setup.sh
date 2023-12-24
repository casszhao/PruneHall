#!/bin/bash




set -e

CONDA_PACKAGES=(
  "python=3.11"
)
PIP_PACKAGES=(
  "accelerate"
  "bert-score"
  "datasets"
  "evaluate"
  "nltk"
  "optimum"
  "rouge-score"
  "scikit-learn"
  "sentencepiece"
  "summac"
  "torch"
  "transformers"
)
ENV_NAME="hl"

module load Anaconda3/2022.05
conda init bash
# Deactivate all conda environments.
for _ in $(seq "$CONDA_SHLVL"); do
    conda deactivate
done

conda create --name $ENV_NAME "${CONDA_PACKAGES[@]}" -y
conda activate $ENV_NAME

pip install "${PIP_PACKAGES[@]}"