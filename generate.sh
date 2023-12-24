#!/bin/bash

set -e

MODELS=(
  "tiiuae/falcon-7b-instruct"
  "mistralai/Mistral-7B-Instruct-v0.1"
  "meta-llama/Llama-2-7b-chat-hf"
  "meta-llama/Llama-2-13b-chat-hf"
  # "meta-llama/Llama-2-70b-chat-hf"
  "facebook/opt-iml-1.3b"
  # "facebook/opt-iml-30b"
)

DATASETS=(
  "polytope"
  "factcc"
  "summeval"
)

METHODS=(
  "fullmodel"
  "magnitude"
  "wanda"
  "sparsegpt"
)

PROMPT_IDS=(
  "A"
  "B"
  "C"
)

for MODEL in "${MODELS[@]}"; do
  for DATA in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
      for PROMPT_ID in "${PROMPT_IDS[@]}"; do
        JOB_ID=$(sbatch --parsable --export=MODEL=$MODEL,DATA=$DATA,METHOD=$METHOD,PROMPT_ID=$PROMPT_ID generate_single.sh)
        echo $JOB_ID $MODEL $DATA $METHOD $PROMPT_ID
      done
    done
  done
done
