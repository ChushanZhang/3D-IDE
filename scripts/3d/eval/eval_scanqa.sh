#!/bin/bash

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

# Add CUDA library paths
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

CKPT="$1"
MODEL_NAME=$(basename "$CKPT")
ANWSER_FILE="$MODEL_NAME/results/scanqa/$MODEL_NAME.jsonl"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 llava/eval/model_scanqa.py \
    --model-path $CKPT \
    --video-folder ./data \
    --embodiedscan-folder data/embodiedscan \
    --n_gpu 8 \
    --frame_sampling_strategy $2 \
    --max_frame_num $3 \
    --question-file data/processed/scanqa_val_llava_style.json \
    --conv-mode qwen_1_5 \
    --answer-file $ANWSER_FILE \
    --overwrite_cfg true

python llava/eval/eval_scanqa.py --input-file $ANWSER_FILE
