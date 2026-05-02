#!/bin/bash

# LoRA training script for 3D-IDE with reduced memory usage.
# Enables LoRA fine-tuning to significantly reduce GPU memory requirements.

# Set up the data folder
IMAGE_FOLDER="data"
VIDEO_FOLDER="data"
DATA_YAML="scripts/3d/train/multi.yaml"

############### Prepare Envs #################
alias python=python3

############### Show Envs ####################
nvidia-smi

################ Training Configuration ################

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"

# Stage 2
PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="llavanext-qwen-3d-ide-lora"
PREV_STAGE_CHECKPOINT="data/models/LLaVA-Video-7B-Qwen2"

echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"
echo "LoRA Training Enabled!"

# For single GPU training
NUM_GPUS=1
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16  # Accumulate gradients to simulate larger batch

# LoRA specific parameters
LORA_R=128  # LoRA rank (higher = more parameters but better quality)
LORA_ALPHA=256  # LoRA scaling factor (typically 2x rank)
LORA_DROPOUT=0.1  # Dropout for LoRA layers

export CUDA_VISIBLE_DEVICES=0

# Run with LoRA enabled
torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" --master_port=29500 \
    llava/train/train_3d.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --embodiedscan_folder data/embodiedscan/ \
    --lora_enable True \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_bias "none" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-5 \
    --mm_vision_adapter_lr=2e-5 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir ./ckpt/$MID_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --torch_compile False \
    --dataloader_drop_last True \
    --mm_newline_position grid \
    --add_spatial_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2 \
    --world_position_embedding_type avg-discrete-sin3d \
    --object_feature_type patch14-pe \
    --ground_head_type infonce \
    --group_by_task_length True \
    --frame_sampling_strategy uniform \
    --frames_upbound 32

echo "LoRA training completed!"
echo "LoRA weights saved to: ./ckpt/$MID_RUN_NAME"