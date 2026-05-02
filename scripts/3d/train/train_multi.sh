#!/bin/bash
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
fi
# Set up the data folder
IMAGE_FOLDER="data"
VIDEO_FOLDER="data"
DATA_YAML="scripts/3d/train/multi.yaml" # e.g exp.yaml

############### Prepare Envs #################
# python3 -m pip install flash-attn --no-build-isolation
alias python=python3
############### Show Envs ####################

nvidia-smi

############### Training Configuration ##############

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# Stage 2
PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="llavanext-qwen-3d-ide"
PREV_STAGE_CHECKPOINT="data/models/LLaVA-Video-7B-Qwen2"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

# Create ckpt directory if it doesn't exist
# Use command line argument if provided, otherwise use default
OUTPUT_BASE="${1:-${OUTPUT_BASE:-./data/exp/ckpt}}"
echo "OUTPUT_BASE: ${OUTPUT_BASE}"
mkdir -p ${OUTPUT_BASE}

# --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
# --mm_vision_tower_lr=2e-6 \

DPT_CHECKPOINT_PATH="${DPT_CHECKPOINT_PATH:-${VGGT_CHECKPOINT_PATH:-VGGT_checkpoints/model.pt}}"

NUM_GPUS=8
BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=$((BATCH_SIZE/NUM_GPUS))
CROSS_VIEW_NEIGHBOR_RANGE=2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" --master_port 43000 \
    llava/train/train_3d.py \
    --deepspeed scripts/zero2_fused_adamw.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --embodiedscan_folder data/embodiedscan/ \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --mm_vision_adapter_lr=2e-6 \
    --dpt_lr=2e-6 \
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
    --output_dir ${OUTPUT_BASE}/$MID_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
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
    --cross_view_neighbor_range $CROSS_VIEW_NEIGHBOR_RANGE \
    --frames_upbound 32 \
    --use_feature_3d_distillation True \
    --skip_3d_pe_fusion True \
    --use_depth_auxiliary_task True \
    --dpt_checkpoint_path "${DPT_CHECKPOINT_PATH}" \
    --weak_geometric_validator True \
    >> "${OUTPUT_BASE}/${MID_RUN_NAME}.log" 2>&1
exit 0;
