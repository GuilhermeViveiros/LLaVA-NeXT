#!/bin/bash

# Enable conda in this shell
eval "$(conda shell.bash hook)"

# Activate environment
conda activate llava-next

# load CUDA 12.4
module load cuda/12.4

export PYTHONPATH=$(pwd):$PYTHONPATH


set -e # stop on errors
set -o pipefail # stop on pipeline errors
set -u # stop on undeclared variables

# define the pythonpath
# Configuration variables (copied from vblock_bsc.tfconf)
REPO=/mnt/home/gviveiros/LLaVA-NeXT
DATA_FOLDER="/mnt/scratch-artemis/gviveiros/ehpc209/vision-data/llava_datasets"
TEXT_MODEL="/mnt/scratch-artemis/gviveiros/ehpc209/LC-EuroLLM/EuroLLM-1.7B-Legacy-32k/"
VISION_MODEL="/mnt/scratch-artemis/gviveiros/ehpc209/hf_models/siglip2-so400m-patch14-384/"
DATA_YAML="/mnt/scratch-artemis/gviveiros/ehpc209/vision-data/visionblocks_v6p5_SFT_euro.yaml"
MODEL_DIR="/mnt/scratch-artemis/gviveiros/ehpc209/vlm_ckpts/finetune/eurollm_1p7b_32k/"
PRETRAINED_MODEL_DIR="/mnt/scratch-artemis/gviveiros/ehpc209/vlm_ckpts/pretrain/eurollm_1p7b_32k/"

# Training parameters
NUM_GPUS=1
MICRO_BATCH_SIZE=2  # Increased from 2 to 8 for the smaller model
GLOBAL_BATCH_SIZE=128
LEARNING_RATE=1e-5
VISION_TOWER_LR=2e-6
WARMUP_RATIO=0.05
MODEL_MAX_LENGTH=8192
NUM_EPOCHS=1
PROMPT_VERSION="qwen_1_5"
MAX_TILES=6

# Create output directory
mkdir -p $MODEL_DIR

# For debugging purposes
export NCCL_DEBUG=WARN
export WANDB_PROJECT="llavanext-vblocks"
export WANDB_MODE=offline

# Clean model names for directory naming
TEXT_MODEL_CLEAN=$(echo $TEXT_MODEL | sed 's/\//_/g')
VISION_MODEL_CLEAN=$(echo $VISION_MODEL | sed 's/\//_/g')
DATASET_NAME_CLEAN=$(basename $DATA_YAML | sed 's/\.yaml$//')

# Create base run name
RUN_NAME="finetune::${TEXT_MODEL_CLEAN}:${VISION_MODEL_CLEAN}:${DATASET_NAME_CLEAN}"
echo "RUN_NAME: ${RUN_NAME}"

# Extract mm_projector for finetuning
PROJECTOR_ARG=""
if [ -f "$PRETRAINED_MODEL_DIR/mm_projector.bin" ]; then
    echo "Extracting mm_projector from pretrained checkpoint..."
    PROJECTOR_ARG="--pretrain_mm_mlp_adapter=$PRETRAINED_MODEL_DIR/mm_projector.bin"
else
    echo "ERROR: mm_projector.bin not found in pretrained checkpoint"
    exit 1
fi

# Define data variables
data_path=$DATA_YAML
image_folder="${DATA_FOLDER}/images"

# Calculate gradient accumulation steps
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE / NUM_GPUS))
echo "Using 'grad_accum_steps=${GRAD_ACCUM_STEPS}'"

# Function to get tile grids
function get_tile_grids() {
    local max_num=${1:-16}

    for n in $(seq 1 $max_num); do
        for i in $(seq 1 $n); do
            for j in $(seq 1 $n); do
                if [ $((i*j)) -le $max_num ]; then
                    echo "$i $j $((i*j))"
                fi
            done
        done
    done | sort -n -k3 | awk '!seen[$1"x"$2]++' | awk '{printf("%s(%dx%d)", NR>1?",":"", $1, $2)}'
}

tile_grids=$(get_tile_grids $MAX_TILES)
echo "Using 'tile_grids=${tile_grids}'"

# Set training-mode specific arguments
TRAIN_ARGS="--mm_tunable_parts=mm_vision_tower,mm_mlp_adapter,mm_language_model"
TRAIN_ARGS="$TRAIN_ARGS --image_aspect_ratio anyres"
TRAIN_ARGS="$TRAIN_ARGS --image_grid_pinpoints \"${tile_grids}\""
TRAIN_ARGS="$TRAIN_ARGS --mm_patch_merge_type spatial_unpad"
TRAIN_ARGS="$TRAIN_ARGS --dataloader_drop_last True"
TRAIN_ARGS="$TRAIN_ARGS --mm_vision_tower_lr ${VISION_TOWER_LR}"

# Run the training command
echo "Running training with ${NUM_GPUS} GPUs"
ACCELERATE_CPU_AFFINITY=1 torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    ${REPO}/llava/train/train_mem.py \
    --deepspeed ${REPO}/scripts/zero3.json \
    --model_name_or_path ${TEXT_MODEL} \
    --vision_tower ${VISION_MODEL} \
    --version ${PROMPT_VERSION} \
    --system_from_data True \
    --data_path ${data_path} \
    --image_folder ${image_folder} \
    --output_dir ${MODEL_DIR} \
    ${PROJECTOR_ARG} \
    ${TRAIN_ARGS} \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${MICRO_BATCH_SIZE} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --save_strategy "steps" \
    --save_steps 0.1 \
    --group_by_modality_length True \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0. \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --bf16 True \
    --tf32 True \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --disable_tqdm True \
    --run_name ${RUN_NAME} \
    --use_liger True 