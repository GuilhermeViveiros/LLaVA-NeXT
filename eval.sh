#!/bin/bash

# Enable conda in this shell
eval "$(conda shell.bash hook)"

# Activate environment
conda activate llava-next-env

cd /mnt/home/gviveiros/LLaVA-NeXT

echo "Current directory: $(pwd)"

# Define tasks array
#tasks=("mmmu_pro_cot" "gqa" "pope" "illusionvqa" "blink")
#tasks=("alm_bench-all")


# alm_bench-all
# textvqa
# m3exam
# cc-ocr-multi-lan
# commute-all-contrastive
# multi30k-all
# ocrbench

# get task from terminal
#tasks=("textvqa" "alm_bench-all" "m3exam" "cc-ocr-multi-lan" "commute-all-contrastive" "multi30k-all" "ocrbench")
#tasks=( "commute-all-contrastive", "ocrbench", "multi30k-all")
# tasks=( "multi30k-all" "ocrbench")
#tasks=( "aya-vision-bench-gen" "m3exam")
tasks=( "m3exam")


# /mnt/scratch-artemis/gviveiros/hf_models/TowerVision-CPT-2B-Full-0.97/
# utter-project/TowerVision-Plus-2B
# /mnt/scratch-artemis/gviveiros/hf_models/TowerVision-CPT-2B-EN/

export PYTHONPATH='/home/gviveiros/LLaVA-NeXT/lmms-eval:$PYTHONPATH'

#model=qwen2_5_vl
#model=qwen2_5_vl_interleave
#pretrained=Qwen/Qwen2.5-VL-7B-Instruct
#pretrained=JefferyZhan/Qwen2.5-VL-7B-Instruct-Vision-R1
#pretrained=Osilly/Vision-R1-7B

pretrained=utter-project/TowerVision-Plus-2B
#pretrained=/mnt/scratch-artemis/gviveiros/TowerVision/llava-next-native/towerp_2b_instruct/

#pretrained=/mnt/scratch-artemis/gviveiros/TowerVision/llava-next-native/towerp_2b_base
#pretrained=/mnt/scratch-artemis/gviveiros/TowerVision/llava-next-native/towerp_2b_base_full_siglip512


echo "Evaluating $pretrained on multiple tasks -> $tasks"

# Loop through each task
for task in "${tasks[@]}"; do

    echo "=========================================="
    echo "Running evaluation for task: $task"
    echo "=========================================="
    
    echo "Running with the following arguments:
    pretrained: $pretrained
    task: $task
    batch_size: $batch_size
    verbosity: DEBUG
    "
    
    # Run evaluation for current task
    python -m lmms_eval \
      --model llava_hf \
      --model_args pretrained=$pretrained, \
      --tasks $task \
      --verbosity=DEBUG \
      --batch_size 1 \
      --show_config \
      --log_samples \
      --output_path ./logs/$task
    
    echo "Completed evaluation for task: $task"
    echo "=========================================="
    echo ""
done
  