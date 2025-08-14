#!/bin/bash

# Enable conda in this shell
eval "$(conda shell.bash hook)"

# Activate environment
conda activate llava-next-env

cd /mnt/home/gviveiros/LLaVA-NeXT

echo "Current directory: $(pwd)"

# Define tasks array
#tasks=("mmmu_pro_cot" "gqa" "pope" "illusionvqa" "blink")
tasks=("alm_bench-all")

model=llava_next
#model=qwen2_5_vl
#model=qwen2_5_vl_interleave
#pretrained=Qwen/Qwen2.5-VL-7B-Instruct
#pretrained=JefferyZhan/Qwen2.5-VL-7B-Instruct-Vision-R1
#pretrained=Osilly/Vision-R1-7B
pretrained=utter-project/TowerVision-Plus-2B

echo "Evaluating $pretrained on multiple tasks -> $tasks"

# Loop through each task
for task in "${tasks[@]}"; do

    # Set batch size based on task
    if [ "$task" = "blink" ]; then
        batch_size=6
    else
        batch_size=26
    fi

    echo "=========================================="
    echo "Running evaluation for task: $task"
    echo "=========================================="
    
    echo "Running with the following arguments:
    model: $model
    pretrained: $pretrained
    task: $task
    batch_size: $batch_size
    verbosity: DEBUG
    "
    
    # Run evaluation for current task
    python -m lmms_eval \
      --model $model \
      --model_args pretrained=$pretrained \
      --tasks $task \
      --verbosity=DEBUG \
      --batch_size $batch_size \
      --show_config \
      --log_samples \
      --output_path ./logs/$task
    
    echo "Completed evaluation for task: $task"
    echo "=========================================="
    echo ""
done

echo "All tasks completed!"
  