#!/bin/bash

# Enable conda in this shell
eval "$(conda shell.bash hook)"

# Activate environment
conda activate llava-next-env

cd /home/gviveiros/LLaVA-NeXT

echo "Current directory: $(pwd)"


# cultural ground and Phi run here


export PYTHONPATH='/home/gviveiros/LLaVA-NeXT/lmms-eval:$PYTHONPATH'

declare -A model_args_map

# Molmo : TODO -> fix this

model_args_map["mistralai/Pixtral-12B-2409"]="device_map=cuda:0,device=cuda:0,tokenizer_mode=mistral"
model_args_map["neulab/CulturalPangea-7B"]="device_map=cuda:0,device=cuda:0"
model_args_map["microsoft/Phi-4-multimodal-instruct"]="device_map=cuda:0,device=cuda:0"
model_args_map["allenai/Molmo-7B-D-0924"]="device_map=cuda:0,device=cuda:0"
model_args_map["google/gemma-3-4b-it"]="device=cuda:0"
model_args_map["google/gemma-3-12b-it"]="device=cuda:0"
model_args_map["CohereForAI/aya-vision-8b"]="device_map=cuda:0,device=cuda:0"
model_args_map["Qwen/Qwen2.5-VL-7B-Instruct"]="device_map=cuda:0,device=cuda:0"
model_args_map["Qwen/Qwen2.5-VL-3B-Instruct"]="device_map=cuda:0,device=cuda:0"
model_args_map["utter-project/TowerVision-Plus-9B"]="device_map=cuda:0,device=cuda:0"
model_args_map["utter-project/TowerVision-Plus-2B"]="device_map=cuda:0,device=cuda:0"
model_args_map["llava-hf/llama3-llava-next-8b-hf"]="device_map=cuda:0,device=cuda:0"
model_args_map["Unbabel/Tower-Plus-9B"]="device_map=cuda:0,device=cuda:0"
# model evals
model_args_map["/mnt/scratch-artemis/gviveiros/TowerVision/towerp_2b_instruct"]="device_map=cuda:0,device=cuda:0"
model_args_map["/mnt/scratch-artemis/gviveiros/TowerVision/towerp_9b_instruct"]="device_map=cuda:0,device=cuda:0"
model_args_map["Unbabel/Tower-Plus-2B"]="device_map=cuda:0,device=cuda:0"
model_args_map["Unbabel/Tower-Plus-9B"]="device_map=cuda:0,device=cuda:0"
model_args_map["/mnt/scratch-artemis/gviveiros/TowerVision/test_inst_2b/TowerVision-Plus-2B"]="device_map=cuda:0,device=cuda:0"
model_args_map["/mnt/scratch-artemis/gviveiros/TowerVision/test_inst_9b/TowerVision-Plus-9B"]="device_map=cuda:0,device=cuda:0"
model_args_map["/mnt/scratch-artemis/gviveiros/TowerVision/TowerVision-Gemma2b-Base"]="device_map=cuda:0,device=cuda:0"
model_args_map["ckp-19360"]="device_map=cuda:0,device=cuda:0"
model_args_map["/mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-2B-0.4"]="device_map=cuda:0,device=cuda:0"
model_args_map["/mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-9B-0.4"]="device_map=cuda:0,device=cuda:0"
model_args_map["/mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-9B-0.2"]="device_map=cuda:0,device=cuda:0"
model_args_map["/mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-9B-0.8"]="device_map=cuda:0,device=cuda:0"
# vision encoder merged
model_args_map["/mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-vision-9B-0.2"]="device_map=cuda:0,device=cuda:0"
model_args_map["/mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-vision-9B-0.4"]="device_map=cuda:0,device=cuda:0"
model_args_map["/mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-vision-9B-0.8"]="device_map=cuda:0,device=cuda:0"
# all merged
model_args_map["/mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-both-9B-0.2"]="device_map=cuda:0,device=cuda:0"
model_args_map["/mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-both-9B-0.4"]="device_map=cuda:0,device=cuda:0"
model_args_map["/mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-both-9B-0.8"]="device_map=cuda:0,device=cuda:0"

model_args_map["Unbabel/Tower-Plus-9B"]="max_images=0,max_videos=0,max_audios=0"
model_args_map["Unbabel/Tower-Plus-2B"]="max_images=0,max_videos=0,max_audios=0"



declare -A model_types

# both Phi and Pixtral seems to use different transformers versions - transformers==4.51.3
# waiting for new env


model_types["mistralai/Pixtral-12B-2409"]="pixtral" # works
model_types["microsoft/Phi-4-multimodal-instruct"]="phi4_multimodal" # does not work
model_types["neulab/CulturalPangea-7B"]="pangea"
model_types["google/gemma-3-4b-it"]="gemma3"
model_types["google/gemma-3-12b-it"]="gemma3"
model_types["CohereForAI/aya-vision-8b"]="aya"
model_types["Qwen/Qwen2.5-VL-7B-Instruct"]="qwen2_5_vl"
model_types["Qwen/Qwen2.5-VL-3B-Instruct"]="qwen2_5_vl"
model_types["llava-hf/llama3-llava-next-8b-hf"]="llava_hf"
# tower evals
model_types["utter-project/TowerVision-Plus-9B"]="llava_hf"
model_types["utter-project/TowerVision-Plus-2B"]="llava_hf"
model_types["/mnt/scratch-artemis/gviveiros/TowerVision/towerp_2b_instruct"]="llava_hf"
model_types["/mnt/scratch-artemis/gviveiros/TowerVision/towerp_9b_instruct"]="llava_hf"
model_types["Unbabel/Tower-Plus-2B"]="vllm"
model_types["Unbabel/Tower-Plus-9B"]="vllm"
model_types["/mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-2B-0.4"]="llava_hf"
model_types["/mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-9B-0.4"]="llava_hf"
model_types["/mnt/scratch-artemis/gviveiros/TowerVision/TowerVision-Gemma2b-Base"]="llava_hf"
model_types["/mnt/scratch-artemis/gviveiros/TowerVision/test_inst_2b/TowerVision-Plus-2B"]="llava_hf"
model_types["/mnt/scratch-artemis/gviveiros/TowerVision/test_inst_9b/TowerVision-Plus-9B"]="llava_hf"
model_types["ckp-19360"]="llava_hf"
model_types["Unbabel/Tower-Plus-9B"]="llava_hf"
model_types["/mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-9B-0.2"]="llava_hf"
model_types["/mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-9B-0.8"]="llava_hf"
model_types["/mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-vision-9B-0.2"]="llava_hf"
model_types["/mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-vision-9B-0.4"]="llava_hf"
model_types["/mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-vision-9B-0.8"]="llava_hf"
model_types["/mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-both-9B-0.2"]="llava_hf"
model_types["/mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-both-9B-0.4"]="llava_hf"
model_types["/mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-both-9B-0.8"]="llava_hf"
model_types["Unbabel/Tower-Plus-9B"]="vllm"
model_types["Unbabel/Tower-Plus-2B"]="vllm"
#models=(utter-project/TowerVision-Plus-9B)

#models=(CohereForAI/aya-vision-8b)
#models=(utter-project/TowerVision-Plus-9B) # Qwen/Qwen2.5-VL-7B-Instruct neulab/CulturalPangea-7B)
#models=(llava-hf/llama3-llava-next-8b-hf CohereForAI/aya-vision-8b)
#models=(utter-project/TowerVision-Plus-9B)
#models=(Qwen/Qwen2.5-VL-7B-Instruct Qwen/Qwen2.5-VL-3B-Instruct)
#models=(google/gemma-3-12b-it google/gemma-3-4b-it)
#models=(Unbabel/Tower-Plus-9B)
#tasks=(wmt24pp)
#models=()
tasks=(mmlu-global)
#tasks=(kaleidoscope-bench-vision)
# export TRANSFORMERS_VERBOSITY=info

models=(/mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-both-9B-0.8 /mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-9B-0.2 /mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-9B-0.8 /mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-2B-0.4 /mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-9B-0.8 /mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-9B-0.4 /mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-vision-9B-0.2 /mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-vision-9B-0.4 /mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-vision-9B-0.8 /mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-both-9B-0.2 /mnt/scratch-artemis/gviveiros/TowerVision/TowerVisionMerged-Plus-both-9B-0.4)
tasks=(cc-ocr-multi-lan ocrbench alm_bench-all)

#models=(Unbabel/Tower-Plus-9B )

for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        echo "--------------------------------------------------------------------------------------"
        
        echo "=========================================="
        echo "Running evaluation for task: $task"
        echo "=========================================="
        
        echo "Running with the following arguments:
        pretrained: $model
        task: $task
        batch_size: 140
        verbosity: DEBUG
        " 

        # reset my gpu memory
        #nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d '[:space:]'
        
        
    
        model_name=$(echo $model | sed 's/\//__/g')    
        model_arguments="${model_args_map[$model]}"
        model_type="${model_types[$model]}"
        model_args="pretrained=$model,$model_arguments"

        echo "model_args: $model_args"
        # if task already exists, for this model, skip
        if [ -d "./logs/$task/$model_name" ]; then
            echo "Task $task already exists, skipping"
            continue
        fi

        # reset my gpu memory
        echo "Resetting gpu memory"
        python -c "import torch; torch.cuda.empty_cache()"

        echo "Running evaluation"
        python -m lmms_eval \
        --model $model_type \
        --model_args $model_args \
        --tasks $task \
        --verbosity=DEBUG \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix $task \
        --output_path ./logs/$task \
        --verbosity=DEBUG 

    done
done

#python -m lmms_eval --model llava_hf --model_args pretrained=utter-project/TowerVision-Plus-9B,device_map=cuda:0,device=cuda:0 --tasks global_mmlu --verbosity=DEBUG --batch_size 6 --log_samples --log_samples_suffix global_mmlu --output_path ./logs/global_mmlu --verbosity=DEBUG 
#python -m lmms_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,device_map=cuda:0,device=cuda:0 --tasks kaleidoscope-bench-vision --verbosity=DEBUG --batch_size 6 --log_samples --log_samples_suffix kaleidoscope-bench-vision --output_path ./logs/kaleidoscope-bench-vision --verbosity=DEBUG 

