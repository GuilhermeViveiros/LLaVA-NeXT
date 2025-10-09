#!/bin/bash

# Enable conda in this shell
eval "$(conda shell.bash hook)"

# Activate environment
conda activate llava-next-env

cd /home/gviveiros/LLaVA-NeXT

echo "Current directory: $(pwd)"


# cultural ground and Phi run here


export PYTHONPATH='/home/gviveiros/LLaVA-NeXT/lm-evaluation-harness:$PYTHONPATH'


# both Phi and Pixtral seems to use different transformers versions - transformers==4.51.3
# waiting for new env


batch_size=12
models=(Unbabel/Tower-Plus-9B)
tasks=(wmt24pp)

# lm_eval --model vllm \
#     --model_args pretrained=EleutherAI/gpt-j-6B \
#     --tasks hellaswag \
#     --device cuda:0 \
#     --batch_size 8

#mmlu_global
echo "Running evaluation"
python -m lm_eval \
--model hf \
--model_args pretrained=Unbabel/Tower-Plus-9B,dtype="bfloat16" \
--device cuda:0 \
--tasks wmt24pp \
--verbosity=DEBUG \
--batch_size 1 \
--num_fewshot 1 \
--output_path ./logs/wmt24pp \
--verbosity=DEBUG 

#python -m lmms_eval --model llava_hf --model_args pretrained=utter-project/TowerVision-Plus-9B,device_map=cuda:0,device=cuda:0 --tasks global_mmlu --verbosity=DEBUG --batch_size 6 --log_samples --log_samples_suffix global_mmlu --output_path ./logs/global_mmlu --verbosity=DEBUG 
#python -m lmms_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,device_map=cuda:0,device=cuda:0 --tasks kaleidoscope-bench-vision --verbosity=DEBUG --batch_size 6 --log_samples --log_samples_suffix kaleidoscope-bench-vision --output_path ./logs/kaleidoscope-bench-vision --verbosity=DEBUG 

