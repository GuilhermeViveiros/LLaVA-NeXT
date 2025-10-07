#! /bin/bash

benchmarks=("commute-all-contrastive")
#benchmarks=("alm_bench-all"
#benchmarks=("m3exam" "ocrbench" "cc-ocr-multi-lan")
#models=("towerp_2b_instruct_full" "towerp_2b_base_full")
models=("towerp_2b_instruct_full" "towerp_2b_base_full" "towerp_9b_base_full" "towerp_9b_instruct_full")
for benchmark in "${benchmarks[@]}"; do
    for model in "${models[@]}"; do
        echo "Processing $benchmark with $model"
        python -m scripts.cxmi_llnext --model_path /mnt/scratch-artemis/gviveiros/TowerVision/llava-next-native/ --benchmark_name $benchmark --model_name $model
    done
done
