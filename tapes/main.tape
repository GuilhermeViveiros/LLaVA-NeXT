import "scslurm.tape"  # Assuming you have this file for SLURM configuration

global {
    ducttape_experimental_imports=true
    ducttape_experimental_submitters=true
    ducttape_experimental_multiproc=true
}

func TrainModel
    < initial_model_dir
    > trained_model_dir
    :: repo
    :: text_model
    :: vision_model
    :: data_folder
    :: data_yaml
    :: prompt_version
    :: system_from_data
    :: external_model_dir
    :: num_gpus
    :: nnodes
    :: save_strategy
    :: save_steps
    :: num_epochs
    :: micro_batch_size
    :: global_batch_size
    :: learning_rate
    :: warmup_ratio
    :: model_max_length
    :: training_mode  # "pretrain" or "finetune"
    :: vision_tower_lr
    :: max_tiles
    :: group_by_modality
    :: use_flash_attn
    :: use_torch_compile
    :: torch_compile_backend
    :: use_liger
    :: wandb_project
{
    # For debugging purposes
    export NCCL_DEBUG=WARN

    export WANDB_PROJECT=$wandb_project
    export WANDB_MODE=offline

    # Clean model names for directory naming
    TEXT_MODEL_CLEAN="${text_model//\//_}"
    VISION_MODEL_CLEAN="${vision_model//\//_}"
    DATASET_NAME_CLEAN="${data_yaml##*/}"
    DATASET_NAME_CLEAN="${DATASET_NAME_CLEAN%.*}"

    # Create base run name
    RUN_NAME="${training_mode}::${TEXT_MODEL_CLEAN}:${VISION_MODEL_CLEAN}:${DATASET_NAME_CLEAN}"
    echo "RUN_NAME: ${RUN_NAME}"

    # Create output directory
    if [ "$external_model_dir" != "" ]; then
        mkdir -p $external_model_dir
        ln -sf $external_model_dir $trained_model_dir
    else
        mkdir -p $trained_model_dir
    fi

    # Extract mm_projector for finetuning
    PROJECTOR_ARG=""
    if [ "$training_mode" == "finetune" ]; then
        # check if initial_model_dir is exists and is a directory
        if [ ! -d "$initial_model_dir" ]; then
            echo "ERROR: initial_model_dir does not exist or is not a directory"
            exit 1
        fi

        echo "Extracting mm_projector from pretrained checkpoint..."
        if [ -f "$initial_model_dir/mm_projector.bin" ]; then
            PROJECTOR_ARG="--pretrain_mm_mlp_adapter=$initial_model_dir/mm_projector.bin"
        else
            echo "ERROR: mm_projector.bin not found in pretrained checkpoint"
            exit 1
        fi
    fi

    # define data variables
    #data_path="${data_folder}/${data_yaml}"
    data_path=$data_yaml
    image_folder="${data_folder}/images"

    # Set training-mode specific arguments
    if [ "$training_mode" == "pretrain" ]; then
        TRAIN_ARGS="--mm_tunable_parts=mm_mlp_adapter"
        TRAIN_ARGS="$TRAIN_ARGS --save_strategy no"
    else
        # function to automatically compute image grid pinpoints
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

        tile_grids=$(get_tile_grids $max_tiles)
        echo "Using 'tile_grids=${tile_grids}'"
        TRAIN_ARGS="--mm_tunable_parts=mm_vision_tower,mm_mlp_adapter,mm_language_model"
        TRAIN_ARGS="$TRAIN_ARGS --image_aspect_ratio anyres"
        TRAIN_ARGS="$TRAIN_ARGS --image_grid_pinpoints \"${tile_grids}\""
        TRAIN_ARGS="$TRAIN_ARGS --mm_patch_merge_type spatial_unpad"
        TRAIN_ARGS="$TRAIN_ARGS --dataloader_drop_last True"
        TRAIN_ARGS="$TRAIN_ARGS --mm_vision_tower_lr ${vision_tower_lr}"
    fi

    # TODO: gradient accumulation can be computed from global_batch_size and micro_batch_size
    grad_accum_steps=$((global_batch_size / micro_batch_size / num_gpus / nnodes))
    echo "Using 'grad_accum_steps=${grad_accum_steps}'"

    # Compute infra args
    master_addr=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    master_port=6020
    rank=$SLURM_PROCID

    # Run the training command
    echo "num_gpus: ${num_gpus}"
    echo "nnodes: ${nnodes}"
    echo "rank: ${rank}"
    echo "master_addr: ${master_addr}"
    echo "master_port: ${master_port}"
    ACCELERATE_CPU_AFFINITY=1 torchrun \
        --nproc_per_node="${num_gpus}" --nnodes="${nnodes}" --node_rank="${rank}" --master_addr="${master_addr}" --master_port="${master_port}" \
        ${repo}/llava/train/train_mem.py \
        --deepspeed ${repo}/scripts/zero3.json \
        --model_name_or_path ${text_model} \
        --vision_tower ${vision_model} \
        --version ${prompt_version} \
        $([ "$drop_images_ratio" != "none" ] && echo "--drop_images_ratio ${drop_images_ratio}" || echo "") \
        $([ "$system_from_data" == "true" ] && echo "--system_from_data True" || echo "") \
        --data_path ${data_path} \
        --image_folder ${image_folder} \
        --output_dir $trained_model_dir \
        $PROJECTOR_ARG \
        $TRAIN_ARGS \
        --mm_vision_select_layer -2 \
        --mm_projector_type mlp2x_gelu \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --num_train_epochs ${num_epochs} \
        --per_device_train_batch_size ${micro_batch_size} \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps ${grad_accum_steps} \
        --evaluation_strategy "no" \
        --save_strategy ${save_strategy} \
        --save_steps ${save_steps} \
        $([ "$group_by_modality" == "true" ] && echo "--group_by_modality_length True" || echo "") \
        --learning_rate ${learning_rate} \
        --weight_decay 0. \
        --warmup_ratio ${warmup_ratio} \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --bf16 True \
        --tf32 True \
        --model_max_length ${model_max_length} \
        --gradient_checkpointing True \
        --dataloader_num_workers 8 \
        --lazy_preprocess True \
        --report_to wandb \
        --disable_tqdm True \
        --run_name $RUN_NAME \
        $([ "$use_flash_attn" == "true" ] && echo "" || echo "--attn_implementation sdpa") \
        $([ "$use_torch_compile" == "true" ] && echo "--torch_compile True --torch_compile_backend ${torch_compile_backend}" || echo "") \
        $([ "$use_liger" == "true" ] && echo "--use_liger True" || echo "")
}

task PretrainModel calls TrainModel
    < initial_model_dir=/dev/null
    > trained_model_dir
    :: repo=@
    :: text_model=@
    :: vision_model=@
    :: data_folder=@
    :: data_yaml=$pretrain_data
    :: prompt_version=$pretrain_prompt
    :: drop_images_ratio=none
    :: system_from_data=$pretrain_system_from_data
    :: external_model_dir=$pretrain_model_dir
    :: num_gpus=$pretrain_gpus
    :: nnodes=$pretrain_nnodes
    :: num_epochs=$pretrain_epochs
    :: micro_batch_size=$pretrain_micro_bsize
    :: global_batch_size=$pretrain_global_bsize
    :: learning_rate=$pretrain_lr
    :: warmup_ratio=$pretrain_warmup
    :: model_max_length=$pretrain_max_length
    :: training_mode=pretrain
    :: max_tiles=1
    :: vision_tower_lr=none
    :: group_by_modality=false
    :: save_strategy=no
    :: save_steps=0
    :: use_flash_attn=true
    :: use_torch_compile=false
    :: torch_compile_backend=inductor
    :: use_liger=true
    :: wandb_project=@
    :: .submitter=@
    :: .C=$pretrain_C
    :: .account=$pretrain_account
    :: .partition=$pretrain_partition
    :: .qos=$pretrain_qos
    :: .nodes=$pretrain_nodes
    :: .gres=$pretrain_gres
    :: .cpus=$pretrain_cpus
    :: .time=$pretrain_time

task FinetuneModel calls TrainModel
    < initial_model_dir=$trained_model_dir@PretrainModel
    > trained_model_dir
    :: repo=@
    :: text_model=@
    :: vision_model=@
    :: data_folder=@
    :: data_yaml=$finetune_data
    :: prompt_version=$finetune_prompt
    :: drop_images_ratio=$drop_images_ratio
    :: system_from_data=$finetune_system_from_data
    :: external_model_dir=$finetune_model_dir
    :: num_gpus=$finetune_gpus
    :: nnodes=$finetune_nnodes
    :: num_epochs=$finetune_epochs
    :: micro_batch_size=$finetune_micro_bsize
    :: global_batch_size=$finetune_global_bsize
    :: learning_rate=$finetune_lr
    :: warmup_ratio=$finetune_warmup
    :: model_max_length=$finetune_max_length
    :: training_mode=finetune
    :: vision_tower_lr=$finetune_vision_lr
    :: group_by_modality=true
    :: max_tiles=$max_tiles
    :: save_strategy=steps
    :: save_steps=$finetune_save_steps
    :: use_flash_attn=true
    :: use_torch_compile=$finetune_compile
    :: torch_compile_backend=inductor
    :: use_liger=true
    :: wandb_project=@
    :: .submitter=@
    :: .C=$finetune_C
    :: .account=$finetune_account
    :: .partition=$finetune_partition
    :: .qos=$finetune_qos
    :: .nodes=$finetune_nodes
    :: .gres=$finetune_gres
    :: .cpus=$finetune_cpus
    :: .time=$finetune_time
    :: .restart_on_timeout=true


task EvaluateModel
    < trained_model_dir=(
        UseExternal:
            false=$trained_model_dir@FinetuneModel
            true=$finetune_model_dir
    )
    > logs
    :: num_gpus=$eval_gpus
    :: tasks=$eval_tasks
    :: step=$eval_step
    :: batch_size=1
    :: .submitter=@
    :: .C=$eval_C
    :: .account=$eval_account
    :: .partition=$eval_partition
    :: .qos=$eval_qos
    :: .nodes=$eval_nodes
    :: .gres=$eval_gres
    :: .cpus=$eval_cpus
    :: .time=$eval_time
{
    # if step is not none
    if [ "$step" != "none" ]; then
        trained_model_dir="${trained_model_dir}/checkpoint-${step}"
    fi

    # if trained_model_dir is not a directory, exit
    #if [ ! -d "$trained_model_dir" ]; then
    #    echo "ERROR: trained_model_dir does not exist or is not a directory"
    #    echo "trained_model_dir: ${trained_model_dir}"
    #    exit 1
    #fi

    echo "Evaluating model..."
    echo "trained_model_dir: ${trained_model_dir}"
    python -m accelerate.commands.launch \
        --num_processes=${num_gpus} \
        -m lmms_eval \
        --model llava \
        --model_args pretrained=${trained_model_dir},model_name=qwen_2 \
        --tasks ${tasks} \
        --batch_size ${batch_size} \
        --log_samples \
        --log_samples_suffix "$tasks" \
        --output_path ./logs/
}


task SyncWandB
    < trained_model_dir=@FinetuneModel
    :: submitter=shell
{
    cd $trained_model_dir/..
    wandb sync --sync-all
}

task UploadModel
    < trained_model_dir=@FinetuneModel
    :: submitter=shell
    :: upload_id=@
{
    # copy non-directory files from trained_model_dir to new folder (to avoid uploading intermediate checkpoints)
    mkdir -p upload_model
    for file in $trained_model_dir/*; do
        if [ -f "$file" ]; then
            cp $file upload_model/
        fi
    done

    huggingface-cli upload $upload_id upload_model .

    # clean up
    rm -rf upload_model
}