global {
    ducttape_output=/lustre/fswork/projects/rech/qjm/ued79zb/tvision-lnext-mtranslation
    
    # Base repository and paths
    repo=/linkhome/rech/genrce01/ued79zb/repos/LLaVA-NeXT
    submitter=shell
    
    # Model parameters
    text_model=(
        TextModel:
            qwen2p5_7b="Qwen/Qwen2.5-7B-Instruct"
            qwen2p5_14b="Qwen/Qwen2.5-14B-Instruct"
    )
    vision_model=(
        VisionModel:
            clip="openai/clip-vit-large-patch14-336"
            siglip2="google/siglip2-so400m-patch14-384"
    )
    
    data_folder="/lustre/fsn1/projects/rech/qjm/uar73ie/datasets/llava_datasets/VisionBlocks-Llava-Format"
    
    # -- Pretraining parameters --
    pretrain_data="/lustre/fsn1/projects/rech/qjm/uar73ie/datasets/llava_datasets/VisionBlocks-Llava-Format/VisionBlocks-pixmo-cap.json"
    pretrain_prompt="plain"
    pretrain_system_from_data=false
    pretrain_gpus=4
    pretrain_nnodes=2
    pretrain_epochs=1
    pretrain_micro_bsize=8
    pretrain_global_bsize=256
    pretrain_lr=1e-3
    pretrain_warmup=0.05
    pretrain_max_length=8192
    pretrain_model_dir=(
        TextModel:
            qwen2p5_7b=(
                VisionModel:
                    clip="/lustre/fsn1/projects/rech/qjm/ued79zb/tvision-lnext-ckpts-vblocks/pretrain/qwen2p5_7b"
                    siglip2="/lustre/fsn1/projects/rech/qjm/ued79zb/tvision-lnext-ckpts-vblocks/pretrain/qwen2p5_7b_siglip2"
            )
            qwen2p5_14b=(
                VisionModel:
                    clip="/lustre/fsn1/projects/rech/qjm/ued79zb/tvision-lnext-ckpts-vblocks/pretrain/qwen2p5_14b_clip"
                    siglip2="/lustre/fsn1/projects/rech/qjm/ued79zb/tvision-lnext-ckpts-vblocks/pretrain/qwen2p5_14b_siglip2"
            )
    )

    # -- Finetuning parameters --
    max_tiles=4
    drop_images_ratio=(DropRatio: 
        0=none 
        0.5=0.5 
        1=1
    )

    finetune_data=(
        Dataset:
            prototype=/lustre/fsn1/projects/rech/qjm/ued79zb/lnext-datasets/multi30k_enfr.json
            multi30k_enfr=/lustre/fsn1/projects/rech/qjm/ued79zb/lnext-datasets/multi30k_enfr.json
            multi30k_frcs_mix1=/linkhome/rech/genrce01/ued79zb/repos/LLaVA-NeXT/multi30k_frcs_mix1.yaml
            multi30k_frcs_mix2=/linkhome/rech/genrce01/ued79zb/repos/LLaVA-NeXT/multi30k_frcs_mix2.yaml
            mtranslation_fr=/linkhome/rech/genrce01/ued79zb/repos/LLaVA-NeXT/mtranslation_fr.yaml
    )
    finetune_prompt=(TextModel: qwen2p5_7b="qwen_1_5" qwen2p5_14b="qwen_1_5")
    finetune_system_from_data=true
    finetune_gpus=4
    finetune_nnodes=1
    finetune_epochs=1
    finetune_micro_bsize=4
    finetune_global_bsize=128
    finetune_lr=1e-5
    finetune_vision_lr=2e-6
    finetune_warmup=0.05
    finetune_save_steps=0.2
    finetune_max_length=4096
    finetune_model_dir=(
        Dataset:
            prototype="/lustre/fsn1/projects/rech/qjm/ued79zb/tvision-lnext-ckpts-mtranslation/finetune/qwen2p5_7b_test"
            multi30k_enfr=(
                DropRatio:
                    0="/lustre/fsn1/projects/rech/qjm/ued79zb/tvision-lnext-ckpts-mtranslation/finetune/qwen2p5_7b_multi30k_enfr"
                    0.5="/lustre/fsn1/projects/rech/qjm/ued79zb/tvision-lnext-ckpts-mtranslation/finetune/qwen2p5_7b_multi30k_enfr_dr0.5"
                    1="/lustre/fsn1/projects/rech/qjm/ued79zb/tvision-lnext-ckpts-mtranslation/finetune/qwen2p5_7b_multi30k_enfr_dr1"
            )
            multi30k_frcs_mix1=(
                DropRatio:
                    0="/lustre/fsn1/projects/rech/qjm/ued79zb/tvision-lnext-ckpts-mtranslation/finetune/qwen2p5_7b_multi30k_frcs_mix1"
                    0.5="/lustre/fsn1/projects/rech/qjm/ued79zb/tvision-lnext-ckpts-mtranslation/finetune/qwen2p5_7b_multi30k_frcs_mix1_dr0.5"
                    1="/lustre/fsn1/projects/rech/qjm/ued79zb/tvision-lnext-ckpts-mtranslation/finetune/qwen2p5_7b_multi30k_frcs_mix1_dr1"
            )
            multi30k_frcs_mix2=(
                DropRatio:
                    0="/lustre/fsn1/projects/rech/qjm/ued79zb/tvision-lnext-ckpts-mtranslation/finetune/qwen2p5_7b_multi30k_frcs_mix2"
                    0.5="/lustre/fsn1/projects/rech/qjm/ued79zb/tvision-lnext-ckpts-mtranslation/finetune/qwen2p5_7b_multi30k_frcs_mix2_dr0.5"
                    1="/lustre/fsn1/projects/rech/qjm/ued79zb/tvision-lnext-ckpts-mtranslation/finetune/qwen2p5_7b_multi30k_frcs_mix2_dr1"
            )
            mtranslation_fr=(
                DropRatio:
                    0="/lustre/fsn1/projects/rech/qjm/ued79zb/tvision-lnext-ckpts-mtranslation/finetune/qwen2p5_7b_mtranslation_fr"
                    0.5="/lustre/fsn1/projects/rech/qjm/ued79zb/tvision-lnext-ckpts-mtranslation/finetune/qwen2p5_7b_mtranslation_fr_dr0.5"
                    1="/lustre/fsn1/projects/rech/qjm/ued79zb/tvision-lnext-ckpts-mtranslation/finetune/qwen2p5_7b_mtranslation_fr_dr1"
            )
    )
    finetune_compile=false

    # -- Evaluation parameters --
    eval_gpus=1
    eval_tasks=(EvalTask: multi30k-fr multi30k-all commute-all-contrastive)
    eval_system_prompt="mtranslation"
    eval_step=(
        Dataset:
            prototype=none
            multi30k_enfr=(
                EvalStep:
                    last="none"
                    0p8="184"
                    0p6="138"
                    0p4="92"
                    0p2="46"
            )
            multi30k_frcs_mix1=(
                EvalStep:
                    last="none"
                    0p8="184"
                    0p6="138"
                    0p4="92"
                    0p2="46"
            )
            multi30k_frcs_mix2=(
                EvalStep:
                    last="none"
                    0p8="184"
                    0p6="138"
                    0p4="92"
                    0p2="46"
            )
            mtranslation_fr=(
                EvalStep:
                    last="none"
                    0p8="264"
                    0p6="198"
                    0p4="132"
                    0p2="66"
            )
    )

    # upload_id for HuggingFace
    upload_id="Unbabel/lnext-qwen2p5-7b-test"

    # Wandb configuration
    wandb_project="llavanext-mtranslation"
    wandb_entity="coderpat"
    
    # -------------
    # --- SLURM ---
    # -------------
    pretrain_C="h100"
    pretrain_account="qjm@h100"
    pretrain_time="8:00:00"
    pretrain_cpus=80
    pretrain_partition=none # "gpu"
    pretrain_qos=none
    pretrain_nodes=2
    pretrain_gres="gpu:4"

    finetune_C="h100"
    finetune_account="qjm@h100"
    finetune_time="20:00:00"
    finetune_cpus=80
    finetune_partition=none
    #finetune_qos="qos_gpu_h100-t4"
    finetune_qos=none
    finetune_nodes=1
    finetune_gres="gpu:4"

    eval_C="h100"
    eval_account="qjm@h100"
    eval_time="1:00:00"
    eval_cpus=40
    eval_partition=none
    eval_qos=none
    eval_nodes=1
    eval_gres="gpu:4"
}

plan Qwen2p5_7b_MTranslation {
    reach FinetuneModel via (Dataset: multi30k_enfr) * (DropRatio: 0 0.5 1)
    reach FinetuneModel via (Dataset: multi30k_frcs_mix1) * (DropRatio: 0 0.5 1)
    reach FinetuneModel via (Dataset: multi30k_frcs_mix2) * (DropRatio: 0 0.5 1)
    reach FinetuneModel via (Dataset: mtranslation_fr) * (DropRatio: 0 0.5 1)
}

plan Qwen2p5_7b_MTranslation_Evals {
    #reach ParseResults via (Dataset: multi30k_enfr) * (DropRatio: 0 0.5 1) * (EvalStep: 0p2 0p4 0p6 0p8 last) * (EvalTask: multi30k-all commute-all-contrastive)
    reach ParseResults via (Dataset: mtranslation_fr) * (DropRatio: 0 1) * (EvalStep: 0p2 0p4 0p6 0p8 last) * (EvalTask: multi30k-all commute-all-contrastive)
    reach ParseResults via (Dataset: multi30k_frcs_mix1) * (DropRatio: 0 1) * (EvalStep: 0p2 0p4 0p6 0p8 last) * (EvalTask: multi30k-all commute-all-contrastive)
    reach ParseResults via (Dataset: multi30k_frcs_mix2) * (DropRatio: 0 1) * (EvalStep: 0p2 0p4 0p6 0p8 last) * (EvalTask: multi30k-all commute-all-contrastive)
}