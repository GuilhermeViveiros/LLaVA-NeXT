#!/bin/bash

JSON_PATH=/lustre/fsn1/projects/rech/qjm/ued79zb/lnext-datasets/multi30k_enfr.json
IMAGE_PATH=/lustre/fsn1/projects/rech/qjm/ued79zb/lnext-datasets/images/multi30k_enfr/
SRC_LANG="English"
TGT_LANG="French"
SRC_FIELD="en"
TGT_FIELD="fr"

PROMPT_TEMPLATE="<image>
Translate from ${SRC_LANG} to ${TGT_LANG}:
{${SRC_FIELD}}
"

# Optional: Process dataset for other languages as well
python scripts/prepare_dataset.py \
    --dataset_name "romrawinjp/multi30k" \
    --split "train" \
    --output_file "$JSON_PATH" \
    --image_dir "$IMAGE_PATH" \
    --image_field "image" \
    --target_field "$TGT_FIELD" \
    --prompt_template "$PROMPT_TEMPLATE" \
    --system_prompt "You are a helpful assistant that translates between languages. Optionally, some image that provide context for the translation might be provided."