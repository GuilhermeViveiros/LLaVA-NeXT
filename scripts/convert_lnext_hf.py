# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert LLaVa-NeXT (LLaVa-1.6) checkpoints from the original repository.

URL: https://github.com/haotian-liu/LLaVA/tree/main.


The command used to obtain original logits is the following:
python llava/eval/run_llava.py --model-path "liuhaotian/llava-v1.6-mistral-7b" --image-file "images/llava_v1_5_radar.jpg" --query "What is shown in this image?" --max_new_tokens 100 --temperature 0

Note: logits are tested with torch==2.1.2.
"""

import argparse
import gc
import glob
import json
import os
import logging
from pathlib import Path

import requests
import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from safetensors import safe_open

from transformers import (
    AddedToken,
    AutoConfig,
    AutoTokenizer,
    LlavaNextConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextImageProcessor,
    LlavaNextProcessor,
    SiglipVisionConfig
)


logger = logging.getLogger(__name__)

KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.": "",
    "model.mm_projector": "multi_modal_projector",
    "model": "model.model",
    #"vision_tower.vision_model": "vision_model",
    "vision_model.model": "vision_model",
    "lm_head": "language_model.lm_head",
    "model.model": "language_model.model",
    "multi_modal_projector.0": "multi_modal_projector.linear_1",
    "multi_modal_projector.2": "multi_modal_projector.linear_2",
    "language_model.model.image_newline": "image_newline",
}


def load_original_state_dict(model_id, local_model_registry=None):
    """
    Loads the original state dict from safetensors files.
    If local_model_registry is provided and exists, loads from local path, otherwise downloads from HuggingFace Hub.

    Args:
        model_id (str): Model identifier for HuggingFace Hub or local path.
        local_model_registry (str, optional): Local directory containing model files.

    Returns:
        dict: State dictionary mapping parameter names to tensors.
    """

    # Prefer local_model_registry if provided and exists
    if local_model_registry is not None and os.path.exists(str(local_model_registry)):
        directory_path = str(local_model_registry)
    else:
        # Otherwise, download from HuggingFace Hub
        directory_path = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors"])

    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    return original_state_dict


def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        new_state_dict[key] = value.to(torch.float16)

    if "language_model.model.embed_tokens.weight" in new_state_dict and "language_model.lm_head.weight" not in new_state_dict:
        print("Adding tied lm_head weights...")
        new_state_dict["language_model.lm_head.weight"] = new_state_dict["language_model.model.embed_tokens.weight"]
    return new_state_dict


def load_image():
    url = "https://cms.mistral.ai/assets/a10b924e-56b3-4359-bf6c-571107811c8f"
    image = Image.open(requests.get(url, stream=True).raw)
    #img_path = "docs/ov_chat_images/example1_tree.png"
    #image = Image.open(img_path)
    return image


def convert_llava_to_hf(model_id, local_model_registry, pytorch_dump_folder_path, push_to_hub=False):
    
    # load original config
    if local_model_registry is not None:
        filepath = Path(local_model_registry) / "config.json"
    else:
        filepath = hf_hub_download(repo_id=model_id, filename="config.json", repo_type="model")
    
    # read json
    with open(filepath) as f:
        data = json.load(f)
        print(data)

    if model_id == "liuhaotian/llava-v1.6-mistral-7b":
        text_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        image_token_id = 32000
    elif model_id == "liuhaotian/llava-v1.6-vicuna-7b":
        text_model_id = "lmsys/vicuna-7b-v1.5"
        image_token_id = 32000
    elif model_id == "liuhaotian/llava-v1.6-vicuna-13b":
        text_model_id = "lmsys/vicuna-13b-v1.5"
        image_token_id = 32000
    elif model_id == "liuhaotian/llava-v1.6-34b":
        text_model_id = "NousResearch/Nous-Hermes-2-Yi-34B"
        image_token_id = 64000
    elif model_id == "lmms-lab/llama3-llava-next-8b":
        text_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        image_token_id = 128256
    elif model_id == "lmms-lab/llava-next-72b":
        text_model_id = "Qwen/Qwen1.5-72B-Chat"
        image_token_id = 151646
    elif model_id == "lmms-lab/llava-next-110b":
        text_model_id = "Qwen/Qwen1.5-110B-Chat"
        image_token_id = 151646
    elif model_id == "Unbabel/lnext-towerqmix-7b-dpo-siglip2-v6":
        text_model_id = "Unbabel/TowerQwen-2.5-7B-WPO-2e-7"
        image_token_id = 151655
    elif model_id == "Unbabel/TowerVision-Plus-2B":
        text_model_id = "Unbabel/TowerVision-Plus-2B"
        image_token_id = 256000 # fix hardcoded: tokenizer.convert_tokens_to_ids("<image>")
    elif model_id == "Unbabel/TowerVision-Plus-9B":
        text_model_id = "Unbabel/TowerVision-Plus-9B"
        image_token_id = 256000 # fix hardcoded: tokenizer.convert_tokens_to_ids("<image>")
    elif model_id == "Unbabel/TowerVision-4-Anthill-CPT":
        text_model_id = "Unbabel/TowerVision-4-Anthill-CPT"
        image_token_id = 256000 # fix hardcoded: tokenizer.convert_tokens_to_ids("<image>")
    else:
        raise ValueError(f"Model {model_id} not supported")


    torch.set_default_dtype(torch.float16)
    model_path = text_model_id if local_model_registry is None else local_model_registry
    text_config = AutoConfig.from_pretrained(model_path)
    
    vision_model_id = data["mm_vision_tower"]
    vision_config = AutoConfig.from_pretrained(vision_model_id)
    if hasattr(vision_config, "vision_config"):
        vision_config = vision_config.vision_config

    feature_layer = -2
    if isinstance(vision_config, SiglipVisionConfig):
        vision_config.num_hidden_layers = vision_config.num_hidden_layers - 1
        vision_config.vision_use_head = False
        feature_layer = -1

    use_fast = False if model_id == "liuhaotian/llava-v1.6-34b" else True

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
   
    # WARNING: hardcoded fixes to the tokenizer
    # tokenizer.add_bos_token = False
    # tokenizer.bos_token = "<|im_start|>"
    # tokenizer.eos_token = "<|im_end|>"

    image_processor = LlavaNextImageProcessor.from_pretrained(vision_model_id)

    # fix some sethings based on `data` 
    image_processor.do_convert_rgb = False # TODO: fix hardcoded
    image_processor.vision_feature_select_strategy = "full" # TODO: fix hardcoded
    image_processor.aspect_ratio_setting = data["image_aspect_ratio"]
    image_processor.image_grid_pinpoints = data["image_grid_pinpoints"]
    image_processor.crop_size = image_processor.size
    image_processor.size = {"shortest_edge": image_processor.size["width"]}

    if model_id in ("liuhaotian/llava-v1.6-mistral-7b", "lmms-lab/llama3-llava-next-8b"):
        # Mistral-7B doesn't have a padding token set yet
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    
    processor = LlavaNextProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        patch_size=vision_config.patch_size,
        image_token="<image>",
        vision_feature_select_strategy=image_processor.vision_feature_select_strategy,
    )

    # import pdb; pdb.set_trace()
    # WARNING: hardcoded fixes to the processor
    # processor.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' }}{% if message['content'] is string %}{{ message['content'] }}{% else %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] }}{% endfor %}{% endif %}{{ '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""

    chat_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'assistant' %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}<start_of_turn>{{ role }}\n{{ message['content'] | trim }}<end_of_turn>\n{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
    processor.chat_template = chat_template

    config = LlavaNextConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        image_grid_pinpoints=image_processor.image_grid_pinpoints,
        use_image_newline_parameter=True,
        image_token_id=image_token_id,
        vision_feature_select_strategy=image_processor.vision_feature_select_strategy,
        vision_feature_layer=feature_layer,
    )

    with init_empty_weights():
        model = LlavaNextForConditionalGeneration(config)
    
    # if the (new) number of tokens is larger than vocab_size, resize the model
    # load original state dict
    state_dict = load_original_state_dict(model_id, model_path)
    state_dict = convert_state_dict_to_hf(state_dict)
    model.load_state_dict(state_dict, assign=True)

    model.eval()

    if config.text_config.vocab_size < len(tokenizer):
        print(f"Resizing model from {config.text_config.vocab_size} to {len(tokenizer)} since model did not have extra embeddings...")
        #pad_shape = 64
        #model.resize_token_embeddings(tokenizer.vocab_size, pad_to_multiple_of=pad_shape)
        model.resize_token_embeddings(len(tokenizer))
        # update config with new vocab size (counting padded embeddings)
        config.text_config.vocab_size = len(tokenizer)

    
    # WARNING: hardcoded fixes to the model -> CHECK THIS VALUES
    print(tokenizer.bos_token_id,tokenizer.eos_token_id)
    import pdb; pdb.set_trace()

    model.generation_config.bos_token_id = 2 # check tokenizer.bos_token_id
    model.generation_config.eos_token_id = 107 # check tokenizer.eos_token_id
    tokenizer.eos_token = "<end_of_turn>"
    print(f"Saving model and processor for {model_id} to {pytorch_dump_folder_path}")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    processor.save_pretrained(pytorch_dump_folder_path)
   

def hf_assessment(model_id, folder_path):
    # load model & processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlavaNextForConditionalGeneration.from_pretrained(folder_path).to(device)
    model.eval()
    processor = LlavaNextProcessor.from_pretrained(folder_path)


    # prepare inputs
    image = load_image()
    # DEBUG: make sure image pixels are the same as the original   
    import numpy as np
    image_pixels = torch.from_numpy(np.array(image))
    original_image_pixels = torch.load("image_pixels.pt", weights_only=False)
    original_image_pixels = torch.from_numpy(original_image_pixels)
    assert torch.allclose(image_pixels, original_image_pixels)
    
    if model_id == "liuhaotian/llava-v1.6-mistral-7b":
        prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
    elif model_id in ["liuhaotian/llava-v1.6-vicuna-7b", "liuhaotian/llava-v1.6-vicuna-13b"]:
        prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWhat is shown in this image? ASSISTANT:"
    elif model_id == "liuhaotian/llava-v1.6-34b":
        prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant\n"
    elif model_id == "lmms-lab/llama3-llava-next-8b":
        prompt = "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|><|start_header_id|>user<|end_header_id|>\n\n<image>\nWhat is shown in this image?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif model_id in ["lmms-lab/llava-next-72b", "lmms-lab/llava-next-110b"]:
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|>\n<|im_start|>assistant\n"
    if model_id == "Unbabel/lnext-towerqmix-7b-dpo-siglip2-v6":
        prompt = "TODO: add prompt"
    elif model_id == "Unbabel/lnext-tower4-sugarloaf-siglip2-v6":
        prompt = "<start_of_turn>user\n<image>\nIs this person really big, or is this building just super small?\n<end_of_turn>assistant"
    elif model_id == "Unbabel/Tower-Plus-2B":
        prompt = "<start_of_turn>user\n<image>\nIs this person really big, or is this building just super small?\n<end_of_turn>assistant"
    else:
        raise ValueError(f"Model {model_id} not supported")
    
    inputs = processor(images=image, text=prompt, return_tensors="pt")

    # # verify inputs
    filepath = "sugarloaf_pixel_inputs.pt"
    original_pixel_values = torch.load(filepath, map_location="cpu", weights_only=True)
    ## assert fails, what can I print to debug?
    # print(original_pixel_values.shape)
    print(inputs.pixel_values.shape)
    # print(inputs.pixel_values[0, 0, :10])
    # print(original_pixel_values[0, 0, :10])
    # # # print overall difference
    # #print(torch.mean(original_pixel_values.half() - inputs.pixel_values.half()))
    # # # print difference per tile (sum all other dimensions)
    # #print(torch.mean(original_pixel_values.half() - inputs.pixel_values.half(), dim=(0, 2, 3, 4)))
    # assert torch.allclose(original_pixel_values.half(), inputs.pixel_values.half())
    # filepath = hf_hub_download(repo_id="nielsr/test-image", filename="llava_1_6_pixel_values.pt", repo_type="dataset")
    # original_pixel_values = torch.load(filepath, map_location="cpu", weights_only=True)
    # assert torch.allclose(original_pixel_values, inputs.pixel_values.half())

    if model_id == "liuhaotian/llava-v1.6-mistral-7b":
        filepath = hf_hub_download(repo_id="nielsr/test-image", filename="llava_1_6_input_ids.pt", repo_type="dataset")
        original_input_ids = torch.load(filepath, map_location="cpu", weights_only=True)
        # replace -200 by image_token_id (since we use token ID = 32000 for the image token)
        original_input_ids[original_input_ids == -200] = image_token_id
        assert original_input_ids[0].tolist() == inputs.input_ids[0].tolist()

    elif model_id == "liuhaotian/llava-v1.6-34b":
        filepath = hf_hub_download(
            repo_id="nielsr/test-image", filename="llava_1_6_34b_input_ids.pt", repo_type="dataset"
        )
        original_input_ids = torch.load(filepath, map_location="cpu", weights_only=True)
        # replace -200 by image_token_id
        original_input_ids[original_input_ids == -200] = image_token_id

        assert original_input_ids[0].tolist() == inputs.input_ids[0].tolist()

    # image_sizes = torch.tensor([[899, 1024]])
    # assert image_sizes[0].tolist() == inputs.image_sizes[0].tolist()

    # verify single forward pass
    print("Single forward pass")
    with torch.inference_mode():
        inputs = inputs.to(device)
        outputs = model(**inputs)
        print("Shape of inputs:", inputs.pixel_values.shape)
        print("Shape of logits:", outputs.logits.shape)
        print("First values of logits:", outputs.logits[0, :3, :3])

       
    # verify generation
    output_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        use_cache=True,
    )

    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    print("Generated text:", repr(generated_text))


    # verify batched generation
    print("Batched generation...")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    cats_image = Image.open(requests.get(url, stream=True).raw)
    # resize the cats_image to be the same size as the image
    cats_image = cats_image.resize(image.size)

    inputs = processor(
        images=[image, cats_image],
        text=[prompt, prompt],
        padding=True,
        return_tensors="pt",
    ).to(device)

    for k, v in inputs.items():
        print(k, v.shape)

    print("Image sizes:", inputs.image_sizes)

    # make sure image_sizes are the same
    # as otherwise batched generation doesn't work
    inputs.image_sizes[1] = inputs.image_sizes[0]

    print("Batched generation...")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        use_cache=True,
    )

    outputs = processor.batch_decode(output_ids, skip_special_tokens=True)
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        help="Hub location of the model to convert",
        default="Unbabel/TowerVision-Plus-2B",
        choices=[
            "liuhaotian/llava-v1.6-mistral-7b",
            "liuhaotian/llava-v1.6-vicuna-7b",
            "liuhaotian/llava-v1.6-vicuna-13b",
            "liuhaotian/llava-v1.6-34b",
            "lmms-lab/llama3-llava-next-8b",
            "lmms-lab/llava-next-72b",
            "lmms-lab/llava-next-110b",
            "Unbabel/lnext-towerqmix-7b-dpo-siglip2-v6",
            "Unbabel/TowerVision-Plus-9B",
            "Unbabel/TowerVision-Plus-2B",
            "Unbabel/TowerVision-4-Anthill-CPT",
        ],
        required=False,
    )
    parser.add_argument(
        "--local_model_registry",
        type=str,
        #default="/gpfs/scratch/ehpc209/retrain_tmp/vlm_ckpts/finetune/towerp_2b_instruct/",
        help="Path to a local folder containing model files to use as a registry instead of downloading from the hub.",
        required=False,
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        type=str,
        #default="/mnt/scratch-artemis/gviveiros/TowerVision/towerp_2b_instruct/",
        help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    args = parser.parse_args()

    convert_llava_to_hf(args.model_id, args.local_model_registry, args.pytorch_dump_folder_path, args.push_to_hub)
    # hf_assessment(args.model_id, args.pytorch_dump_folder_path)
    

    if args.push_to_hub:
        model = LlavaNextForConditionalGeneration.from_pretrained(args.pytorch_dump_folder_path)
        processor = LlavaNextProcessor.from_pretrained(args.pytorch_dump_folder_path)
        checkpoint_name = args.model_id.split("/")[-1]
        print(f"Pushing to repo utter-project/{checkpoint_name}")
        model.push_to_hub(f"utter-project/{checkpoint_name}")
        processor.push_to_hub(f"utter-project/{checkpoint_name}")

    

    