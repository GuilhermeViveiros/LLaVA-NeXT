#!/usr/bin/env python3
"""
Script to convert Hugging Face datasets into the format used by LLaVA-NeXT.
This script handles both single-image and multi-image datasets.
"""

import os
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face datasets to LLaVA-NeXT format"
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, 
        help="Name of the Hugging Face dataset to convert"
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--subset", type=str, default=None,
        help="Dataset subset name (if applicable)"
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="Path to save the converted dataset"
    )
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Directory where images are stored (if not using HF dataset's images)"
    )
    parser.add_argument(
        "--image_field", type=str, default="image",
        help="Field name containing image path or image data in the dataset"
    )
    parser.add_argument(
        "--prompt_template", type=str, default="<image>",
        help="Template for the prompt. Use {field_name} to insert dataset fields. Example: '<image>\\nDescribe this {object_type}.'"
    )
    parser.add_argument(
        "--target_field", type=str, default="correct_answer",
        help="Field name containing the answer in the dataset"
    )
    parser.add_argument(
        "--id_field", type=str, default="id",
        help="Field name containing the unique ID in the dataset"
    )
    parser.add_argument(
        "--add_image_tag", action="store_true",
        help="Add <image> tag before the question",
    )
    parser.add_argument(
        "--system_prompt", type=str, default=None,
        help="Optional system prompt to add to the conversation"
    )
    return parser.parse_args()


def apply_template(template: str, example: Dict[str, Any]) -> str:
    """Apply a template string using fields from the example."""
    # Find all field placeholders in the template
    placeholders = re.findall(r'\{([^}]+)\}', template)
    
    # Warn if there are placeholders that are not in the example
    for placeholder in placeholders:
        if placeholder not in example:
            print(f"WARNING: Placeholder '{placeholder}' not found in example")
    
    # Replace each placeholder with its value from the example
    result = template
    for field in placeholders:
        if field in example:
            result = result.replace(f"{{{field}}}", str(example[field]))
        else:
            # Keep the placeholder if field doesn't exist
            print(f"WARNING: Field '{field}' not found in example, prompt will be kept templated")
    
    return result

def format_conversation(prompt: str, answer: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    """Format a prompt-answer pair into the conversation format used by LLaVA-NeXT."""
    conv = [{"from": "system", "value": system_prompt}] if system_prompt else []
    conv.append({"from": "human", "value": prompt})
    conv.append({"from": "gpt", "value": answer})
    return conv

def get_image_path(example: Dict[str, Any], id: str, image_field: str, image_dir: str) -> Any:
    """Extract image path from the example, handling various formats."""
    if not hasattr(get_image_path, "_warning_shown"):
        get_image_path._warning_shown = False

    if image_field not in example:
        raise ValueError(f"Image field '{image_field}' not found in example")
    
    image_data = example[image_field]
    
    # If the image field contains actual image data (like PIL Image or bytes)
    # we need to save it and return the path
    if not isinstance(image_data, str):
        # check if args.image_dir is empty if it exists, otherwise create_dir
        if not get_image_path._warning_shown and os.path.exists(image_dir) and os.path.isdir(image_dir) and len(os.listdir(image_dir)) > 0:
            print(f"WARNING: Image directory {image_dir} already exists but not empty, careful.")
            get_image_path._warning_shown = True
        else:
            os.makedirs(image_dir, exist_ok=True)
        
        # For HuggingFace datasets that return PIL Images or other image objects
        # we need to save the image and return the path
        if hasattr(image_data, 'save'):  # PIL Image
            if not args.image_dir:
                raise ValueError("image_dir must be provided when dataset contains PIL Images")
            
            # Create a filename based on the example ID or index
            filename = f"{id}.jpg"
            save_path = os.path.join(image_dir, filename)
            
            # Save the image
            image_data.save(save_path)
            return save_path
        else:
            raise ValueError(f"Unsupported image data type: {type(image_data)}. Please provide image paths instead.")
    else:
        # check if the image_data is a valid image path, assume it is a relative path with the image_dir
        image_path = image_data if os.path.isabs(image_data) else os.path.join(image_dir, image_data)

        # check if the image_data is a valid image path
        # TODO: check if the image_data is a valid image file
        #if not os.path.exists(image_path):
        #    raise ValueError(f"Image path {image_path} does not exist")
    
    # Otherwise, use the image path directly from the dataset
    return image_path

def load_culturalground_dataset(args):
    """Load the CulturalGround dataset from Hugging Face."""
    # Construct the full local path to the dataset directory in your cache.
    local_path = os.path.expanduser("~/.cache/huggingface/hub/datasets--neulab--CulturalGround/snapshots/7e1c549862550a7b4698b734385449fae6c02ecd/")
    # Load the dataset using the local path and specify the data files.
    dataset = load_dataset(
        "json",  # Use "json" as the builder name
        data_files={
            "train": [
                os.path.join(local_path, "CulturalGround-MCQs-Filtered-7M.json"),
                #os.path.join(local_path, "CulturalGround-OE-Filtered-14M.json"),
                #os.path.join(local_path, "CulturalGround-MCQ-Unfiltered-8M.json"),
                #os.path.join(local_path, "CulturalGround-OE-Unfiltered-22M.json"),
            ]
        }
    )
    return dataset["train"]

def convert_dataset(args):
    """Convert a Hugging Face dataset to LLaVA-NeXT format."""
    print(f"Loading dataset: {args.dataset_name}")
    
    # Load the dataset from Hugging Face
    # if args.subset:
    #     dataset = load_dataset(args.dataset_name, args.subset, split=args.split)
    # else:
    #     import pdb; pdb.set_trace()
    #     dataset = load_dataset(args.dataset_name, split=args.split)
    dataset = load_culturalground_dataset(args)
    print(f"Loaded {len(dataset)} examples")
    
    # Convert the dataset to LLaVA-NeXT format
    converted_data = []
    
    for idx, example in tqdm(enumerate(dataset), total=len(dataset), desc="Converting dataset"):
        # Extract fields based on the provided arguments
        item_id = str(example.get(args.id_field, f"{idx:09d}"))
        image_path = get_image_path(example, item_id, args.image_field, args.image_dir)
        #prompt = apply_template(args.prompt_template, example)
        answer = example[args.target_field]
        import pdb; pdb.set_trace()
        # Create the converted item
        # FIXME: we should add the source language of the samples...
        converted_item = {
            "id": item_id,
            "image": image_path,
            "data_source": args.dataset_name
        }
        converted_item["conversations"] = example["conversations"] 
        # format_conversation(
        #     prompt, answer, args.system_prompt
        # )
        
        converted_data.append(converted_item)
        
    # Save the converted dataset
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"Converted dataset saved to {args.output_file}")
    print(f"Total examples: {len(converted_data)}")


if __name__ == "__main__":
    args = parse_args()
    print("Args: ", args)
    convert_dataset(args)

    
    # example usage:
    # python scripts/prepare_dataset.py \
    #     --dataset_name "/mnt/home/gviveiros/.cache/huggingface/hub/datasets--neulab--CulturalGround/snapshots/7e1c549862550a7b4698b734385449fae6c02ecd/" \
    #     --split "/mnt/home/gviveiros/.cache/huggingface/hub/datasets--neulab--CulturalGround/snapshots/7e1c549862550a7b4698b734385449fae6c02ecd/CulturalGround-MCQs-Filtered-7M.json" \
    #     --output_file "data/llava-next-culturalground-mcqs-filtered-7m.json" \
    #     --image_dir "/mnt/home/gviveiros/.cache/huggingface/hub/datasets--neulab--CulturalGround/snapshots/7e1c549862550a7b4698b734385449fae6c02ecd/CultureGroundImages/"
        
