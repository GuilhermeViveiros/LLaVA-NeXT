# Cell 1: Imports
import torch
from PIL import Image
import copy
from pathlib import Path
import datasets
import requests
import numpy as np
# Import LLaVA components
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


# Cell 2: Function to load the model
def load_model(model_path, token=None):
    """
    Load the LLaVA model, tokenizer, and image processor
    
    Args:
        model_path: HuggingFace model path or local path
    
    Returns:
        tokenizer, model, image_processor, device
    """
    # Set device
    device = "cuda"
    device_map = f"{device}:0"
    
    # Load model
    model_name = get_model_name_from_path(model_path)
    print("Model name: ", model_name)
    llava_args = {
        "multimodal": True,
        "attn_implementation": "sdpa" if torch.version.cuda and torch.__version__ >= "2.1.2" else "eager"
    }
    
    print("Loading model... model_path: ", model_path, "model_name: ", model_name)
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path, None, model_name, torch_dtype="bfloat16", device_map=device_map, **llava_args
    )
    model.eval()
    
    return tokenizer, model, image_processor, device


# Cell 4: Define the inference function
def run_inference(
    image, 
    prompt,
    tokenizer,
    model,
    image_processor,
    device,
    conv_template="qwen_2",
    max_new_tokens=512,
    temperature=0.,
    log_prompt=False
):
    """
    Run inference with a pre-loaded LLaVA model
    
    Args:
        image_path: Path to the image file
        prompt: Text prompt to send to the model
        tokenizer, model, image_processor: Pre-loaded model components
        device: The device to run inference on
        conv_template: Conversation template to use
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0 for deterministic)
    
    Returns:
        The model's response
    """
    # Load and process image
    #image = Image.open(image_path).convert('RGB')
    image_size = [image.size[0], image.size[1]]
    # process image
    processed_image = process_images([image], image_processor, model.config)

    # DEBUG: dump processed_image to file .pt
    torch.save(processed_image, "processed_image.pt")
    if torch.cuda.is_available():
        processed_image = processed_image.to(dtype=torch.bfloat16, device=device)
    
    # Prepare conversation
    # if DEFAULT_IMAGE_TOKEN not in prompt:
    #    print(f"Adding {DEFAULT_IMAGE_TOKEN} to prompt")
    #    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    
    # This is safer for llama3 models
    if "llama_3" in conv_template:
        conv = copy.deepcopy(conv_templates[conv_template])
    else:
        conv = conv_templates[conv_template].copy()
    
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    
    def pad_sequence(input_ids, batch_first, padding_value):
        if tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    if log_prompt:
        print(prompt_text)
    
    
    # Tokenize input
    input_ids_list = [tobkenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(device)]
    print(input_ids_list[0])
    
    # Generate response
    pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(device)
    attention_masks = input_ids.ne(pad_token_ids).to(device)
    
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            pad_token_id=pad_token_ids,
            eos_token_id=tokenizer.eos_token_id,
            images=processed_image,
            image_sizes=[image_size],
            do_sample=temperature > 0,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    
    # Decode output
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    response = output.split(conv.roles[1] + ": ")[-1].strip()
    
    return response

# Define Likelihood function
def calculate_likelihood(
    image, 
    prompt,
    target,
    tokenizer,
    model,
    image_processor,
    device,
    conv_template="qwen_2",
    log_prompt=False
):
    """
    Calculate the likelihood of a given translation for an image-text pair
    
    Args:
        image: PIL image
        source_text: Source text to translate
        target_text: Target translation to evaluate
        tokenizer, model, image_processor: Pre-loaded model components
        device: The device to run inference on
        conv_template: Conversation template to use
    
    Returns:
        tuple: (loss value, boolean indicating if greedy prediction matches target exactly)
    """
    # Process image
    image_size = [image.size[0], image.size[1]]
    processed_image = process_images([image], image_processor, model.config)
    
    if torch.cuda.is_available():
        processed_image = processed_image.to(dtype=torch.float16, device=device)
    
    # Prepare prompt
    if DEFAULT_IMAGE_TOKEN not in prompt:
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    
    # Prepare conversation
    if "llama_3" in conv_template:
        conv = copy.deepcopy(conv_templates[conv_template])
    else:
        conv = conv_templates[conv_template].copy()
    
    # First create context-only conversation (up to model's turn)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    context_prompt = conv.get_prompt()

    if log_prompt:
        print(context_prompt)
    
    # Tokenize context
    context_ids = tokenizer_image_token(context_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(device)
    
    # Now create full conversation with target answer
    conv.messages[-1][1] = target  # Set the model's response to the target text
    full_prompt = conv.get_prompt()
    if log_prompt:
        print(full_prompt)
    
    # Tokenize full prompt
    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    print(input_ids)
    
    # Create labels with -100 for context part (we don't compute loss for it)
    labels = input_ids.clone()
    labels[0, :context_ids.shape[0]] = -100
    
    # Calculate loss
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids, 
            labels=labels, 
            images=processed_image, 
            use_cache=True, 
            image_sizes=[image_size]
        )
    
    loss = outputs["loss"]
    
    # Check if greedy prediction matches target
    logits = outputs["logits"]
    greedy_tokens = logits.argmax(dim=-1)
    target_tokens = input_ids[:, context_ids.shape[0]:]  # Skip the context tokens
    greedy_tokens = greedy_tokens[:, context_ids.shape[0]:input_ids.shape[1]]  # Use only relevant predictions
    max_equal = (greedy_tokens == target_tokens).all()
    
    return float(loss.item())


if __name__ == "__main__":
    model_path = "/mnt/scratch-artemis/gviveiros/TowerVision/llava-next-native/towerp_2b_instruct/"
    tokenizer, model, image_processor, device = load_model(model_path)
    print(f"Model loaded on {device}")

    # # Example 1
    # url = "https://cms.mistral.ai/assets/a10b924e-56b3-4359-bf6c-571107811c8f"
    # image = Image.open(requests.get(url, stream=True).raw)
    # prompt = """<image>\nIs this person really big, or is this building just super small?"""

    # response = run_inference(
    #     image, 
    #     prompt, 
    #     tokenizer, 
    #     model, 
    #     image_processor, 
    #     device,
    #     conv_template="gemma2_instruct",
    #     log_prompt=True
    # )
    # print(f"Model Greedy Response: {response}")

    # Example 2
    # load dataset
    ds = datasets.load_dataset("Unbabel/commute_multimodal_mt", split="de")
    sample = ds[0]
    print(sample)
    #display(sample["image"])
    #tokenizer.eos_token = "<|im_end|>"
    prompt = f"What's in this image?"
    print(f"Query: {prompt}")
    response = run_inference(
        sample["image"],
        prompt, tokenizer, model, image_processor, device, conv_template="gemma2_instruct", log_prompt=True)
    print(f"Model Greedy Response: {response}")
    print("-"*10)
    prompt = f"Translate from English to Portuguese:\n{sample['source']}"
    print(f"Query: {prompt}")
    response = run_inference(sample["image"], prompt, tokenizer, model, image_processor, device, conv_template="gemma2_instruct", log_prompt=True)
    print(f"Model Greedy Response: {response}")