# Cell 1: Imports
from operator import imod
import torch
from PIL import Image
import json
import argparse
from tqdm import tqdm
import base64
from io import BytesIO
import copy
from pathlib import Path
import datasets
from llava.constants import tower_language_support

# Import LLaVA components
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


# Cell 2: Function to load the model
def load_model(model_path, token=None, device="cuda"):
    """
    Load the LLaVA model, tokenizer, and image processor
    
    Args:
        model_path: HuggingFace model path or local path
    
    Returns:
        tokenizer, model, image_processor, device
    """
    # Set device
    #device = "cuda"
    assert device == "cuda", f"Device must be cuda, got {device}"
    device_map = f"{device}:0"

    #device_map = f"{device}:0"
    
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
    model.to(device)
    model.eval()
    
    return tokenizer, model, image_processor, device

# Cell 4: Define the inference function
def run_inference(
    image, 
    prompt,
    tokenizer,
    model,
    image_processor,
    device="cuda",
    conv_template="gemma2_instruct",
    max_new_tokens=512,
    temperature=0.,
    log_prompt=False,
    skip_special_tokens:bool=True
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

    # DEBUG: dump image file pixel values (before any processing)
    #import numpy as np
    #image_pixels = np.array(image)
    #torch.save(image_pixels, "image_pixels.pt")

    processed_image = process_images([image], image_processor, model.config)

    #if torch.cuda.is_available():
    processed_image = processed_image.to(dtype=torch.bfloat16, device=device)
    
    # Prepare conversation
    if DEFAULT_IMAGE_TOKEN not in prompt:
       print(f"Adding {DEFAULT_IMAGE_TOKEN} to prompt")
       prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    
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
    input_ids_list = [tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(device)]
    print(input_ids_list[0])
    
    # Generate response
    pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(device)
    attention_masks = input_ids.ne(pad_token_ids).to(device)
    
    attention_masks = attention_masks.to(device)
    processed_image = processed_image.to(device)
    input_ids = input_ids.to(device)
    model = model.to(device)
    #image_size = image_size.to(device)
    #pad_token_ids = pad_token_ids.to(device)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            pad_token_id=pad_token_ids,
            eos_token_id=107,
            images=processed_image,
            image_sizes=[image_size],
            do_sample=temperature > 0,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    
    # Decode output
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=skip_special_tokens)[0]
    response = output.split(conv.roles[1] + ": ")[-1].strip()
    
    return response



# Define Likelihood function
def calculate_likelihood(
    prompt,
    target,
    image, 
    tokenizer,
    model,
    image_processor,
    device,
    conv_template="gemma2_instruct",
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
    if image is not None:
        image_size = [image.size[0], image.size[1]]
        processed_image = process_images([image], image_processor, model.config)
        processed_image = processed_image.to(dtype=torch.bfloat16, device=device)
    else:
        processed_image = None
        image_size = []

    
    # Prepare prompt
    if image is not None and DEFAULT_IMAGE_TOKEN not in prompt:
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
    perplexity = torch.exp(loss)
    return float(perplexity.item())


class DataLoader:
    def __init__(self):

        self._dataset_mappings = {
            "alm_bench-all": "sonalsannigrahi/alm-bench-lang-split",
            "m3exam": "neulab/PangeaBench-m3exam",
            "ocrbench": "echo840/OCRBench",
            "cc-ocr-multi-lan": "wulipc/CC-OCR",
        }
       
        #self.ds = datasets.load_dataset(self.benchmark_hf)
    
    def load_dataset(self, dataset_name):
        # fetch dataset from the mapping
        try:
            self.benchmark_name = dataset_name
            dataset_hf = self._dataset_mappings[dataset_name]
        except KeyError:
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        if dataset_name == "cc-ocr-multi-lan":
            dataset = datasets.load_dataset(dataset_hf, "multi_lan_ocr")["test"]
            data = []
            for split in dataset.keys():
                if tower_language_support(split):
                    data.extend(dataset[split])
            return data
        elif dataset_name == "m3exam":
            data = []
            dataset = datasets.load_dataset(dataset_hf)
            for split in dataset.keys():
                if tower_language_support(split):
                    data.extend(dataset[split])
            return data
        elif dataset_name == "alm_bench-all":
            dataset = datasets.load_dataset(dataset_hf)
            data = []
            for split in dataset.keys():
                if tower_language_support(split):
                    data.extend(dataset[split])
                else:
                    print("Tower language not supported: ", split, "for dataset ", dataset_name, "skipping...")
            return data
        elif dataset_name == "ocrbench":
            return datasets.load_dataset(dataset_hf)["test"]
        else:
            import pdb; pdb.set_trace()
            raise ValueError(f"Dataset {dataset_name} not supported")

    def get_language(self, sample):
        if self.benchmark_name == "cc-ocr-multi-lan":
            return sample["split"]
        elif self.benchmark_name == "m3exam":
            return sample["language"]
        elif self.benchmark_name == "alm_bench-all":
            return sample["Language"]
        elif self.benchmark_name == "ocrbench":
            return "english"
        else:
            raise ValueError(f"Dataset {self.benchmark_name} not supported")

    def get_prompt_image_target(self, sample):
        if self.benchmark_name == "cc-ocr-multi-lan":
            # if img is a str, download it from the url
            if isinstance(sample["image"], str):
                sample["image"] = Image.open(BytesIO(base64.b64decode(sample["image"])))
            return sample["question"], sample["image"], sample["answer"]
        elif self.benchmark_name == "alm_bench-all":
            return sample["Translated_Question"], sample["file_name"], sample["Translated_Answer"]
        elif self.benchmark_name == "ocrbench":
            return sample["question"], sample["image"], sample["answer"]
        elif self.benchmark_name == "m3exam":
            images_urls = [v for k, v in sorted(sample.items()) if k.startswith('image_') and v]
            images = [Image.open(BytesIO(base64.b64decode(img_url))) for img_url in images_urls if img_url != 'None']
            if len(images) > 1:
                image = None
            else:
                image = images[0]
            return sample["question_text"], image, sample["answer_text"]
        else:
            raise ValueError(f"Dataset {self.benchmark_name} not supported")

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/mnt/scratch-artemis/gviveiros/TowerVision/llava-next-native/")
    parser.add_argument("--benchmark_name", type=str, required=True, 
            choices=["alm_bench-all", "m3exam", 
                    "ocrbench", "cc-ocr-multi-lan", 
                    ],
            help="Name of the benchmark dataset to use")
    parser.add_argument("--model_name", type=str, choices=["towerp_2b_instruct_full", "towerp_2b_base_full"], required=True)
    parser.add_argument("--results_path", type=str, default="results")
    #default="towerp_2b_instruct_full")
    args = parser.parse_args()
    print(f"Arguments: {args}")
    # loader
    dataloader = DataLoader()
    # load model
    tokenizer, model, image_processor, device = load_model(args.model_path + args.model_name)
    assert model.device == torch.device("cuda:0"), f"Model not loaded on cuda, got {model.device}"

    # load data
    ds = dataloader.load_dataset(args.benchmark_name)
    # pass
    samples_ll = {
        "likelihood": [],
        "control_likelihood": []
    }
    for idx, sample in tqdm(enumerate(ds), total=len(ds)):
        # get sample language
        prompt, image, target = dataloader.get_prompt_image_target(sample)
        if image is None:
            continue
        
        # run inference
        # response = run_inference(image, prompt, tokenizer, model, image_processor, device, log_prompt=True)
        # calculate likelihood
        correct_likelihood = calculate_likelihood(
            prompt, target, image=image, 
            tokenizer=tokenizer, model=model, image_processor=image_processor, device=device, log_prompt=True)
        correct_control = calculate_likelihood(
            prompt, target, image=None, 
            tokenizer=tokenizer, model=model, image_processor=image_processor, device=device)
        
        #print(f"Greedy Response: {response}")
        print(f"Correct Likelihood: {correct_likelihood}")
        print(f"Correct Control: {correct_control}")
        
        samples_ll["likelihood"].append(correct_likelihood)
        samples_ll["control_likelihood"].append(correct_control)
    # save results (ignore per language), treat all languages as the same
    # create subfolder for the benchmark name
    samples_path = Path(args.results_path).joinpath(args.benchmark_name)
    samples_path.mkdir(parents=True, exist_ok=True)
    with open(f"{samples_path}/cxmi_{args.model_name}.json", "w") as f:
        json.dump(samples_ll, f)
    
    # "alm_bench-all", "m3exam", "ocrbench", "cc-ocr-multi-lan"
    # python -m scripts.cxmi_llnext --model_path /mnt/scratch-artemis/gviveiros/TowerVision/llava-next-native/ --benchmark_name alm_bench-all --model_name towerp_2b_instruct_full