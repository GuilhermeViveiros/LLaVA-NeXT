# Cell 1: Imports
import imp
import numpy as np
from operator import imod
import os
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

# Import LLaVA components
try:
    from llava.constants import tower_language_support
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
except:
    print("LLaVA not found, carrying on without it")


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
    return float(loss.item())


class DataLoader:
    def __init__(self):

        self._dataset_mappings = {
            "alm_bench-all": "sonalsannigrahi/alm-bench-lang-split",
            "m3exam": "neulab/PangeaBench-m3exam",
            "ocrbench": "echo840/OCRBench",
            "cc-ocr-multi-lan": "wulipc/CC-OCR",
            "commute-all-contrastive": "Unbabel/commute_multimodal_mt",
            "blink": "BLINK-Benchmark/BLINK",
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
            # filter languages not supported by TowerVision
            return dataset.filter(lambda x: tower_language_support(x["l2-category"]))
        elif dataset_name == "commute-all-contrastive":
            langs = ["fr", "de", "cs"]
            dataset = []
            for idx, lang in enumerate(langs):
                ds = datasets.load_dataset(dataset_hf, split=lang)
                # add language to the dataset
                if lang == "cs":
                    lang = "Czech"
                elif lang == "de":
                    lang = "German"
                elif lang == "fr":
                    lang = "French"
                ds = ds.map(lambda x: {"split": lang})
                dataset.extend(ds)
            return dataset
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
        elif dataset_name == "blink":
            splits = ["Art_Style", "Counting", "Forensic_Detection", "Functional_Correspondence", "IQ_Test", "Jigsaw", "Multi-view_Reasoning", "Object_Localization", "Relative_Depth", "Relative_Reflectance", "Semantic_Correspondence", "Spatial_Relation", "Visual_Correspondence", "Visual_Similarity"]
            data = []
            for split in splits:
                data.extend(datasets.load_dataset(dataset_hf, split)["val"])
            return data
        else:
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
        elif self.benchmark_name == "commute-all-contrastive":
            return sample["split"]
        else:
            raise ValueError(f"Dataset {self.benchmark_name} not supported")

    def get_idx_task_prompt_image_target(self, sample) -> tuple[str, str, str, list[Image.Image], str]:
        """
        Get the prompt, image, and target for a given sample
        Ignore samples with more than one image
        """
        if self.benchmark_name == "cc-ocr-multi-lan":
            # if img is a str, download it from the url
            if isinstance(sample["image"], str):
                sample["image"] = Image.open(BytesIO(base64.b64decode(sample["image"])))
            return sample["question"], sample["image"], sample["answer"]
        elif self.benchmark_name == "alm_bench-all":
            return sample["Translated_Question"], sample["file_name"], sample["Translated_Answer"]
        elif self.benchmark_name == "ocrbench":
            answer = sample["answer"]
            if isinstance(answer, list):
                print("Answer is a list, taking the first element", answer , "to -> ", answer[0])
                answer = answer[0]
            return sample["question"], sample["image"], answer
        elif self.benchmark_name == "m3exam":
            images_urls = [v for k, v in sorted(sample.items()) if k.startswith('image_') and v]
            images = [Image.open(BytesIO(base64.b64decode(img_url))) for img_url in images_urls if img_url != 'None']
            if len(images) > 1:
                image = None
            else:
                image = images[0]
            return sample["question_text"], image, sample["answer_text"]
        elif self.benchmark_name == "commute-all-contrastive":
            prompt = f"Translate from English to {sample['split']}:\n{sample['source']}"
            return sample.get("idx", None), sample["split"], prompt, sample["image"], sample["correct_translation"]
        
        elif self.benchmark_name == "blink":
            images = [sample[k] for k in sample.keys() if k.startswith("image_")]
            images = [img for img in images if img is not None]
            if len(images) > 1:
                image = None
            else:
                image = images[0]
            return sample["idx"], sample["sub_task"], sample["prompt"], image, "Answer: " + sample["answer"]
        else:
            raise ValueError(f"Dataset {self.benchmark_name} not supported")

def compute_cxmi(model, tokenizer, image_processor, device, benchmark_name, results_path):
    # load data
    ds = dataloader.load_dataset(benchmark_name)
    # pass
    samples_ll = {
        "idx": [],
        "likelihood": [],
        "control_likelihood": [],
        "task": []
    }
    for idx_, sample in tqdm(enumerate(ds), total=len(ds)):
        # get sample language
        idx, sub_task, prompt, image, target = dataloader.get_idx_task_prompt_image_target(sample)
        if idx is None:
            idx = idx_
        
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
        samples_ll["task"].append(sub_task)
        samples_ll["idx"].append(idx)
    # save results (ignore per language), treat all languages as the same
    # create subfolder for the benchmark name
    samples_path = Path(results_path).joinpath(benchmark_name)
    samples_path.mkdir(parents=True, exist_ok=True)
    with open(f"{samples_path}/cxmi_{args.model_name}.json", "w") as f:
        json.dump(samples_ll, f)

def func_compute_cxmi(likelihoods, control_likelihoods):
    # likelihoods and control_likelihoods are negative average log likelihoods
    # lets ignore values under 1e-4
    likelihoods = [l for l in likelihoods if l > 1e-3]
    control_likelihoods = [l for l in control_likelihoods if l > 1e-3]
    likelihoods = [np.exp(-l + 1e-10) for l in likelihoods] # average likelihood
    control_likelihoods = [np.exp(-l + 1e-10) for l in control_likelihoods] # average likelihood
    #import pdb; pdb.set_trace()
    cxmis = [l_c/l for l, l_c in zip(likelihoods, control_likelihoods)]
    return np.nanmean(cxmis)

def func_compute_accuracy(likelihoods, control_likelihoods):
    accuracies = [l < l_c for l, l_c in zip(likelihoods, control_likelihoods)]
    return np.nanmean(accuracies)

#print(f"CXMI (Correct): {compute_cxmi(correct_likelihoods, correct_control_likelihoods)}")
#print(f"CXMI (Incorrect): {compute_cxmi(incorrect_likelihoods, incorrect_control_likelihoods)}")


def analyse_results(results_path):
    # iterate over the results path
    # - benchmark name
    # - - - cxmi_{model_name_1}.json
    # - - - cxmi_{model_name_2}.json
    model_results = {}
    for benchmark_name in os.listdir(results_path):
        model_results[benchmark_name] = {}
        
        for model_name in os.listdir(os.path.join(results_path, benchmark_name)):
            with open(os.path.join(results_path, benchmark_name, model_name), "r") as f:
                data = json.load(f)
            model_name = model_name.replace("cxmi_", "").replace(".json", "")
            model_results[benchmark_name][model_name] = data

    # compute cxmi for each model
    for benchmark_name, models in model_results.items():
        print("Benchmark: ", benchmark_name)
       
        for model_name, data in models.items():
            # some benchmarks can be divided by tasks
            # if task on data, present the results per task
            if "task" in data:
                # aggregate results per task
                task_results = {}
                for task, likelihoods, control_likelihoods in zip(data["task"], data["likelihood"], data["control_likelihood"]):
                    if task not in task_results:
                        task_results[task] = {"likelihood": [], "control_likelihood": []}
                    task_results[task]["likelihood"].append(likelihoods)
                    task_results[task]["control_likelihood"].append(control_likelihoods)
                
                #  calculate cxmi and accuracy per task
                for task, vals in task_results.items():
                    likelihoods = vals["likelihood"]
                    control_likelihoods = vals["control_likelihood"]
                    cxmi = func_compute_cxmi(likelihoods, control_likelihoods)
                    accuracy = func_compute_accuracy(likelihoods, control_likelihoods)
                    print("  Model: ", model_name.replace("towerp_", ""))
                    print("    Task: ", task)
                    print(f"     CXMI: {round(cxmi, 3)}, Accuracy: {round(accuracy, 3)}")
            else:
                cxmi = func_compute_cxmi(data["likelihood"], data["control_likelihood"])
                accuracy = func_compute_accuracy(data["likelihood"], data["control_likelihood"])
                print("  Model: ", model_name.replace("towerp_", ""))
                print(f"    CXMI: {round(cxmi, 3)}, Accuracy: {round(accuracy, 3)}")
        print("--------------------------------")
            #print(f"Accuracy ({model_name}), Benchmark: {benchmark_name}: {accuracy}")

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/mnt/scratch-artemis/gviveiros/TowerVision/llava-next-native/")
    parser.add_argument("--benchmark", type=str, required=False, 
            choices=["alm_bench-all", "m3exam", 
                    "ocrbench", "cc-ocr-multi-lan", 
                    "commute-all-contrastive", "blink", "all"
                    ],
            help="Name of the benchmark dataset to use")
    parser.add_argument("--model_name", type=str, choices=["towerp_2b_instruct_full", "towerp_2b_base_full", "towerp_9b_base_full", "towerp_9b_instruct_full"], required=False, default="towerp_9b_instruct_full")
    parser.add_argument("--results_path", type=str, default="results")
    
    #default="towerp_2b_instruct_full")
    args = parser.parse_args()
    print(f"Arguments: {args}")
    
    # loader
    dataloader = DataLoader()
    # compute cxmi (if benchmark_name and model_name are provided)
    if args.benchmark:
        # load model
        tokenizer, model, image_processor, device = load_model(args.model_path + args.model_name)
        assert model.device == torch.device("cuda:0"), f"Model not loaded on cuda, got {model.device}"
        if args.model_name is None:
            raise ValueError("Model name is required when benchmark name is provided")
        if args.benchmark == "all":
            benchmarks = ["commute-all-contrastive", "blink"]
            for benchmark in benchmarks:
                compute_cxmi(model, tokenizer, image_processor, device, benchmark, args.results_path)
        else:
            compute_cxmi(model, tokenizer, image_processor, device, args.benchmark, args.results_path)
    
    # analyse results
    analyse_results(args.results_path)
    
    # to process) "alm_bench-all", "m3exam", "ocrbench", "cc-ocr-multi-lan"
    # python -m scripts.cxmi_llnext --model_path /mnt/scratch-artemis/gviveiros/TowerVision/llava-next-native/ --benchmark_name alm_bench-all --model_name towerp_2b_instruct_full