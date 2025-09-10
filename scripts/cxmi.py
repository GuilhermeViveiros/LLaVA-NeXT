# Cell 1: Imports
import torch
from PIL import Image
import json
from pathlib import Path
import datasets
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
#from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, IGNORE_INDEX
from datetime import datetime
import logging
from tqdm import tqdm

# only allow visibility and use of gpu 4
#os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# check nvidia-smi
# os.system("nvidia-smi")
#print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100

def setup_logging(log_file="cxmi_experiment.log"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_results(results, metadata, output_dir="results", filename=None):
    """Save results with metadata to JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cxmi_results_{timestamp}.json"
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    full_path = output_path / filename
    
    # Prepare data for saving
    save_data = {
        "metadata": metadata,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(full_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    
    return full_path

def load_results(file_path):
    """Load results from JSON file"""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["results"], data["metadata"]

def calculate_accuracy(results):
    """Calculate accuracy from results"""
    correct_count = 0
    total_count = 0
    
    for sample_id, sample_results in results.items():
        if "w/images" in sample_results and "w/oimages" in sample_results:
            # Get the last values (most recent results for this sample)
            lp_with = sample_results["w/images"][-1] if sample_results["w/images"] else float('inf')
            lp_noimg = sample_results["w/oimages"][-1] if sample_results["w/oimages"] else float('inf')
            
            # Lower loss means better prediction, so image helps if lp_with < lp_noimg
            if lp_with < lp_noimg:
                correct_count += 1
            total_count += 1
    
    return correct_count / total_count if total_count > 0 else 0.0

def load_model(model_name:str, dtype:str="bfloat16", device_map:str="auto"):
    """
    Load the model and processor
    """
    processor = LlavaNextProcessor.from_pretrained(model_name)
    if dtype == "bfloat16":
        dtype = torch.bfloat16
    elif dtype == "float16":
        dtype = torch.float16
    else:
        raise ValueError(f"Invalid dtype: {dtype}")
    print(f"Loading model {model_name} with dtype {dtype} and device_map {device_map}")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name, 
        device_map=device_map,
        torch_dtype=dtype,
        attn_implementation="sdpa"
    )
    device = model.device
    return processor, model, device

def prepare_prompt(query: str, processor: LlavaNextProcessor):

    # Prepare conversation
    if DEFAULT_IMAGE_TOKEN not in query:
       print(f"Adding {DEFAULT_IMAGE_TOKEN} to prompt")
       query = DEFAULT_IMAGE_TOKEN + "\n" + query
    
    conversation = [
        {
            "role": "user", 
            "content": query
        }
    ]
    
    prompt = processor.tokenizer.apply_chat_template(
        conversation, 
        tokenize=False,
        add_generation_prompt=True
    )
    # remove <bos> from prompt
    # prompt = prompt.replace("<bos>", "")
    
    return prompt

# Define Likelihood function
def calculate_likelihood(
    image, 
    prompt,
    target,
    model,
    processor,
    device,
    conv_template="gemma2_instruct",
    log_prompt:bool=True
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

    # Step 1) Prepare context prompt
    has_image = image is not None
    if DEFAULT_IMAGE_TOKEN not in prompt and has_image:
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    
    context_conv = [{"role": "user", "content": prompt}]

    context_prompt = processor.tokenizer.apply_chat_template(
        context_conv,
        tokenize=False,
        add_generation_prompt=True # adds assistant start-of-turn, but no answer
    )

    if log_prompt:
        print("Context prompt:\n", context_prompt)

    # remove bos token
    context_prompt = context_prompt.replace("<bos>", "")

    if has_image: # if image is provided, use it
        context_inputs = processor(
            images=image,
            text=context_prompt,
            return_tensors="pt",
        ).to(device)
        pixel_values = context_inputs["pixel_values"]
    else:
        context_inputs = processor(
            text=context_prompt,
            return_tensors="pt"
        ).to(device)
        pixel_values = None

    context_len = context_inputs["input_ids"].shape[1]

    # Step 2) Tokenize target directly
    target_ids = processor.tokenizer(
        target,
        return_tensors="pt",
        add_special_tokens=False
    ).input_ids.to(device)

     # optionally append EOS so the model has a termination token to predict
    if processor.tokenizer.eos_token_id is not None:
        # append eos to the target sequence so we include the model's probability of finishing.
        target_ids = torch.cat([target_ids, torch.tensor([[processor.tokenizer.eos_token_id]], device=device)], dim=1)

    # Step 3) Concatenate context and target
    input_ids = torch.cat([context_inputs["input_ids"], target_ids], dim=1)

    # labels
    labels = input_ids.clone()
    labels[0, :context_len] = IGNORE_INDEX

    # check to be sure that the labels are correct
    # assert labels[0, :context_inputs["input_ids"].shape[1]] == -100
    # assert target == processor.tokenizer.decode(labels[0, context_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # Step4) Calculate loss
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            images=pixel_values, # from processor
            labels=labels
        )
    loss = outputs["loss"] # average negative log-likelihood over active tokens
    logits = outputs["logits"]
    
    
    log_likelihood = logits.log_softmax(dim=-1)
    log_likelihood = log_likelihood[:, context_inputs["input_ids"].shape[1]:input_ids.shape[1]]
    log_likelihood = log_likelihood[torch.arange(log_likelihood.shape[0]), torch.arange(log_likelihood.shape[1]), target_ids]
    
    # Alternative (faster) - recover from HF loss:
    # number of active (non-ignored) tokens in labels
    num_active = (labels != IGNORE_INDEX).sum().item()
    # HF loss is the average negative log-likelihood over active tokens 
    # So: sum_logprob = - loss * num_active (negative mean log probability -> log probability)
    loss_based_sum_logprob = -loss * num_active

    # import pdb; pdb.set_trace()
    # Step5) Greedy check
    # greedy_tokens = logits.argmax(dim=-1)
    # greedy_tokens = greedy_tokens[:, context_inputs["input_ids"].shape[1]:input_ids.shape[1]]  # Use only relevant predictions
    # # processor.tokenizer.decode(greedy_tokens[0], skip_special_tokens=True)
    # target_tokens = input_ids[:, context_inputs["input_ids"].shape[1]:]  # Skip the context tokens
    # print(f"Greedy tokens: {processor.tokenizer.decode(greedy_tokens, skip_special_tokens=True)}")
    # print(f"Target tokens: {processor.tokenizer.decode(target_tokens, skip_special_tokens=True)}")
    # max_equal = (greedy_tokens == target_tokens).all()

    
    return loss, loss_based_sum_logprob #, float(loss.item()) #, percentage_equal


if __name__ == "__main__":
    # model_path = "utter-project/TowerVision-Plus-2B"
    model_path = "utter-project/TowerVision-4-Anthill-CPT"
    # load model
    processor, model, device = load_model(model_path)    
    # load dataset
    ds = datasets.load_dataset("Unbabel/commute_multimodal_mt", split="fr")
    # run inference on full dataset
    results = []

    for i, sample in tqdm(enumerate(ds), total=len(ds)):
        # define prompt
        prompt = f"Translate from English to French:\n{sample['source']}"
        try:
            correct_likelihood, correct_likelihood_unnormalized = calculate_likelihood(
                sample["image"], prompt, sample["correct_translation"], 
                model=model, processor=processor, device=device)
            correct_control, correct_control_unnormalized = calculate_likelihood(
                sample["image"], prompt, sample["correct_translation"], 
                model=model, processor=processor, device=device, log_prompt=True)
            incorrect_likelihood, incorrect_likelihood_unnormalized = calculate_likelihood(
                sample["image"], prompt, sample["incorrect_translation"], 
                model=model, processor=processor, device=device)
            incorrect_control, incorrect_control_unnormalized = calculate_likelihood(
                sample["image"], prompt, sample["incorrect_translation"], 
                model=model, processor=processor, device=device)
            # print(f"Greedy Response: {response}")
            print(f"Correct Likelihood: {correct_likelihood}")
            print(f"Correct Control: {correct_control}")
            print(f"Incorrect Likelihood: {incorrect_likelihood}")
            print(f"Incorrect Control: {incorrect_control}")
        except Exception as e:
            print(f"Error calculating likelihood for sample {i}: {e}")
            continue
        
        results.append({
            "correct_likelihood": (correct_likelihood, correct_likelihood_unnormalized),
            "incorrect_likelihood": (incorrect_likelihood, incorrect_likelihood_unnormalized),
            "correct_control_likelihood": (correct_control, correct_control_unnormalized),
            "incorrect_control_likelihood": (incorrect_control, incorrect_control_unnormalized)
        })
        print(f"{i}/{len(ds)}")
       
        
    # save results in json
    model_name = model_path.split("/")[-1]
    with open(f"{model_name}_results_fr.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    
    # # print accuracy
    # correct_counts = [1 if lp_with_normalized < lp_noimg_normalized else 0 for lp_with_normalized, lp_noimg_normalized in zip(results["w/images"], results["w/oimages"])]
    # accuracy = sum(correct_counts) / len(correct_counts)
    # print(f"Accuracy: {accuracy}")

    