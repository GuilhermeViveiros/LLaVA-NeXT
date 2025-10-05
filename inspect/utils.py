import torch
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model

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
    if device == "cuda":
        device_map = f"{device}:0"
    else:
        device_map = "cpu"
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
    model.eval()
    
    return tokenizer, model, image_processor, device