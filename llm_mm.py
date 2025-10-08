from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, LlavaNextConfig
from transformers import Gemma2Config, Gemma2ForCausalLM
import torch


def merge_llm_mm(mm_model, llm_model, alpha: float):
    # extract text only llm parameters from mm_model
    llm_mm_params = {
        name: param for name, param in mm_model.language_model.named_parameters()
    }

    # remove head from llm_model
    llm_params = {
        name: param for name, param in llm_model.model.named_parameters()
    }
    # merge the parameters
    merged_params = {}
    try:
        for (name, mm_param), (_, llm_param) in zip(llm_mm_params.items(), llm_params.items()):
            # for the embeddings we need a special treatment (since mm model hads another token for the visual tokens)
            if "embed_tokens" in name:
                merged_params[name] = torch.cat(
                    [alpha * mm_param.data[:-1] + (1 - alpha) * llm_param.data,
                    mm_param.data[-1:]],
                    dim=0,
                )
            else:
                merged_params[name] = alpha * mm_param.data + (1 - alpha) * llm_param.data
    except Exception as e:
        raise Exception(f"Error merging llm and mm parameters: {e} {name} {mm_param.data.shape} {llm_param.data.shape}")
    
    # load merged parameters to mm_model
    mm_model.language_model.load_state_dict(merged_params)

    # return the merged model
    return mm_model

if __name__ == "__main__":
    # training-free recovery solution for retaining text-only performance
    # check section 3.2 from Aya-Vision: https://arxiv.org/abs/2505.08751
    size = "9B"
    mm_model_name = f"utter-project/TowerVision-Plus-{size}"
    llm_model_name = f"Unbabel/Tower-Plus-{size}"
    device = "cuda:0"

    # use sdpa attention
    kwargs = {
        "dtype": "bfloat16",
        "local_files_only": True,
        "trust_remote_code": False,
    }

    # load the models
    tokenizer = LlavaNextProcessor.from_pretrained(mm_model_name, local_files_only=True)
    # load mm model
    cfg_pretrained = LlavaNextConfig.from_pretrained(mm_model_name)
    mm_model = LlavaNextForConditionalGeneration.from_pretrained(mm_model_name, low_cpu_mem_usage=True, config=cfg_pretrained, attn_implementation="sdpa", device_map=device, **kwargs)
    # load llm model
    cfg_pretrained = Gemma2Config.from_pretrained(llm_model_name)
    llm_model = Gemma2ForCausalLM.from_pretrained(llm_model_name, low_cpu_mem_usage=True, config=cfg_pretrained, attn_implementation="sdpa", device_map=device, **kwargs)

    # merging coefficient alpha = 0.4
    alpha = 0.4

    # send models to cpu
    mm_model.to("cpu")
    llm_model.to("cpu")

    # merge the models
    merged_model = merge_llm_mm(mm_model, llm_model, alpha)
    print(f"Merged model successfully created with alpha = {alpha}")

    # save the merged model
    output_dir = "/mnt/scratch-artemis/gviveiros/TowerVision/"
    print(f"Saving merged model to {output_dir}/TowerVisionMerged-Plus-{size}-{alpha}")
    merged_model.save_pretrained(f"{output_dir}/TowerVisionMerged-Plus-{size}-{alpha}")
    tokenizer.save_pretrained(f"{output_dir}/TowerVisionMerged-Plus-{size}-{alpha}")
