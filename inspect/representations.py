### Import libraries ###
import os
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle
import datasets
import torch
import matplotlib.pyplot as plt
from utils import load_model
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import matplotlib.colors as mcolors
# Import LLaVA components
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

# TODO: Remove special tokens from the visualization and hidden states

### Get model likelihood ###
# Define Likelihood function
def inference(
    image, 
    prompt,
    target,
    tokenizer,
    model,
    image_processor,
    device,
    conv_template="gemma2_instruct",
    log_prompt=False,
):
    """
    Calculate the likelihood of a given translation for an image-text pair, plus the index of each modality token
    
    Args:
        image: PIL image
        source_text: Source text to translate
        target_text: Target translation to evaluate
        tokenizer, model, image_processor: Pre-loaded model components
        device: The device to run inference on
        conv_template: Conversation template to use
    
    Returns:
        tuple: (hidden states, attentions, modality indices)
    """

    # with profile(
    #     activities=[
    #         ProfilerActivity.CUDA,
    #         ProfilerActivity.CPU
    #     ],
    #     profile_memory=True,
    #     record_shapes=True,
    #     with_stack=True,
    #     with_flops=True
    # ) as prof:

        #with record_function("process_image"):
            # Process image
    image_size = [image.size[0], image.size[1]]
    processed_image = process_images([image], image_processor, model.config)
    processed_image = processed_image.to(dtype=torch.bfloat16, device=device)

    # Prepare prompt
    if DEFAULT_IMAGE_TOKEN not in prompt:
        #print(f"Adding {DEFAULT_IMAGE_TOKEN} to prompt")
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt


    conv = conv_templates[conv_template].copy()
    # First create context-only conversation (up to model's turn)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    context_prompt = conv.get_prompt()

    if log_prompt:
        print("Context prompt: ", context_prompt)

    # Tokenize context
    context_ids = tokenizer_image_token(context_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(device)

    # Now create full conversation with target answer
    conv.messages[-1][1] = target  # Set the model's response to the target text
    full_prompt = conv.get_prompt()
    
    if log_prompt:
        print("Full prompt: ", full_prompt)

    # Tokenize full prompt
    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    # get text token indices
    text_token_indices = input_ids.clone()
    text_token_indices[0, :context_ids.shape[0]] = -100
    text_token_indices = text_token_indices.nonzero()

    # Create labels with -100 for context part (we don't compute loss for it)
    labels = input_ids.clone()
    labels[0, :context_ids.shape[0]] = -100

    # Calculate loss
    #with record_function("model_inference"):
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids, 
            labels=labels, 
            images=processed_image, 
            use_cache=False, 
            image_sizes=[image_size],
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )
    
    #torch.cuda.synchronize()

    # Get loss, hidden states, and attentions
    loss = outputs["loss"]
    hidden_states = outputs["hidden_states"]
    attentions = outputs["attentions"]
    token_indices = outputs["token_indices"]
    
    # loss is the average negative log-likelihood over active tokens
    #active_tokens = (labels != -100).sum()
    #log_likelihood = -loss * active_tokens
    # compute perplexity
    #perplexity = torch.exp(loss)

    # ----- MEMORY + PROFILING REPORT -----
    print("\nPeak CUDA memory usage:",
          torch.cuda.max_memory_allocated(device) / 1e9, "GB")

    #print(prof.key_averages().table(
    #    sort_by="cuda_memory_usage", row_limit=20)
    #)

    # You can also export to TensorBoard for detailed timeline
    # prof.export_chrome_trace("trace_gemma2.json")
    # export to tensorboard prof


        
    return hidden_states, attentions, token_indices
    #return float(loss.item()), float(perplexity.item()), float(log_likelihood.item())

### Extract hidden states and attentions from a given file ###
def extract_hidden_states_and_attentions(dataset:str, model_path:str, outputs_folder:str):
    ### Load dataset ###
    ds = datasets.load_dataset(dataset)
    langs = {
        "ar": "Arabic",
        "cs": "Czech",
        "de": "German",
        "fr": "French",
        "ru": "Russian",
        "zh": "Chinese",
    }
    ### Load model ###
    tokenizer, model, image_processor, device = load_model(model_path, device="cuda")
    model.to("cuda")
    print(f"Model loaded on {device}")
    ### Get model likelihood ###
    #for lang in ds.keys():
    lang = "fr"
    print(f"Language: {lang}")
    ## WARNING: This will take a while and a lot of disck space -> pay attention when running this
    for i, sample in tqdm(enumerate(ds[lang]), total=10):
        language = langs[lang]
        prompt = f"Translate from English to {language}:\n{sample['source']}"
        # extract hidden states and attentions, plus the index of each modality token
        hidden_states, attentions, token_indices = inference(sample["image"], prompt, sample["correct_translation"], tokenizer, model, image_processor, device)
        values = {
            "hidden_states": [h.detach().to(dtype=torch.float32, device='cpu').numpy() for h in hidden_states],
            "attentions": [a.detach().to(dtype=torch.float32, device='cpu').numpy() for a in attentions],
            "token_indices": token_indices,
        }
        # save all three in a single file
        np.save(f"{outputs_folder}/values_{lang}_{i}.npy", values)

        # save hidden states and attentions
        #np.save(f"{outputs_folder}/hidden_states_{lang}_{i}.npy", [h.detach().to(dtype=torch.float32, device='cpu').numpy() for h in hidden_states])
        #np.save(f"{outputs_folder}/attentions_{lang}_{i}.npy", [a.detach().to(dtype=torch.float32, device='cpu').numpy() for a in attentions])
        # delete hidden states and attentions
        del hidden_states, attentions
        torch.cuda.empty_cache()


def load_state(file_path:str): # file structure - values_{lang}_{sample_index}.npy -> values = {hidden_states, attentions, token_indices}
    ### Load hidden states, attentions & token indices from a given file ###
    # get lang from file path
    lang = file_path.split("/")[-2]
    # get sample index from file path
    sample_index = file_path.split("/")[-1].split(".")[0].split("_")[-1]
    # load hidden states and attentions
    data = np.load(file_path, allow_pickle=True).item()

    return lang, sample_index, [*data.values()]
    


if __name__ == "__main__":
    outputs_folder = "/mnt/scratch-artemis/gviveiros/results2/"
    # extract hidden states and attentions
    extract_hidden_states_and_attentions(
        dataset="Unbabel/commute_multimodal_mt",
        model_path="/mnt/scratch-artemis/gviveiros/TowerVision/llava-next-native/towerp_9b_instruct/",
        outputs_folder=outputs_folder
    )
    # load hidden states and attentions
    lang, sample_index, values = load_state(f"{outputs_folder}/values_fr_0.npy")
    h_states, \
        attentions, \
            token_indices = values
    
    token_indices = token_indices[0]
    # Create an array to store modality labels for each token position
    seq_len = np.empty(sum(len(indices) for indices in token_indices.values()), dtype=object)
    for modality, indices in token_indices.items():
        for index in indices:
            seq_len[index.item()] = modality

    # every element of seq_len is not None
    assert all(seq_len[i] is not None for i in range(len(seq_len))), "None is not allowed in seq_len"

    # Map modalities to colors
    unique_modalities = list(set(seq_len))
    modality_to_color = {modality: color for modality, color in zip(unique_modalities, ['orange', 'blue'])}
    modality_to_int = {modality: i for i, modality in enumerate(unique_modalities)}
    color_labels = np.array([modality_to_int[mod] for mod in seq_len])

    # Let's apply PCA to the hidden states to 2 dimensions and progressively append the distribution to the image
    num_layers = len(h_states)
    images_per_row = 5
    num_rows = math.ceil(num_layers / images_per_row)

    plt.figure(figsize=(images_per_row * 3, num_rows * 3))
    for layer_idx, (h_layer, attn_layer) in tqdm(enumerate(zip(h_states, attentions)), total=num_layers):
        # layer shape: (sequence_length, hidden_size)
        layer = np.squeeze(h_layer)
        attn_layer = np.squeeze(attn_layer)
        # average across heads
        attn_layer = np.mean(attn_layer, axis=0)
        layer = attn_layer
        # ---- Preprocessing ----
        layer_norm = layer / (np.linalg.norm(layer, axis=1, keepdims=True) + 1e-8)
        # Remove magnitude outliers
        magnitudes = np.linalg.norm(layer, axis=1)
        thr = np.mean(magnitudes) + 3 * np.std(magnitudes)
        outlier_mask = magnitudes > thr
       
        layer_filtered = layer_norm[~outlier_mask]
        color_labels_filtered = color_labels[~outlier_mask]
        # get how many outliers are from each modality
        for modality in unique_modalities:
            print(f"Modality {modality}: Removed {outlier_mask[color_labels == modality_to_int[modality]].sum()} out of {len(outlier_mask[color_labels == modality_to_int[modality]])} tokens ({100*outlier_mask[color_labels == modality_to_int[modality]].mean():.2f}%)")
        #import pdb; pdb.set_trace()
        print(f"Removed {outlier_mask.sum()} out of {len(outlier_mask)} tokens ({100*outlier_mask.mean():.2f}%)")

        # apply pca (keep 2 dimensions - modalities interactions)
        pca_values = PCA(n_components=2).fit_transform(layer_filtered)

        # compute subplot index
        row = layer_idx // images_per_row
        col = layer_idx % images_per_row
        ax = plt.subplot(num_rows, images_per_row, layer_idx + 1)

        # Scatter plot of PCA - modalities interactions, colored by modality
        # Use blue for modality 0 and orange for modality 1
        custom_cmap = mcolors.ListedColormap(['orange', 'blue'])
        scatter = plt.scatter(
            pca_values[:, 0],
            pca_values[:, 1],
            c=color_labels_filtered,
            cmap=custom_cmap,
            alpha=0.5,
            s=10
        )
        plt.title(f'Layer {layer_idx} - Modalities Interactions')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')

        # Only add legend to the first subplot
        if layer_idx == 0:
            handles = [
                plt.Line2D([0], [0], marker='o', color='w', label=modality,
                           markerfacecolor=modality_to_color[modality], markersize=8)
                for modality in unique_modalities
            ]
            plt.legend(handles=handles, title="Modality", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f"{outputs_folder}/hidden_states_pca_distribution.png")
    plt.close()

    #print(f"Hidden states shape: {[h.shape for h in h_states]}")
    #print(f"Attentions shape: {attentions.shape}")