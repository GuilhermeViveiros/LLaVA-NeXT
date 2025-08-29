import json
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
sns.set_theme(style="whitegrid")
from tqdm import tqdm
import yaml
from collections import Counter
import math
import random
from dataclasses import dataclass, field
from torch.utils.data import Dataset
import torch
import os
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from llava.constants import TOWER_VISION_LANGUAGES


TOWER_VISION_LANGUAGES_IDX = {v: k for k, v in TOWER_VISION_LANGUAGES.items()}

# language detection model
LANG2IDX = {
    "Arabic": 0,
    "Basque": 1,
    "Breton": 2,
    "Catalan": 3,
    "Chinese_China": 4,
    "Chinese_Hongkong": 5,
    "Chinese_Taiwan": 6,
    "Chuvash": 7,
    "Czech": 8,
    "Dhivehi": 9,
    "Dutch": 10,
    "English": 11,
    "Esperanto": 12,
    "Estonian": 13,
    "French": 14,
    "Frisian": 15,
    "Georgian": 16,
    "German": 17,
    "Greek": 18,
    "Hakha_Chin": 19,
    "Indonesian": 20,
    "Interlingua": 21,
    "Italian": 22,
    "Japanese": 23,
    "Kabyle": 24,
    "Kinyarwanda": 25,
    "Kyrgyz": 26,
    "Latvian": 27,
    "Maltese": 28,
    "Mongolian": 29,
    "Persian": 30,
    "Polish": 31,
    "Portuguese": 32,
    "Romanian": 33,
    "Romansh_Sursilvan": 34,
    "Russian": 35,
    "Sakha": 36,
    "Slovenian": 37,
    "Spanish": 38,
    "Swedish": 39,
    "Tamil": 40,
    "Tatar": 41,
    "Turkish": 42,
    "Ukrainian": 43,
    "Welsh": 44,
    None: 45,
    "Korean": 46,
}

IDX2LANG = {v: k for k, v in LANG2IDX.items()}

@dataclass
class DataArguments:
    dataset_paths: list[str] = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str):
        super(LazySupervisedDataset, self).__init__()
        self.list_data_dict = []
        data_args = DataArguments()
    
        # Handle multiple JSON files specified in the data_path
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                print(f"Loading {full_path}")
                with open(full_path, "r") as file:
                    cur_data_dict = json.load(file)
                    print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # FIXME: harcoded datasets to explore the multilinguality distribution
                datasets = [x for x in datasets if x.get("json_path").split("/")[-1] in ["pangea-cultural-150k.json", "pangea-multi-1m.json", "pixmo-cap-translated.json"]]
                
                data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None

                    print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            data_args.dataset_paths = [data_path]
            print(f"Loading {data_path}")
            with open(data_path, "r") as file:
                cur_data_dict = json.load(file)
                print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)

        print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        print("Formatting inputs...Skip in lazy mode")
        self.data_args = data_args
        # only use ids ['7', '9', '33', '34', '40', '41', '49', '88', '105', '121']
        # self.list_data_dict = [x for x in self.list_data_dict if x.get("id") in ['7', '9', '33', '34', '40', '41', '49', '88', '105', '121']]
    def __len__(self):
        return len(self.list_data_dict)
    def __getitem__(self, index):
        return self.list_data_dict[index]

def plot_language_distributions(messages_by_lang, samples_by_lang):
    # Sort by count
    msgs_sorted = sorted(messages_by_lang.items(), key=lambda x: x[1], reverse=True)
    samples_sorted = sorted(samples_by_lang.items(), key=lambda x: x[1], reverse=True)

    # Split into labels and values
    msg_labels, msg_counts = zip(*msgs_sorted)
    sample_labels, sample_counts = zip(*samples_sorted)

    # Mark tower languages
    tower_langs = set(TOWER_VISION_LANGUAGES.values())

    # --- Align labels ---
    # find common labels
    common_labels = [lang for lang in msg_labels if lang in sample_labels]
    # labels unique to each list
    msg_unique = [lang for lang in msg_labels if lang not in common_labels]
    sample_unique = [lang for lang in sample_labels if lang not in common_labels]

    # final ordered labels: common first, then unique from messages, then unique from samples
    ordered_labels = common_labels + msg_unique + sample_unique

    # reorder counts to match the ordered_labels
    msg_counts_ordered = [msg_counts[msg_labels.index(lang)] if lang in msg_labels else 0 for lang in ordered_labels]
    sample_counts_ordered = [sample_counts[sample_labels.index(lang)] if lang in sample_labels else 0 for lang in ordered_labels]

    # replace None with "No Majority" for samples
    ordered_labels = ["No Majority" if lang is None else lang for lang in ordered_labels]

    # create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot messages per language ---
    total_msgs = sum(msg_counts_ordered)
    colors = ["orange" if lang in tower_langs else "steelblue" for lang in ordered_labels]
    bars1 = axes[0].bar(ordered_labels, msg_counts_ordered, color=colors)

    
    axes[0].set_xticklabels(ordered_labels, rotation=75, ha="right")
    axes[0].set_title("Messages per Language")
    axes[0].set_xlabel("Language")
    axes[0].set_ylabel("Number of Messages")

    for bar, count in zip(bars1, msg_counts_ordered):
        height = bar.get_height()
        pct = 100 * count / total_msgs
        axes[0].text(bar.get_x() + bar.get_width() / 2, height, f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)

    # --- Plot samples per language ---
    total_samples = sum(sample_counts_ordered)
    colors = ["orange" if lang in tower_langs else "steelblue" for lang in ordered_labels]
    bars2 = axes[1].bar(ordered_labels, sample_counts_ordered, color=colors)
    axes[1].set_xticklabels(ordered_labels, rotation=75, ha="right")
    axes[1].set_title("Samples per Language")
    axes[1].set_xlabel("Language")
    axes[1].set_ylabel("Number of Samples")

    for bar, count in zip(bars2, sample_counts_ordered):
        height = bar.get_height()
        pct = 100 * count / total_samples
        axes[1].text(bar.get_x() + bar.get_width() / 2, height, f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)

    # --- Add a single legend for the whole figure ---
    legend_elements = [
        Patch(facecolor='orange', label='TowerVision Languages'),
        Patch(facecolor='steelblue', label='Other Languages')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space at top for legend
    plt.savefig("samples_per_language.png", dpi=300)
    plt.show()

def plot_tower_vision_languages_distribution(samples):
    """
    messages_by_lang: dict mapping language names to number of messages
    """
    from matplotlib.patches import Patch

    # create a bar for each language supported in messages_by_lang + TOWER_VISION_LANGUAGES
    # the bars should be sorted by the number of messages
    # add a mark for the languages that are part of the TOWER_VISION_LANGUAGES
    tower_vision_languages = set(TOWER_VISION_LANGUAGES.values())
    # Separate languages into Tower Vision vs others
    languages = set(list(samples.keys()) + list(tower_vision_languages))
    counts = [samples.get(lang, 0) for lang in languages]
    
    # save in a json file the languages of tower and counts of messages
    with open("tower_vision_languages_messages_count.json", "w") as f:
        json.dump({lang: messages_by_lang.get(lang, 0) for lang in tower_vision_languages}, f, ensure_ascii=False, indent=2)

    # Sort by number of messages
    sorted_pairs = sorted(zip(languages, counts), key=lambda x: x[1], reverse=True)
    sorted_languages, sorted_counts = zip(*sorted_pairs)
    
    # Assign colors: highlight Tower Vision languages
    colors = ['orange' if lang in tower_vision_languages else 'steelblue' for lang in sorted_languages]

    # Plot
    plt.figure(figsize=(14, 6))
    bars = plt.bar(sorted_languages, sorted_counts, color=colors)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{height}',
            ha='center',
            va='bottom'
        )
    
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Number of messages")
    plt.title("Samples per language (Tower Vision languages highlighted in orange)")

    # Add legend for orange and blue
    legend_elements = [
        Patch(facecolor='orange', label='TowerVision Languages'),
        Patch(facecolor='steelblue', label='Other Languages')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("messages_per_language_tower_vision_languages.png", dpi=300)
    
def maj_lang_per_conv(language_map):
    # Reduce to majority language per conversation
    final_langs = {}
    for cid, langs in language_map.items():
        lang_counts = Counter(langs)
        maj_lang, count = lang_counts.most_common(1)[0]
       
        if count / len(langs) >= 0.5:
            final_langs[cid] = maj_lang
        else:
            final_langs[cid] = None
    
    return final_langs

def batch_detect_languages(dataset, tokenizer, model, batch_size=64, device="cuda"):
    conv_ids, msgs = [], []
    for sample in tqdm(dataset, desc="Processing samples", total=len(dataset)):
        cid = sample.get("id")
        conv_msgs = [
            m["value"].replace("<image>", "").replace("\n", "").strip()
            for m in sample.get("conversations", [])[1:]
            if m.get("from") != "human"
        ]
        for msg in conv_msgs:
            conv_ids.append(cid)
            msgs.append(msg)
        #break
    language_map = {}
    for i in tqdm(range(0, len(msgs), batch_size), desc="Batch processing"):
        batch_msgs = msgs[i:i+batch_size]
        batch_ids = conv_ids[i:i+batch_size]

        encoded = tokenizer(batch_msgs, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**encoded)
        preds = torch.argmax(outputs.logits, dim=1).tolist()

        for cid, pred in zip(batch_ids, preds):
            lang = IDX2LANG[pred]
            language_map.setdefault(cid, []).append(lang)
    # language_map now has: {conversation_id: [lang_msg1, lang_msg2, ...]}
    return language_map

def classify_languages():
    # sft conversational data
    dataset = LazySupervisedDataset(data_path="/mnt/scratch-artemis/gviveiros/TowerVision/visionblocks_v6p5_SFT_euro.yaml")

    # load the language detection model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("jb2k/bert-base-multilingual-cased-language-detection")
    model = AutoModelForSequenceClassification.from_pretrained("jb2k/bert-base-multilingual-cased-language-detection")
    model = model.to(device)
    
    # count number of samples per language
    conversation_langs = batch_detect_languages(dataset, tokenizer, model, batch_size=64)
    
    # Save results
    # the model uses a lot of japonese and chinese-taiwan as korean
    with open("conversation_languages.json", "w") as f:
        json.dump(conversation_langs, f, ensure_ascii=False, indent=2)


def analyse(language_counts_file: str):
    # count number of samples per language
    with open(language_counts_file, "r") as f:
        language_counts = json.load(f)
    # iterate each sample and replace any Chinese_Taiwan or Japanese with Korean
    for sample_id, language_list in language_counts.items():
        if language_list:
            language_list = [lang if lang != "Chinese_Taiwan" and lang != "Japanese" else "Korean" for lang in language_list]
            language_counts[sample_id] = language_list
    
    # count number of messages per language
    # Flatten all language lists and count individual languages
    all_languages = []
    for sample_id, language_list in language_counts.items():
        if language_list:  # handle None/empty lists
            all_languages.extend(language_list)
    # ignore languages with less than 500 messages
    messages_by_lang = {lang: count for lang, count in Counter(all_languages).items() if count >= 100}
    

    #messages_by_lang = Counter(all_languages)
    print("Messages per language:")
    for lang, count in sorted(messages_by_lang.items(), key=lambda x: x[1], reverse=True):
        print(f"{lang}: {count} messages")
    
    print(f"\nTotal messages: {sum(messages_by_lang.values())}")
    print(f"Total samples: {len(language_counts)}")
    # get message samples ratio
    print(f"Message samples ratio: {round(sum(messages_by_lang.values()) / len(language_counts), 2)}")
    print(f"Unique languages: {len(messages_by_lang)}")
    
    # lets check the distribution of samples per language
    majority_lang_per_samples = maj_lang_per_conv(language_counts)
    samples_by_lang = Counter(majority_lang_per_samples.values())
    # ignore languages with less than 50 samples
    samples_by_lang = {lang: count for lang, count in samples_by_lang.items() if count >= 50}
    
    print("\nSamples per language:")
    for lang, count in samples_by_lang.items():
        print(f"{lang}: {count} samples")

    # print number of samples per language for TOWER_LANGUAGES
    print("\nTOWER_LANGUAGES breakdown:")
    for accr, lang in TOWER_VISION_LANGUAGES.items():
        if lang in samples_by_lang:
            print(f"{accr}: {samples_by_lang[lang]} samples, {messages_by_lang.get(lang, 0)} messages. Ratio: {round(messages_by_lang.get(lang, 0) / samples_by_lang[lang], 2)}")
        else:
            print(f"{accr}: 0 samples, 0 messages")


    # create a subplot
    # 1. chart that contains the number of messages per language in a sort order
    # 2. chart that contains the number of samples per language in a sort order
    # 3. each bar of the charts should have a mark it the language is part of the TOWER_LANGUAGES
    plot_language_distributions(messages_by_lang, samples_by_lang)

def analyse_given_tower_vision_languages(language_counts_file: str):
    # count number of samples per language
    with open(language_counts_file, "r") as f:
        language_counts = json.load(f)
    # iterate each sample and replace any Chinese_Taiwan or Japanese with Korean
    for sample_id, language_list in language_counts.items():
        if language_list:
            language_list = [lang if lang != "Chinese_Taiwan" and lang != "Japanese" else "Korean" for lang in language_list]
            language_list = [lang if lang != "Chinese_China" and lang != "Chinese_Hongkong" else "Chinese" for lang in language_list]
            language_counts[sample_id] = language_list
        
    import pdb; pdb.set_trace()
    majority_lang_per_samples = maj_lang_per_conv(language_counts)

    # get tamil samples
    tamil_samples = [sample_id for sample_id, lang in majority_lang_per_samples.items() if "Tamil" == lang]
    

    
    # count number of messages per language
    # Flatten all language lists and count individual languages
    all_languages = []
    for sample_id, language_list in language_counts.items():
        if language_list:  # handle None/empty lists
            all_languages.extend(language_list)
    # ignore languages with less than 500 messages
    messages_by_lang = {lang: count for lang, count in Counter(all_languages).items() if count >= 100}
    
    # lets create a bar for each language supported by TowerVision
    
    plot_tower_vision_languages_distribution(messages_by_lang)

    



if __name__ == "__main__":
    # main()
    analyse_given_tower_vision_languages("conversation_languages.json")
    