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
        
    def __len__(self):
        return len(self.list_data_dict)
    def __getitem__(self, index):
        return self.list_data_dict[index]

    
def main():
    dataset = LazySupervisedDataset("/mnt/scratch-artemis/gviveiros/TowerVision/visionblocks_v6p5_SFT_euro.yaml")
    images_per_sample = {}
    import pdb; pdb.set_trace() 
    for sample in dataset:
        id = sample["id"]
        conv = sample["conversations"]
        nb_imgs = 0
        for c in conv:
            if "<image>" in c["value"]:
                nb_imgs += 1
        images_per_sample.setdefault(nb_imgs, []).append(id)
    import pdb; pdb.set_trace()
    # count how many samples per number of images
    images_per_sample_count = {k: len(v) for k, v in images_per_sample.items()}
    import pdb; pdb.set_trace()
    # plot the distribution of images per sample
    plt.hist(list(images_per_sample_count.keys()), bins=range(max(images_per_sample_count.keys()) + 1))
    plt.savefig("images_per_sample.png")
    plt.show()
                
                


if __name__ == "__main__":
    main()
    