#!/usr/bin/env python3
"""
Script to convert Hugging Face datasets into the format used by LLaVA-NeXT.
This script handles both single-image and multi-image datasets.
"""

import os
import json
from collections import Counter
import random
import argparse
from typing import Union
from pathlib import Path
from typing import List, Dict, Any, Optional
from llava.constants import TOWER_VISION_LANGUAGES_TO_ADD
import json

class CulturalGroundDataset:
    def __init__(self, data_path: str, verbose: bool = False):
        """
        Args:
            data_path: Path to the json file
            verbose: Whether to print verbose output
        """
        self.data_path = data_path
        self.verbose = verbose
        self.data = self.load_data()
        self._set_of_languages = set(self.data.keys())

    def load_data(self):
        """
        Loads the data from the json file
        """
        try:
            data = {}
            with open(self.data_path, "r") as f:
                file_data = json.load(f)
            # aggregate data by language
            for item in file_data:
                data.setdefault(item["language"], []).append(item)
        except Exception as e:
            print(f"Error loading data from {self.data_path}: {e}")
            raise e
        except FileNotFoundError:
            print(f"File {self.data_path} not found")
            raise FileNotFoundError(f"File {self.data_path} not found")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {self.data_path}: {e}")
            raise json.JSONDecodeError(f"Error decoding JSON from {self.data_path}: {e}")
        return data

    def samples(self, langs: Union[str, List[str]], max_samples: Optional[int] = None):
        """
        Args:
            lang:s Language to get samples for
        Returns:
            List of samples
        """
        if isinstance(langs, str):
            langs = [langs]
        # check if any lang is not in the dataset
        _set = []
        for lang in langs:
            if lang not in self._set_of_languages:
                print(f"Language {lang} not found in dataset")
            else:
                _set.append(lang)
        langs = _set
        # for each sample get randomly max_samples samples
        if max_samples is not None:
            for lang in langs:
                nb_samples = min(max_samples, len(self.data[lang]))
                print(f"Getting {nb_samples} samples for language {lang}")
                self.data[lang] = random.sample(self.data[lang], nb_samples)
        return [item for lang in langs for item in self.data[lang]]

    def lang_counts(self, langs: Optional[List[str]] = None):
        """
        Args:
            langs: List of languages to count (optional)
        Returns:
            Dictionary with language counts
        """
        langs = self.samples(langs)
        # return samples for each language
        return {lang: len(lang) for lang in langs}

    def intersection(self, x: Union[List[str], str]) -> List[str]:
        """
        Args:
            x: List of languages to intersect with
        Returns:
            List of languages that are in the intersection
        """
        if isinstance(x, str):
            x = [x]
            
        overlap = [item for item in x if item in self._set_of_languages]
        non_overlap = [item for item in x if item not in self._set_of_languages]

        print("Overlapping ratio: ", len(overlap) / (len(overlap) + len(non_overlap)))
        print("Non-overlapping languages: ", non_overlap)

        return [item for item in self._set_of_languages if item in x]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # /mnt/home/gviveiros/.cache/huggingface/hub/datasets--neulab--CulturalGround/snapshots/7e1c549862550a7b4698b734385449fae6c02ecd/CulturalGround-MCQs-Filtered-7M.json
    # parser.add_argument("--data_path", type=str, default="/mnt/home/gviveiros/.cache/huggingface/hub/datasets--neulab--CulturalGround/snapshots/7e1c549862550a7b4698b734385449fae6c02ecd/CulturalGround-MCQs-Filtered-7M.json")
    parser.add_argument("--data_path", type=str, default="/mnt/home/gviveiros/.cache/huggingface/hub/datasets--neulab--CulturalGround/snapshots/7e1c549862550a7b4698b734385449fae6c02ecd/CulturalGround-OE-Filtered-14M.json")
    # parser.add_argument("--data_path", type=str, default="data/Curated-CulturalGround-MCQs-Filtered-379834.json")

    parser.add_argument("--output_folder", type=str, default="data/")
    parser.add_argument("--lang_of_interest", type=str, nargs='+', default=["en", "de", "nl", "pt", "ru", "zh", "ko", "es", "fr", "it"])
    args = parser.parse_args()

    print("Args: ", args)
    

    dataset = CulturalGroundDataset(
        args.data_path,
        verbose=True
    )

    # tower vision supported languages (at the moment)
    # dataset.intersection(
    #     ["en", "de", "nl", "pt", "ru", "zh", "ko", "es", "fr", "it"]
    # )
    # backbone (llm) of tower vision supported languages
    # dataset.lang_counts(["ko", "hi", "sv", "pl", "is", "ja", "uk", "fi", "hu", "cs", "ro", "no", "da"])
    # dataset.intersection(
    #     ["ko", "hi", "sv", "pl", "is", "ja", "uk", "fi", "hu", "cs", "ro", "no", "da"]
    # )
    
    from llava.constants import TOWER_VISION_LANGUAGES_TO_ADD
    samples = dataset.samples(TOWER_VISION_LANGUAGES_TO_ADD, max_samples=50000)
    nb_samples = len(samples)
    data_source = args.data_path.split("/")[-1].split("-")[0]
    # add for each sample this key
    # import pdb; pdb.set_trace()
    for sample in samples:
        sample["data_source"] = data_source
        sample["image"] = data_source + "/" + "/".join(sample["image"].split("/")[1:])
    
    # output file will be the same as the dataset name but in the output folder
    output_file = os.path.join(args.output_folder, "Curated-" + args.data_path.split("/")[-1].replace("14M", str(nb_samples)))

    print("Output file: ", output_file)
    # save dataset to json
    with open(output_file, "w") as f:
        json.dump(samples, f)