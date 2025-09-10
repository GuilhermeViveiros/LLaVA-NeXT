import os
import sys


LANGUAGE_LABELS = {
    "english": "en",
    "german": "de",
    "chinese": "zh",
    "russian": "ru",
    "french": "fr",
    "czech": "cs",
    "portuguese": "pt",
    "vietnamese": "vi",
    "afrikaans": "af",
    "italian": "it",
}

TOWER_VISION_LANGUAGES_TO_ADD = [
    "ko", "hi", "sv", "pl", "is", "ja", "uk", "fi", "hu", "cs", "ro", "no", "da", "nl"
]

# TowerVision Languages
TOWER_VISION_LANGUAGES = {
    "de": "German",
    "nl": "Dutch",
    "is": "Icelandic",
    "es": "Spanish", # latin america
    "fr": "French",
    "pt": "Portuguese", # dielects also supports pt-BR
    "uk": "Ukrainian",
    "hi": "Hindi",
    "zh": "Chinese", # dielects supports Simplified and Traditional
    "ru": "Russian",
    "cs": "Czech",
    "ko": "Korean",
    "ja": "Japanese",
    "it": "Italian",
    "en": "English",
    "da": "Danish",
    "pl": "Polish",
    "hu": "Hungarian",
    "sv": "Swedish",
    "no": "Norwegian", # dielects supports Norwegian Bokmål and Norwegian Nynorsk
    "ro": "Romanian",
    "fi": "Finnish",
}

def get_lan(key: str):
    for lan, label in LANGUAGE_LABELS.items():
        if key.lower() == lan.lower():
            return label
    #logger.error(f"No language found for key: {key}")
    return key

def parse_alm_benchmark(results: dict, nb_samples: dict):
    out = {}
    samples_per_language = {}
    for k,v in results.items():
        if "alm-bench" in k:
            lan = k.split("-")[-1]
            lan = get_lan(lan)
            samples_per_language[lan] = nb_samples[k]["original"]
            out[lan] = v["exact_match,none"]
    if len(out) == 0:
        raise ValueError("No ALM benchmark results found")
    # create an average across all the languages
    if out.get("all", None) is None:
        out["all"] = sum(out.values()) / len(out.values())
    samples_per_language["all"] = sum(samples_per_language.values())
    return out, samples_per_language

def parse_textvqa_benchmark(results: dict, nb_samples: dict):
    return {"all": results["textvqa_val"]["exact_match,none"]}, {"all": nb_samples["textvqa_val"]["original"]}

def parse_ccocr_benchmark(results: dict, nb_samples: dict):
    out = {}
    samples_per_language = {}
    for k,v in results.items():
        if "cc-ocr-multi-lan-" in k:
            lan = k.split("-")[-1]
            lan = get_lan(lan)
            out[lan] = v["ocr_results,none"]["macro_f1_score"]
            samples_per_language[lan] = nb_samples[k]["original"]
    if len(out) == 0:
        raise ValueError("No CC-OCR benchmark results found")
    # create an average across all the languages
    if out.get("all", None) is None:
        out["all"] = sum(out.values()) / len(out.values())
    samples_per_language["all"] = sum(samples_per_language.values())
    return out, samples_per_language

def parse_commute_benchmark(results: dict):
    out = {}
    for k,v in results.items():
        if "commute-all-contrastive" in k:
            continue
        if "commute-" in k:
            lan = k.split("-")[-1]
            lan = get_lan(lan)
            out[lan] = v["results,none"]["contrastive_accuracy"]
    
    import pdb; pdb.set_trace()
    if len(out) == 0:
        raise ValueError("No Commute benchmark results found")
    # create an average across all the languages
    if out.get("all", None) is None:
        out["all"] = sum(out.values()) / len(out.values())
    return out

def parse_m3exam_benchmark(results: dict, nb_samples: dict):
    out = {}
    samples_per_language = {}
    for k,v in results.items():
        if "m3exam_" in k:
            lan = k.split("_")[-1]
            lan = get_lan(lan)
            out[lan] = v["m3exam,none"]
            samples_per_language[lan] = nb_samples[k]["original"]
    if len(out) == 0:
        raise ValueError("No M3Exam benchmark results found")
    # create an average across all the languages
    if out.get("all", None) is None:
        out["all"] = sum(out.values()) / len(out.values())
    samples_per_language["all"] = sum(samples_per_language.values())
    return out, samples_per_language

def parse_ocr_benchmark(results: dict, nb_samples: dict):
    return {"all": results["ocrbench"]["ocrbench_accuracy,none"]}, {"all": nb_samples["ocrbench"]["original"]}

def parse_multi30k_benchmark(results: dict, nb_samples: dict):
    out = {}
    samples_per_language = {}
    for k,v in results.items():
        if "multi30k-all" in k:
            continue
    
        if "multi30k-" in k:
            lan = k.split("-")[-1]
            lan = get_lan(lan)
            out[lan] = v["results,none"]["avg_XCOMET-XL_score"]
            samples_per_language[lan] = nb_samples[k]["original"]
    if len(out) == 0:
        raise ValueError(f"No Multi30K benchmark results found")
    # create an average across all the languages
    if out.get("all", None) is None:
        out["all"] = sum(out.values()) / len(out.values())
    samples_per_language["all"] = sum(samples_per_language.values())
    return out, samples_per_language

