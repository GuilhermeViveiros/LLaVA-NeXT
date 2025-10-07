CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# TowerVision Languages
TOWER_VISION_LANGUAGES = {
    "de": "German",
    "nl": "Dutch",
    #"is": "Icelandic",
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
    #"da": "Danish",
    "pl": "Polish",
    #"hu": "Hungarian",
    #"sv": "Swedish",
    "no": "Norwegian", # dielects supports Norwegian Bokmål and Norwegian Nynorsk
    "ro": "Romanian",
    "zhs": "Chinese Simplified",
    "zht": "Chinese Traditional",
    "cz": "Czech",
    #"fi": "Finnish",
}

SUPPORTED_BYTEXT_BUTNOT_FOR_VISION = [
    "is", "fi", "hu", "sv", "da"
]

TOWER_VISION_ORIGINAL_LANGUAGES = [
    "en", "de", "nl", "pt", "ru", "zh", "ko", "es", "fr", "it"
]

TOWER_VISION_LANGUAGES_TO_ADD = [
    "hi", "pl", "ja", "uk", "cs", "ro", "no", "nl", "zhs", "zht", "cz"
]

# TOWER_VISION_LANGUAGES_TO_ADD = [
#     "hi", "pl", "ja", "uk", "cs", "ro", "no", "nl"
# ]

def tower_language_support(language:str):
    # check if language is in TOWER_VISION_LANGUAGES values
    # all to lower case
    if language == "chinese_simplified" or language == "chinese_traditional":
        language = "chinese"
    language = language.lower()
    values = [v.lower() for v in TOWER_VISION_LANGUAGES.values()]
    return language in values

# sv — Swedish
# is — Icelandic
# fi — Finnish
# hu — Hungarian
# da — Danish
