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


TOWER_VISION_LANGUAGES_TO_ADD = [
    "ko", "hi", "sv", "pl", "is", "ja", "uk", "fi", "hu", "cs", "ro", "no", "da", "nl"
]

# sv — Swedish
# is — Icelandic
# fi — Finnish
# hu — Hungarian
# da — Danish
