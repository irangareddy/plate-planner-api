
import re

import wordninja
import yaml

# Load YAML configuration for normalization
with open("src/ml_training/normalizer_config.yaml") as f:
    config = yaml.safe_load(f)

DESCRIPTORS = set(config.get("descriptors", []))
UNITS = set(config.get("units", []))
STOPWORDS = set(config.get("stopwords", []))
BLACKLIST = set(config.get("blacklist", []))

def normalize_ingredient(text, fallback=True, return_score=False):
    original = text
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)

    # WordNinja splitting
    split_tokens = []
    for word in text.split():
        split_tokens.extend(wordninja.split(word))

    # Filter unwanted tokens
    filtered = [
        t for t in split_tokens
        if t not in DESCRIPTORS and t not in UNITS and t not in STOPWORDS and t not in BLACKLIST and len(t) > 2
    ]

    if not filtered and fallback:
        last_token = split_tokens[-1] if split_tokens else ""
        return (last_token, "fallback") if return_score else last_token

    score = "strong" if len(filtered) >= 2 else "weak"
    normalized = " ".join(filtered)

    return (normalized, score) if return_score else normalized
