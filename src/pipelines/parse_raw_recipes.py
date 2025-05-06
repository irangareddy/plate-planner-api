import ast
import os
import re

import pandas as pd
import wordninja
import yaml

# --- Paths ---
RAW_DATA_PATH = "app/src/data/raw/recipe_dataset_200k.csv"
CLEANED_DATA_PATH = "/app/src/data/processed/ingredient_substitution/cleaned_ner.csv"
NORMALIZER_YAML_PATH = "src/ml_training/normalizer_config.yaml"  # <-- Adjust if needed

# --- Load YAML config ---
with open(NORMALIZER_YAML_PATH) as f:
    config = yaml.safe_load(f)

DESCRIPTORS = set(config.get("descriptors", []))
UNITS = set(config.get("units", []))
STOPWORDS = set(config.get("stopwords", []))
BLACKLIST = set(config.get("blacklist", []))

# --- Normalizer ---
def normalize_ingredient(text, fallback=True):
    text = text.lower()
    text = re.sub(r"[^a-z\\s]", "", text)

    split_tokens = []
    for word in text.split():
        split_tokens.extend(wordninja.split(word))

    filtered = [
        t for t in split_tokens
        if t not in DESCRIPTORS and t not in UNITS and t not in STOPWORDS and t not in BLACKLIST and len(t) > 2
    ]

    if not filtered and fallback:
        return split_tokens[-1] if split_tokens else text

    return " ".join(filtered)

# --- Load Dataset ---
print("📥 Loading dataset...")
df = pd.read_csv(RAW_DATA_PATH)

print("🔍 Parsing NER column...")
df["ner_list"] = df["NER"].apply(ast.literal_eval)

print("🧼 Normalizing ingredients using YAML-driven config...")
df["ner_list_cleaned"] = df["ner_list"].apply(lambda lst: [normalize_ingredient(x) for x in lst])

# --- Save Cleaned Output ---
print(f"💾 Saving cleaned data to {CLEANED_DATA_PATH}...")
os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)
df[["title", "ner_list_cleaned", "directions"]].to_csv(CLEANED_DATA_PATH, index=False)

print("✅ Done! Normalized dataset saved.")
