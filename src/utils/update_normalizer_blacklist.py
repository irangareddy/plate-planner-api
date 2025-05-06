import string

import yaml
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# --- Config ---
W2V_MODEL_PATH = "/app/src/data/models/ingredient_substitution/ingredient_w2v.model"
NORMALIZER_CONFIG_PATH = "src/ml_training/normalizer_config.yaml"
TOP_K = 1000  # How many top tokens to consider

# --- Load Word2Vec ---
print("📦 Loading Word2Vec model...")
model = Word2Vec.load(W2V_MODEL_PATH)

# --- Load YAML Config ---
with open(NORMALIZER_CONFIG_PATH) as f:
    config = yaml.safe_load(f)

existing_blacklist = set(config.get("blacklist", []))
whitelist = set(config.get("whitelist", [])) if "whitelist" in config else set()

# --- Utility: Mark noisy or meaningless tokens ---
def is_noise(token):
    # Skip multi-word tokens — assume those are valid unless split logic is added
    if " " in token:
        return False
    return (
        len(token) < 4 or
        token.lower() in ENGLISH_STOP_WORDS or
        token.lower() in string.punctuation or
        not token.isalpha()
    )

# --- Extract and Filter ---
print(f"🔍 Scanning top {TOP_K} tokens...")
vocab = model.wv.index_to_key[:TOP_K]

# Filter: single words only and not whitelisted
candidates = [t for t in vocab if is_noise(t) and t not in whitelist]
new_blacklist = sorted(existing_blacklist.union(candidates))

# Update config
config["blacklist"] = new_blacklist

# --- Save updated YAML ---
with open(NORMALIZER_CONFIG_PATH, "w") as f:
    yaml.dump(config, f)

print(f"✅ Blacklist updated. Now contains {len(new_blacklist)} entries.")
print(f"🆕 New tokens added: {len(new_blacklist) - len(existing_blacklist)}")
