import ast
import os
import ssl

import nltk
import pandas as pd
import spacy
from nltk.corpus import verbnet as vn
from tqdm import tqdm

# ----------------- Paths -----------------
CLEANED_DATA_PATH = "/data/processed/ingredient_substitution/cleaned_ner.csv"
ACTIONS_DATA_PATH = "/data/processed/ingredient_substitution/cleaned_ner_actions.csv"

# ----------------- Setup -----------------
tqdm.pandas()

# Bypass SSL verification for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# ----------------- NLP Initialization -----------------
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])


# ----------------- VerbNet Configuration -----------------
def get_valid_culinary_classes():
    """Dynamically find valid VerbNet classes containing known cooking verbs"""
    try:
        nltk.data.find("corpora/verbnet")
    except LookupError:
        nltk.download("verbnet", quiet=True)

    # Required NLTK resources
    nltk.download("wordnet", quiet=True)

    # Known culinary verbs to look for in VerbNet classes
    target_verbs = {"bake", "boil", "fry", "grill", "roast", "steam", "stir"}
    valid_classes = set()

    for class_id in vn.classids():
        try:
            class_verbs = set(map(str.lower, vn.lemmas(class_id)))
            if class_verbs & target_verbs:
                valid_classes.add(class_id)
        except ValueError:
            continue

    return list(valid_classes)


def initialize_culinary_verbs():
    """Initialize comprehensive culinary verb set with validation"""
    culinary_classes = get_valid_culinary_classes()
    culinary_verbs = set()

    for cls in culinary_classes:
        try:
            culinary_verbs.update(map(str.lower, vn.lemmas(cls)))
        except ValueError as e:
            print(f"⚠️ Skipping invalid class {cls}: {e!s}")

    # Extended manual culinary verbs
    additional_verbs = {
        "chop", "dice", "mince", "grate", "whisk", "knead", "season",
        "marinate", "peel", "julienne", "crush", "score", "deglaze",
        "temper", "proof", "ferment", "emulsify", "brine", "sear",
        "sauté", "simmer", "poach", "braise", "blanch", "glaze", "baste"
    }

    culinary_verbs.update(additional_verbs)
    return culinary_verbs


CULINARY_VERBS = initialize_culinary_verbs()


# ----------------- Processing Functions -----------------
def safe_literal_eval(x):
    """Safely parse stringified lists"""
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []


def extract_culinary_actions(directions_series):
    """Batch process directions to extract culinary verbs"""
    texts = []

    print("\nPreprocessing directions...")
    for directions in tqdm(directions_series, desc="Preprocessing"):
        if isinstance(directions, list):
            full_text = " ".join(directions)
        elif isinstance(directions, str):
            try:
                steps = ast.literal_eval(directions)
                full_text = " ".join(steps) if isinstance(steps, list) else directions
            except:
                full_text = directions
        else:
            full_text = str(directions)
        texts.append(full_text)

    print("\nExtracting culinary actions...")
    actions_list = []

    for doc in tqdm(nlp.pipe(texts, batch_size=128, n_process=4),
                    total=len(texts),
                    desc="Processing"):
        verbs = set()
        for token in doc:
            if token.pos_ == "VERB":
                lemma = token.lemma_.lower()
                if lemma in CULINARY_VERBS:
                    verbs.add(lemma)
        actions_list.append(list(verbs))

    return actions_list


# ----------------- Main Execution -----------------
if __name__ == "__main__":
    print(f"\nLoaded {len(CULINARY_VERBS)} culinary verbs")
    print("Sample verbs:", sorted(CULINARY_VERBS)[:15], "...\n")

    df = pd.read_csv(CLEANED_DATA_PATH)

    print("Parsing NER data...")
    df["ner_list_cleaned"] = df["ner_list_cleaned"].progress_apply(safe_literal_eval)

    print("\nExtracting culinary actions...")
    df["actions"] = extract_culinary_actions(df["directions"])

    os.makedirs(os.path.dirname(ACTIONS_DATA_PATH), exist_ok=True)
    df[["title", "ner_list_cleaned", "directions", "actions"]].to_csv(ACTIONS_DATA_PATH, index=False)

    print(f"\n✅ Success! Saved culinary actions to {ACTIONS_DATA_PATH}")
    print("Sample output:", df["actions"].head(3).tolist())
