import argparse
import re

import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- Paths -----------------
INGREDIENT_W2V_MODEL_PATH = "/data/models/ingredient_substitution/ingredient_w2v.model"
ACTION_W2V_MODEL_PATH = "/data/models/ingredient_substitution/action_w2v.model"

# ----------------- Parameters -----------------
ING_WEIGHT = 0.8
ACT_WEIGHT = 0.2
TOP_K = 5

# ----------------- Load Models -----------------
print("📦 Loading Word2Vec models...")
ingredient_model = Word2Vec.load(INGREDIENT_W2V_MODEL_PATH)
action_model = Word2Vec.load(ACTION_W2V_MODEL_PATH)

# ----------------- Noise Filtering -----------------
def is_valid_ingredient(word):
    # Filters out stopwords, very short tokens, or non-alphabetic
    return (
        word.isalpha() and
        len(word) > 2 and
        word.lower() not in ENGLISH_STOP_WORDS and
        not re.fullmatch(r"[a-z]", word.lower())
    )

# ----------------- Context Vector Builder -----------------
def build_vector(ingredients, actions, substitute=None):
    if substitute:
        ingredients = [ing if ing != substitute[0] else substitute[1] for ing in ingredients]

    ing_vecs = [ingredient_model.wv[word] for word in ingredients if word in ingredient_model.wv]
    act_vecs = [action_model.wv[word] for word in actions if word in action_model.wv]

    ing_vec = np.mean(ing_vecs, axis=0) if ing_vecs else np.zeros(ingredient_model.vector_size)
    act_vec = np.mean(act_vecs, axis=0) if act_vecs else np.zeros(action_model.vector_size)

    return np.concatenate([ing_vec * ING_WEIGHT, act_vec * ACT_WEIGHT])

# ----------------- Substitution Suggestion -----------------
def suggest_substitute(original_ingredient, ingredients, actions, topk=TOP_K):
    if original_ingredient not in ingredients:
        print(f"⚠️ '{original_ingredient}' not found in recipe.")
        return []

    original_vec = build_vector(ingredients, actions)
    candidates = [
        w for w in ingredient_model.wv.index_to_key
        if w != original_ingredient and is_valid_ingredient(w)
    ]

    scores = []
    for candidate in candidates:
        new_vec = build_vector(ingredients, actions, substitute=(original_ingredient, candidate))
        score = cosine_similarity([original_vec], [new_vec])[0][0]
        scores.append((candidate, score))

    top_subs = sorted(scores, key=lambda x: x[1], reverse=True)[:topk]

    print(f"\n🔍 Top {topk} substitutes for '{original_ingredient}' in this context:\n")
    for i, (sub, score) in enumerate(top_subs):
        print(f"{i+1}. {sub} (similarity: {score:.4f})")
    return top_subs

# ----------------- CLI -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingredient", type=str, required=True, help="Ingredient to substitute")
    args = parser.parse_args()

    # Example test case
    test_recipe = {
        "ingredients": ["butter", "sugar", "flour", "vanilla"],
        "actions": ["mix", "bake"]
    }

    suggest_substitute(args.ingredient, test_recipe["ingredients"], test_recipe["actions"])
