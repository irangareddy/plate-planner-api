# 03_build_context_vectors.py (Enhanced)
import ast
import os

import numpy as np
import pandas as pd
from data.processed.substitution_config import SubstitutionConfig
from gensim.models import Word2Vec
from tqdm import tqdm

# ----------------- Paths -----------------
CLEANED_ACTIONS_PATH = "/data/processed/ingredient_substitution/cleaned_ner_actions.csv"
INGREDIENT_W2V_MODEL_PATH = "/data/models/ingredient_substitution/ingredient_w2v.model"
ACTION_W2V_MODEL_PATH = "/data/models/ingredient_substitution/action_w2v.model"
CONTEXT_VECTOR_PATH = "/data/processed/ingredient_substitution/context_vectors.npy"
CONTEXT_META_PATH = "/data/processed/ingredient_substitution/context_metadata.csv"

# ----------------- Parameters -----------------
ING_WEIGHT = SubstitutionConfig.INGREDIENT_WEIGHT
ACT_WEIGHT = SubstitutionConfig.ACTION_WEIGHT


# ----------------- Helper Functions -----------------
def safe_literal_eval(x):
    """Safely parse string representations of lists"""
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []


def create_directory(path):
    """Ensure output directory exists"""
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ----------------- Main Pipeline -----------------
def main():
    # --- Step 1: Load and Prepare Data ---
    print("📦 Loading dataset...")
    df = pd.read_csv(CLEANED_ACTIONS_PATH)

    print("🔍 Parsing list columns...")
    tqdm.pandas()
    df["ner_list_cleaned"] = df["ner_list_cleaned"].progress_apply(safe_literal_eval)
    df["actions"] = df["actions"].progress_apply(safe_literal_eval)

    # --- Step 2: Train Models ---
    print("🧠 Training Ingredient Word2Vec...")
    ingredient_model = Word2Vec(
        sentences=df["ner_list_cleaned"].tolist(),
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        sg=1
    )
    create_directory(INGREDIENT_W2V_MODEL_PATH)
    ingredient_model.save(INGREDIENT_W2V_MODEL_PATH)

    print("🧠 Training Action Word2Vec...")
    action_model = Word2Vec(
        sentences=df["actions"].tolist(),
        vector_size=50,
        window=3,
        min_count=1,
        workers=4,
        sg=1
    )
    create_directory(ACTION_W2V_MODEL_PATH)
    action_model.save(ACTION_W2V_MODEL_PATH)

    # --- Step 3: Build Context Vectors ---
    def build_context_vector(ingredients, actions):
        """Create weighted context vector with error handling"""
        try:
            ing_vecs = [ingredient_model.wv[w] for w in ingredients if w in ingredient_model.wv]
            act_vecs = [action_model.wv[w] for w in actions if w in action_model.wv]

            if not ing_vecs and not act_vecs:
                return np.zeros(ingredient_model.vector_size + action_model.vector_size)

            ing_vec = np.mean(ing_vecs, axis=0) * ING_WEIGHT if ing_vecs else np.zeros(ingredient_model.vector_size)
            act_vec = np.mean(act_vecs, axis=0) * ACT_WEIGHT if act_vecs else np.zeros(action_model.vector_size)

            return np.concatenate([ing_vec, act_vec])

        except Exception as e:
            print(f"Error building vector: {e}")
            return np.zeros(ingredient_model.vector_size + action_model.vector_size)

    print("⚙️ Building context vectors...")
    context_vectors = df.progress_apply(
        lambda row: build_context_vector(row["ner_list_cleaned"], row["actions"]),
        axis=1
    )
    context_matrix = np.vstack(context_vectors)

    # --- Step 4: Save Output ---
    print("💾 Saving outputs...")
    create_directory(CONTEXT_VECTOR_PATH)
    np.save(CONTEXT_VECTOR_PATH, context_matrix)

    create_directory(CONTEXT_META_PATH)
    df[["title"]].to_csv(CONTEXT_META_PATH, index=False)

    print("✅ Done! Context vectors + metadata ready for FAISS/similarity search.")


if __name__ == "__main__":
    main()
