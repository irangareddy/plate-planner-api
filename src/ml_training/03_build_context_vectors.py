# 03_build_context_vectors.py

import pandas as pd
import ast
import os
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec

# ----------------- Paths -----------------
CLEANED_ACTIONS_DATA_PATH = '/Users/rangareddy/Development/Projects/plate-planner-api/src/data/processed/cleaned_ner_actions.csv'
INGREDIENT_W2V_MODEL_PATH = '/Users/rangareddy/Development/Projects/plate-planner-api/src/data/models/ingredient_w2v.model'
ACTION_W2V_MODEL_PATH = '/Users/rangareddy/Development/Projects/plate-planner-api/src/data/models/action_w2v.model'
CONTEXT_VECTORS_PATH = '/Users/rangareddy/Development/Projects/plate-planner-api/src/data/processed/context_vectors.npy'
CONTEXT_METADATA_PATH = '/Users/rangareddy/Development/Projects/plate-planner-api/src/data/processed/context_metadata.csv'

# ----------------- Setup -----------------
tqdm.pandas()

# ----------------- Step 1: Load dataset -----------------
print("Loading cleaned dataset with actions...")
df = pd.read_csv(CLEANED_ACTIONS_DATA_PATH)


# ----------------- Step 2: Parse fields -----------------
def safe_literal_eval(x):
    try:
        return ast.literal_eval(x)
    except:
        return []


print("Parsing ner_list_cleaned and actions...")
df['ner_list_cleaned'] = df['ner_list_cleaned'].progress_apply(safe_literal_eval)
df['actions'] = df['actions'].progress_apply(safe_literal_eval)

# ----------------- Step 3: Train Word2Vec Models -----------------

# Prepare training data
ingredient_sentences = df['ner_list_cleaned'].tolist()
action_sentences = df['actions'].tolist()

print("Training Word2Vec on ingredients...")
ingredient_model = Word2Vec(
    sentences=ingredient_sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1  # skip-gram
)
os.makedirs(os.path.dirname(INGREDIENT_W2V_MODEL_PATH), exist_ok=True)
ingredient_model.save(INGREDIENT_W2V_MODEL_PATH)

print("Training Word2Vec on actions...")
action_model = Word2Vec(
    sentences=action_sentences,
    vector_size=50,
    window=3,
    min_count=1,
    workers=4,
    sg=1
)
action_model.save(ACTION_W2V_MODEL_PATH)


# ----------------- Step 4: Build Context Vectors -----------------

def build_context_vector(ingredients, actions, ingredient_model, action_model):
    ing_vecs = [ingredient_model.wv[word] for word in ingredients if word in ingredient_model.wv]
    act_vecs = [action_model.wv[word] for word in actions if word in action_model.wv]

    if not ing_vecs and not act_vecs:
        return np.zeros(ingredient_model.vector_size + action_model.vector_size)

    # Average separately, then concatenate
    if ing_vecs:
        ing_vec = np.mean(ing_vecs, axis=0)
    else:
        ing_vec = np.zeros(ingredient_model.vector_size)

    if act_vecs:
        act_vec = np.mean(act_vecs, axis=0)
    else:
        act_vec = np.zeros(action_model.vector_size)

    return np.concatenate([ing_vec, act_vec])


print("Building context vectors...")
context_vectors = np.vstack(df.progress_apply(
    lambda row: build_context_vector(row['ner_list_cleaned'], row['actions'], ingredient_model, action_model),
    axis=1
))

print(f"Saving context vectors to {CONTEXT_VECTORS_PATH}...")
np.save(CONTEXT_VECTORS_PATH, context_vectors)

# Save metadata (title + link to context vector index)
print(f"Saving context metadata to {CONTEXT_METADATA_PATH}...")
df[['title']].to_csv(CONTEXT_METADATA_PATH, index=False)

print("✅ Done! Context vectors and metadata saved.")
