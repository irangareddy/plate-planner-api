# src/api/fastapi_app.py
import ast

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import faiss
import os
from gensim.models import Word2Vec


# ----------------- Paths -----------------
CONTEXT_VECTORS_PATH = '/Users/rangareddy/Development/Projects/plate-planner-api/src/data/processed/context_vectors.npy'
CONTEXT_METADATA_PATH = '/Users/rangareddy/Development/Projects/plate-planner-api/src/data/processed/context_metadata.csv'
FAISS_INDEX_PATH = '/Users/rangareddy/Development/Projects/plate-planner-api/src/data/models/faiss_context.index'

INGREDIENT_W2V_MODEL_PATH = '/Users/rangareddy/Development/Projects/plate-planner-api/src/data/models/ingredient_w2v.model'
ACTION_W2V_MODEL_PATH = '/Users/rangareddy/Development/Projects/plate-planner-api/src/data/models/action_w2v.model'

# ----------------- Initialize FastAPI -----------------
app = FastAPI(title="Plate Planner API", version="0.1")

# ----------------- Load Models and Data at Startup -----------------
print("Loading FAISS index and metadata...")

context_vectors = np.load(CONTEXT_VECTORS_PATH)
metadata = pd.read_csv(CONTEXT_METADATA_PATH)
d = context_vectors.shape[1]

index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading Word2Vec models...")
ingredient_model = Word2Vec.load(INGREDIENT_W2V_MODEL_PATH)
action_model = Word2Vec.load(ACTION_W2V_MODEL_PATH)
print("✅ Word2Vec models loaded.")

print("✅ Models and Data loaded.")


class SubstitutionRequest(BaseModel):
    ingredient: str
    actions: List[str] = []
    top_k: int = 5

# ----------------- Pydantic Models -----------------
class RecommendationRequest(BaseModel):
    context_vector: List[float]
    top_k: int = 5

# ----------------- API Endpoints -----------------
@app.post("/recommend_recipes")
async def recommend_recipes(req: RecommendationRequest):
    vector = np.array(req.context_vector, dtype=np.float32).reshape(1, -1)

    D, I = index.search(vector, k=req.top_k)
    results = []

    for idx in I[0]:
        if idx < len(metadata):
            results.append({
                "title": metadata.iloc[idx]['title'],
                "recipe_id": int(idx)  # optional: could be more metadata later
            })

    return {"recommendations": results}


# def build_context_vector_from_words(ingredient, actions):
#     ing_vec = np.zeros(ingredient_model.vector_size)
#     act_vec = np.zeros(action_model.vector_size)
#
#     if ingredient in ingredient_model.wv:
#         ing_vec = ingredient_model.wv[ingredient]
#
#     act_vecs = [action_model.wv[action] for action in actions if action in action_model.wv]
#     if act_vecs:
#         act_vec = np.mean(act_vecs, axis=0)
#
#     combined = np.concatenate([ing_vec, act_vec])
#
#     return combined.reshape(1, -1)

def build_context_vector_from_words(ingredient, actions, ingredient_weight=0.6, action_weight=0.4):
    ing_vec = np.zeros(ingredient_model.vector_size)
    act_vec = np.zeros(action_model.vector_size)

    if ingredient in ingredient_model.wv:
        ing_vec = ingredient_model.wv[ingredient]

    act_vecs = [action_model.wv[action] for action in actions if action in action_model.wv]
    if act_vecs:
        act_vec = np.mean(act_vecs, axis=0)

    combined_ing = ing_vec * ingredient_weight
    combined_act = act_vec * action_weight

    combined = np.concatenate([combined_ing, combined_act])

    return combined.reshape(1, -1)




# ----------------- New API Endpoint -----------------
@app.post("/substitute_ingredient")
async def substitute_ingredient(req: SubstitutionRequest):
    context_vec = build_context_vector_from_words(req.ingredient, req.actions)

    D, I = index.search(context_vec, k=req.top_k)

    # Load full recipe ingredients
    full_df = pd.read_csv('/Users/rangareddy/Development/Projects/plate-planner-api/src/data/processed/cleaned_ner_actions.csv')
    full_df['ner_list_cleaned'] = full_df['ner_list_cleaned'].apply(ast.literal_eval)

    all_ingredients = []

    for idx in I[0]:
        if idx < len(full_df):
            ingredients = full_df.iloc[idx]['ner_list_cleaned']
            all_ingredients.extend(ingredients)

    # Filter: Remove original ingredient itself
    filtered_ingredients = [ing for ing in all_ingredients if ing.lower() != req.ingredient.lower()]

    # Count most common ingredients
    from collections import Counter
    substitute_counts = Counter(filtered_ingredients)

    # Top 5 suggested substitutes
    substitutes = [ing for ing, count in substitute_counts.most_common(5)]

    return {"substitutes": substitutes}


@app.get("/")
async def root():
    return {"message": "Plate Planner API is running!"}
