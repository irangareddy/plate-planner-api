from typing import List
from ast import literal_eval
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
faiss.omp_set_num_threads(1)
import os

# -------------------------
# Constants (adjust paths)
# -------------------------
MODEL_NAME = 'all-MiniLM-L6-v2'
RECIPE_METADATA_PATH = '/Users/rangareddy/Development/OSS/plate-planner-api/src/data/processed/recipe_suggestion/recipe_metadata.csv'
EMBEDDINGS_PATH = '/Users/rangareddy/Development/OSS/plate-planner-api/src/data/processed/recipe_suggestion/recipe_embeddings.npy'
FAISS_INDEX_PATH = '/Users/rangareddy/Development/OSS/plate-planner-api/src/data/models/recipe_suggestion/recipe_index.faiss'

# -------------------------
# Load model + index once
# -------------------------
print("🔄 Loading model, metadata, and FAISS index...")
model = SentenceTransformer(MODEL_NAME)
metadata_df = pd.read_csv(RECIPE_METADATA_PATH)

recipe_embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
faiss.normalize_L2(recipe_embeddings)

index = faiss.read_index(FAISS_INDEX_PATH)

print(f"✅ Loaded: {len(metadata_df)} recipes, FAISS index with {index.ntotal} vectors.")

# -------------------------
# Recipe Suggestion Logic
# -------------------------
def suggest_recipes(
    ingredients: List[str],
    top_n: int = 5,
    rerank_weight: float = 0.6,
    raw_k: int = 50,
    min_overlap: int = 2
):
    """
    Suggest recipes based on semantic similarity + ingredient overlap.

    Args:
        ingredients (List[str]): User input ingredients.
        top_n (int): Number of results to return.
        rerank_weight (float): Importance of overlap vs. semantic score.
        raw_k (int): FAISS candidate pool size.
        min_overlap (int): Minimum overlapping ingredients to include.

    Returns:
        List[Dict]: Ranked recipes with scores and metadata.
    """

    query_vec = model.encode([' '.join(ingredients)])
    faiss.normalize_L2(query_vec)
    distances, indices = index.search(query_vec, raw_k)

    results = []
    for i, idx in enumerate(indices[0]):
        row = metadata_df.iloc[idx]
        try:
            recipe_ingredients = set(literal_eval(row["NER"]))
        except (ValueError, SyntaxError):
            continue

        overlap = len(set(ingredients) & recipe_ingredients)
        if overlap < min_overlap:
            continue

        sem_score = float(distances[0][i])
        overlap_score = overlap / max(len(ingredients), 1)
        combined = (1 - rerank_weight) * sem_score + rerank_weight * overlap_score

        results.append({
            "title": row["title"],
            "ingredients": row["NER"],
            "semantic_score": sem_score,
            "overlap_score": overlap_score,
            "combined_score": combined
        })

    results = sorted(results, key=lambda x: x['combined_score'], reverse=True)
    for rank, r in enumerate(results[:top_n], 1):
        r["rank"] = rank

    return results[:top_n]
