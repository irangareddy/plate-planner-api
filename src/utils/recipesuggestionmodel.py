from ast import literal_eval

import faiss
import numpy as np
import pandas as pd
from src.config.paths import DataPaths
from sentence_transformers import SentenceTransformer

# -------------------------
# Constants
# -------------------------
faiss.omp_set_num_threads(1)

MODEL_NAME = "all-MiniLM-L6-v2"
paths = DataPaths()

RECIPE_METADATA_PATH = paths.recipe_metadata
EMBEDDINGS_PATH = paths.recipe_embeddings
FAISS_INDEX_PATH = paths.recipe_faiss_index

# -------------------------
# Load model + index once
# -------------------------
print("🔄 Loading model, metadata, and FAISS index...")

model = SentenceTransformer(MODEL_NAME)
metadata_df = pd.read_csv(RECIPE_METADATA_PATH)

recipe_embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
faiss.normalize_L2(recipe_embeddings)

index = faiss.read_index(str(FAISS_INDEX_PATH))

print(f"✅ Loaded: {len(metadata_df)} recipes, FAISS index with {index.ntotal} vectors.")

# -------------------------
# Recipe Suggestion Logic
# -------------------------
def suggest_recipes(
    ingredients: list[str],
    top_n: int = 5,
    rerank_weight: float = 0.6,
    raw_k: int = 50,
    min_overlap: int = 2
) -> list[dict]:
    """Suggest recipes based on semantic similarity + ingredient overlap."""
    query_vec = model.encode([" ".join(ingredients)])
    faiss.normalize_L2(query_vec)
    distances, indices = index.search(query_vec, raw_k)

    # Convert input list to a set for faster lookups:
    input_set = set(ingredients)

    results: list[dict] = []
    for dist_i, idx in enumerate(indices[0]):
        row = metadata_df.iloc[idx]
        try:
            raw_list = literal_eval(row["NER"])
        except (ValueError, SyntaxError):
            continue

        # 1) dedupe the recipe's ingredient list, preserving order
        seen = set()
        unique_full_list = []
        for ing in raw_list:
            if ing not in seen:
                unique_full_list.append(ing)
                seen.add(ing)

        # 2) find the intersection as a set (no duplicates)
        overlap_set = unique_full_list and (input_set & set(unique_full_list))
        if len(overlap_set) < min_overlap:
            continue

        # 3) compute semantic similarity in [0,1]
        sem_score = float(distances[0][dist_i])            # now already cosine ∈[-1,1]
        sem_score = max(0.0, min(1.0, sem_score))          # drop negatives, clamp to [0,1]

        # 4) compute overlap in [0,1]
        overlap_score = len(overlap_set) / max(len(ingredients), 1)

        # 5) convex blend & clamp combined_score in [0,1]
        combined_score = (1 - rerank_weight)*sem_score + rerank_weight*overlap_score
        combined_score = max(0.0, min(1.0, combined_score))

        results.append({
            "title": row["title"],
            "ingredients": [i for i in unique_full_list if i in overlap_set],
            "semantic_score": sem_score,
            "overlap_score": overlap_score,
            "combined_score": combined_score,
        })

    # sort, rank, and return top_n
    results = sorted(results, key=lambda x: x["combined_score"], reverse=True)
    for rank, r in enumerate(results[:top_n], start=1):
        r["rank"] = rank

    return results[:top_n]
