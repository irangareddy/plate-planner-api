from ast import literal_eval

import faiss
import numpy as np
import pandas as pd
from config.paths import DataPaths  # 👈 centralized path handling
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
):
    """Suggest recipes based on semantic similarity + ingredient overlap."""
    query_vec = model.encode([" ".join(ingredients)])
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

    results = sorted(results, key=lambda x: x["combined_score"], reverse=True)
    for rank, r in enumerate(results[:top_n], 1):
        r["rank"] = rank

    return results[:top_n]
