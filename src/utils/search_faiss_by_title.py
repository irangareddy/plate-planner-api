import argparse
import os
import random
from datetime import datetime

import faiss
import numpy as np
import pandas as pd

# ----------------- Paths -----------------
CONTEXT_VECTORS_PATH = "/app/src/data/processed/ingredient_substitution/context_vectors.npy"
CONTEXT_METADATA_PATH = "/app/src/data/processed/ingredient_substitution/context_metadata.csv"
FAISS_INDEX_PATH = "/app/src/data/models/ingredient_substitution/faiss_context.index"
RESULT_PATH = f"app/src/data/results/hit_at_5k_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# ----------------- Load Assets -----------------
print("Loading metadata and vectors...")
metadata = pd.read_csv(CONTEXT_METADATA_PATH)
vectors = np.load(CONTEXT_VECTORS_PATH)

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

# ----------------- Mode: Interactive Search -----------------
def search_by_title(title: str, k: int = 5):
    matches = metadata[metadata["title"].str.lower().str.contains(title.lower())]

    if matches.empty:
        print(f"❌ No matches found for '{title}'")
        return

    for i, (_, row) in enumerate(matches.iterrows()):
        print(f"[{i}] {row['title']}")

    idx = int(input(f"\nEnter index of recipe to search (0–{len(matches)-1}): "))
    selected_index = matches.index[idx]

    print(f"\n🔍 Searching similar recipes for: {metadata.iloc[selected_index]['title']}\n")

    D, I = index.search(vectors[selected_index:selected_index+1], k)

    print("Top similar recipes:")
    for j, recipe_idx in enumerate(I[0]):
        title = metadata.iloc[recipe_idx]["title"]
        distance = D[0][j]
        print(f"{j+1}. {title} (distance: {distance:.4f})")

# ----------------- Mode: Hit@K Evaluation -----------------
def evaluate_hit_at_k(n_samples: int = 100, k: int = 5):
    print(f"\n🔍 Running Hit@{k} evaluation on {n_samples} random samples...")
    sample_indices = random.sample(range(len(vectors)), n_samples)
    hits = 0
    lines = []

    for i, idx in enumerate(sample_indices):
        title = metadata.iloc[idx]["title"]
        vec = vectors[idx:idx+1]
        D, I = index.search(vec, k + 1)

        match_indices = [i for i in I[0] if i != idx][:k]
        match_titles = [metadata.iloc[i]["title"] for i in match_indices]

        hit = any(title.lower() in mt.lower() or mt.lower() in title.lower() for mt in match_titles)
        hits += int(hit)

        lines.append(f"\n[{i+1}] Query: {title}")
        lines.extend([f"  → {rank+1}. {mt} (distance: {D[0][j+1]:.4f})"
                      for rank, (j, mt) in enumerate(zip(range(k), match_titles, strict=False))])
        lines.append(f"  ✅ HIT: {hit}\n")

    score = hits / n_samples
    lines.append(f"\n=== Hit@{k} Score: {score:.2%} ===")

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"\n✅ Evaluation complete. Hit@{k} = {score:.2%}")
    print(f"📄 Results saved to {RESULT_PATH}")

# ----------------- Entry Point -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, help="(Optional) Recipe title to search for manually")
    parser.add_argument("--topk", type=int, default=5, help="Top-K similar results to return")
    parser.add_argument("--evaluate", action="store_true", help="Run Hit@K evaluation instead of interactive search")
    parser.add_argument("--samples", type=int, default=100, help="Number of random samples to evaluate Hit@K")
    args = parser.parse_args()

    if args.evaluate:
        evaluate_hit_at_k(n_samples=args.samples, k=args.topk)
    elif args.title:
        search_by_title(args.title, args.topk)
    else:
        print("⚠️ Please provide either --title for search or --evaluate for batch testing.")
