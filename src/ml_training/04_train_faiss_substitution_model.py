# 04_train_faiss_substitution_model.py

import numpy as np
import pandas as pd
import faiss
import os

# ----------------- Paths -----------------
CONTEXT_VECTORS_PATH = '/Users/rangareddy/Development/Projects/plate-planner-api/src/data/processed/context_vectors.npy'
CONTEXT_METADATA_PATH = '/Users/rangareddy/Development/Projects/plate-planner-api/src/data/processed/context_metadata.csv'
FAISS_INDEX_PATH = '/Users/rangareddy/Development/Projects/plate-planner-api/src/data/models/faiss_context.index'

# ----------------- Step 1: Load Context Vectors -----------------
print("Loading context vectors...")
context_vectors = np.load(CONTEXT_VECTORS_PATH)

print("Loading metadata...")
metadata = pd.read_csv(CONTEXT_METADATA_PATH)

# ----------------- Step 2: Build FAISS Index -----------------
d = context_vectors.shape[1]  # dimension of context vectors

print(f"Building FAISS index with vector dimension {d}...")

index = faiss.IndexFlatL2(d)  # L2 (euclidean) distance index
index.add(context_vectors)

print(f"Total vectors added: {index.ntotal}")

# ----------------- Step 3: Save FAISS Index -----------------
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

faiss.write_index(index, FAISS_INDEX_PATH)

print(f"✅ FAISS index saved at {FAISS_INDEX_PATH}")

# ----------------- Step 4: Quick Search Test -----------------
print("Running a quick nearest neighbor search...")

D, I = index.search(context_vectors[:1], k=5)  # Search top-5 similar to first recipe

print("Most similar recipes to the first recipe:")
for idx in I[0]:
    print(f"Title: {metadata.iloc[idx]['title']}")

print("✅ Substitution system ready.")
