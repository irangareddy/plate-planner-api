import os

import faiss
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from tqdm import tqdm

# ----------------- Paths -----------------
CONTEXT_VECTORS_PATH = "/app/src/data/processed/ingredient_substitution/context_vectors.npy"
CONTEXT_METADATA_PATH = "/app/src/data/processed/ingredient_substitution/context_metadata.csv"
FAISS_INDEX_PATH = "/app/src/data/models/ingredient_substitution/faiss_context.index"


# ----------------- Main Logic -----------------
def build_faiss_index():
    # Load data
    context_vectors = np.load(CONTEXT_VECTORS_PATH)
    metadata = pd.read_csv(CONTEXT_METADATA_PATH)

    # Data validation
    assert len(metadata) == context_vectors.shape[0], \
        f"Metadata rows ({len(metadata)}) ≠ vectors ({context_vectors.shape[0]})"

    # Normalization
    context_vectors = normalize(context_vectors, axis=1, norm="l2")

    # Index setup
    d = context_vectors.shape[1]

    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(d))
    else:
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 100)
        index.train(context_vectors)  # Only for IVF indices

    # Batch add for large datasets
    batch_size = 10000
    for i in tqdm(range(0, len(context_vectors), batch_size),
                  desc="Indexing batches"):
        index.add(context_vectors[i:i + batch_size])

    # Save
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Verification
    test_vector = context_vectors[0].reshape(1, -1)
    D, I = index.search(test_vector, 5)
    assert len(I[0]) == 5, "Index search failed basic test"

    return index


if __name__ == "__main__":
    index = build_faiss_index()
    print(f"✅ FAISS index built with {index.ntotal} vectors")
