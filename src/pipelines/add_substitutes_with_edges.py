import ast
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ------------------ Config ------------------
CLEANED_ACTIONS_PATH = "/data/processed/ingredient_substitution/cleaned_ner_actions.csv"
INGREDIENT_W2V_PATH = "/data/models/ingredient_substitution/ingredient_w2v.model"
ACTION_W2V_PATH = "/data/models/ingredient_substitution/action_w2v.model"
EXPORT_PATH = f"/mnt/data/substitution_edges_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

TOP_K = 5
SIM_THRESHOLD = 0.85
ING_WEIGHT = 0.9
ACT_WEIGHT = 0.1

# ------------------ Helpers ------------------
def is_valid_token(token):
    return (
        token.isalpha() and len(token) > 2 and
        token.lower() not in ENGLISH_STOP_WORDS and
        not re.fullmatch(r"[a-z]", token.lower())
    )

def build_vector(tokens, vecs, dim):
    vec_list = [vecs.get(w, np.zeros(dim)) for w in tokens]
    return np.mean(vec_list, axis=0) if vec_list else np.zeros(dim)

def process_row(row, ingredient_vecs, action_vecs, ingredient_model, dim_ing, dim_act):
    ingredients = [w for w in row["ner_list_cleaned"] if is_valid_token(w)]
    actions = [w for w in row["actions"] if is_valid_token(w)]
    if len(ingredients) < 2:
        return []

    orig_ing_vec = build_vector(ingredients, ingredient_vecs, dim_ing) * ING_WEIGHT
    orig_act_vec = build_vector(actions, action_vecs, dim_act) * ACT_WEIGHT
    original_vec = np.concatenate([orig_ing_vec, orig_act_vec])

    results = []
    for ing in ingredients:
        if ing not in ingredient_model.wv:
            continue
        try:
            similar = ingredient_model.wv.most_similar(ing, topn=TOP_K)
        except KeyError:
            continue
        for candidate, _ in similar:
            if candidate == ing or not is_valid_token(candidate):
                continue
            substituted = [candidate if x == ing else x for x in ingredients]
            sub_ing_vec = build_vector(substituted, ingredient_vecs, dim_ing) * ING_WEIGHT
            sub_vec = np.concatenate([sub_ing_vec, orig_act_vec])  # reuse original action vector
            sim = cosine_similarity([original_vec], [sub_vec])[0][0]
            if sim >= SIM_THRESHOLD:
                results.append((ing, candidate, round(sim, 4)))
    return results

def process_chunk(chunk, ingredient_vecs, action_vecs, ingredient_model, dim_ing, dim_act):
    return [
        res for row in chunk.itertuples(index=False)
        for res in process_row(row._asdict(), ingredient_vecs, action_vecs, ingredient_model, dim_ing, dim_act)
    ]

# ------------------ Main ------------------
def main():
    start_all = time.time()

    # Step 1: Load and parse data
    start = time.time()
    print("📦 Loading and parsing data...")
    df = pd.read_csv(CLEANED_ACTIONS_PATH)
    tqdm.pandas(desc="Parsing ner_list_cleaned")
    df["ner_list_cleaned"] = df["ner_list_cleaned"].progress_apply(ast.literal_eval)
    tqdm.pandas(desc="Parsing actions")
    df["actions"] = df["actions"].progress_apply(ast.literal_eval)
    print(f"⏱️ Loaded and parsed in {round(time.time() - start, 2)} seconds")

    # Step 2: Load models and vectorize
    start = time.time()
    print("🧠 Loading models + precomputing vectors...")
    ingredient_model = Word2Vec.load(INGREDIENT_W2V_PATH)
    action_model = Word2Vec.load(ACTION_W2V_PATH)
    dim_ing = ingredient_model.vector_size
    dim_act = action_model.vector_size

    unique_ingredients = {t for lst in df["ner_list_cleaned"] for t in lst if is_valid_token(t)}
    unique_actions = {t for lst in df["actions"] for t in lst if is_valid_token(t)}

    ingredient_vecs = {t: ingredient_model.wv[t] for t in tqdm(unique_ingredients, desc="Ingredient Vecs") if t in ingredient_model.wv}
    action_vecs = {t: action_model.wv[t] for t in tqdm(unique_actions, desc="Action Vecs") if t in action_model.wv}
    print(f"⏱️ Vector cache built in {round(time.time() - start, 2)} seconds")

    # Step 3: Parallel substitution computation
    start = time.time()
    print("⚙️ Running parallel substitution generation...")
    NUM_CHUNKS = 200
    chunks = np.array_split(df, NUM_CHUNKS)
    flattened = []

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [
            executor.submit(process_chunk, chunk, ingredient_vecs, action_vecs,
                            ingredient_model, dim_ing, dim_act)
            for chunk in chunks
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc="🔄 Processing chunks"):
            flattened.extend(f.result())
    print(f"⏱️ Substitution processing done in {round(time.time() - start, 2)} seconds")

    # Step 4: Save results
    start = time.time()
    print("💾 Writing CSV file...")
    df_edges = pd.DataFrame(flattened, columns=["source", "target", "score"])
    df_edges.to_csv(EXPORT_PATH, index=False)
    print(f"📄 CSV saved in {round(time.time() - start, 2)} seconds")

    # Step 5: Display and finish
    import ace_tools as tools
    tools.display_dataframe_to_user(name="Optimized Substitution Edges", dataframe=df_edges)
    print(f"✅ Done. Total runtime: {round(time.time() - start_all, 2)} seconds")
    print(f"📂 Output path: {EXPORT_PATH}")

if __name__ == "__main__":
    main()
