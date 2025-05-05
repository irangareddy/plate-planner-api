import os
import ast
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from neo4j import GraphDatabase
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import time

# ------------------ Paths ------------------
CLEANED_ACTIONS_PATH = '/Users/rangareddy/Development/OSS/plate-planner-api/src/data/processed/cleaned_ner_actions.csv'
INGREDIENT_W2V_PATH = '/Users/rangareddy/Development/OSS/plate-planner-api/src/data/models/ingredient_w2v.model'
ACTION_W2V_PATH = '/Users/rangareddy/Development/OSS/plate-planner-api/src/data/models/action_w2v.model'

# ------------------ Neo4j Setup ------------------
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ------------------ Parameters ------------------
TOP_K = 5
SIM_THRESHOLD = 0.85
ING_WEIGHT = 0.9
ACT_WEIGHT = 0.1
BATCH_SIZE = 100  # Number of substitutions to commit at once

# ------------------ Load Data + Models ------------------
print("📦 Loading dataset and models...")
df = pd.read_csv(CLEANED_ACTIONS_PATH)
df['ner_list_cleaned'] = df['ner_list_cleaned'].apply(ast.literal_eval)
df['actions'] = df['actions'].apply(ast.literal_eval)

ingredient_model = Word2Vec.load(INGREDIENT_W2V_PATH)
action_model = Word2Vec.load(ACTION_W2V_PATH)

# ------------------ Utility ------------------
def is_valid_token(token):
    return (
        token.isalpha()
        and len(token) > 2
        and token.lower() not in ENGLISH_STOP_WORDS
        and not re.fullmatch(r"[a-z]", token.lower())
    )

def build_vector(ingredients, actions):
    ing_vecs = [ingredient_model.wv[w] for w in ingredients if w in ingredient_model.wv]
    act_vecs = [action_model.wv[w] for w in actions if w in action_model.wv]

    ing_vec = np.mean(ing_vecs, axis=0) if ing_vecs else np.zeros(ingredient_model.vector_size)
    act_vec = np.mean(act_vecs, axis=0) if act_vecs else np.zeros(action_model.vector_size)

    return np.concatenate([ing_vec * ING_WEIGHT, act_vec * ACT_WEIGHT])

def write_batch_to_neo4j(tx, records):
    tx.run("""
        UNWIND $rows AS row
        MATCH (a:Ingredient {name: row.source})
        MATCH (b:Ingredient {name: row.target})
        MERGE (a)-[r:SUBSTITUTES_WITH]->(b)
        SET r.score = row.score
    """, rows=records)

# ------------------ Main Logic ------------------
def main():
    print("🔍 Generating context-aware substitutions...\n")
    start = time.time()
    batch = []

    with driver.session() as session:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            ingredients = [w for w in row['ner_list_cleaned'] if is_valid_token(w)]
            actions = [w for w in row['actions'] if is_valid_token(w)]

            if len(ingredients) < 2:
                continue

            original_vec = build_vector(ingredients, actions)

            for ing in ingredients:
                if ing not in ingredient_model.wv:
                    continue

                try:
                    similar_candidates = ingredient_model.wv.most_similar(ing, topn=TOP_K)
                except KeyError:
                    continue

                for candidate, _ in similar_candidates:
                    if candidate == ing or not is_valid_token(candidate):
                        continue

                    substituted = [candidate if x == ing else x for x in ingredients]
                    sub_vec = build_vector(substituted, actions)
                    similarity = cosine_similarity([original_vec], [sub_vec])[0][0]

                    if similarity >= SIM_THRESHOLD:
                        batch.append({
                            "source": ing,
                            "target": candidate,
                            "score": round(similarity, 4)
                        })

                    if len(batch) >= BATCH_SIZE:
                        session.execute_write(write_batch_to_neo4j, batch)
                        batch.clear()

        # Final batch
        if batch:
            session.execute_write(write_batch_to_neo4j, batch)

    print(f"\n✅ Done! All substitutions written to Neo4j.")
    print(f"⏱️ Total time taken: {round(time.time() - start, 2)} seconds")

if __name__ == "__main__":
    main()
