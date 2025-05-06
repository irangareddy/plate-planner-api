# 08_build_similar_to_edges.py

import re

from config.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from config.paths import DataPaths
from gensim.models import Word2Vec
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tqdm import tqdm

paths = DataPaths()
INGREDIENT_W2V_MODEL_PATH = str(paths.ingredient_w2v)

# Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Load Word2Vec model
ingredient_model = Word2Vec.load(INGREDIENT_W2V_MODEL_PATH)
TOP_N = 5

# ------------------ Utility ------------------

def is_valid_term(term):
    return (
        term.isalpha()
        and len(term) > 2
        and term.lower() not in ENGLISH_STOP_WORDS
        and not re.fullmatch(r"[a-z]", term.lower())
    )

def create_similar_relationship(tx, source, target, score):
    tx.run("""
        MATCH (a:Ingredient {name: $source})
        MATCH (b:Ingredient {name: $target})
        MERGE (a)-[r:SIMILAR_TO]->(b)
        SET r.score = $score
    """, source=source, target=target, score=score)

# ------------------ Main ------------------

def main():
    print("📦 Loading ingredient vocabulary...")
    all_ingredients = list(ingredient_model.wv.index_to_key)
    valid_ingredients = [ing for ing in all_ingredients if is_valid_term(ing)]

    print(f"Processing {len(valid_ingredients)} cleaned ingredients for SIMILAR_TO edges...")

    with driver.session() as session:
        for source in tqdm(valid_ingredients):
            try:
                similar_items = ingredient_model.wv.most_similar(source, topn=TOP_N)
                for target, similarity in similar_items:
                    if source == target or not is_valid_term(target):
                        continue
                    session.execute_write(create_similar_relationship, source, target, float(similarity))
            except KeyError:
                continue

    print("✅ SIMILAR_TO relationships built and saved to Neo4j.")

if __name__ == "__main__":
    main()
