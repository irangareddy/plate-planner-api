# 08_build_similar_to_edges.py

import os
import numpy as np
from neo4j import GraphDatabase
from gensim.models import Word2Vec
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
INGREDIENT_W2V_MODEL_PATH = "/Users/rangareddy/Development/OSS/plate-planner-api/src/data/models/ingredient_w2v.model"

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
        and not re.fullmatch(r'[a-z]', term.lower())
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
