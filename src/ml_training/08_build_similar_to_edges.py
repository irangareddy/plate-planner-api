# 08_build_similar_to_edges.py

import os
import numpy as np
from neo4j import GraphDatabase
from gensim.models import Word2Vec
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

INGREDIENT_W2V_MODEL_PATH = "/Users/rangareddy/Development/Projects/plate-planner-api/src/data/models/ingredient_w2v.model"

# Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Load Ingredient Embedding Model
ingredient_model = Word2Vec.load(INGREDIENT_W2V_MODEL_PATH)

# How many similar ingredients to link
TOP_N = 5


def create_similar_relationships(tx, source, target, score):
    tx.run("""
        MATCH (a:Ingredient {name: $source})
        MATCH (b:Ingredient {name: $target})
        MERGE (a)-[r:SIMILAR_TO]->(b)
        SET r.score = $score
    """, source=source, target=target, score=score)


def main():
    with driver.session() as session:
        all_ingredients = list(ingredient_model.wv.index_to_key)

        print(f"Processing {len(all_ingredients)} ingredients for SIMILAR_TO relationships...")

        for ing in tqdm(all_ingredients):
            try:
                # Get Top-N similar ingredients
                similar = ingredient_model.wv.most_similar(ing, topn=TOP_N)

                for target, similarity in similar:
                    # Write each relationship
                    session.execute_write(create_similar_relationships, ing, target, float(similarity))

            except KeyError:
                # Ingredient missing in model
                continue

    print("✅ SIMILAR_TO relationships built successfully!")


if __name__ == "__main__":
    main()
