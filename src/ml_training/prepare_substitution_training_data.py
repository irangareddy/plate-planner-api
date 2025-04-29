# src/ml_training/prepare_substitution_training_data.py

from neo4j import GraphDatabase
from gensim.models import Word2Vec
import numpy as np
import os
import pickle
from dotenv import load_dotenv
from tqdm import tqdm

tqdm.pandas()

# Load environment variables
load_dotenv()

# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

# Paths
MODEL_PATH = "/Users/rangareddy/Development/Projects/plate-planner-api/src/data/models/ingredient_w2v.model"
OUTPUT_FEATURES_PATH = "/Users/rangareddy/Development/Projects/plate-planner-api/src/ml_models/substitution_features.npy"
OUTPUT_LABELS_PATH = "/Users/rangareddy/Development/Projects/plate-planner-api/src/ml_models/substitution_labels.pkl"

os.makedirs("/Users/rangareddy/Development/Projects/plate-planner-api/src/ml_models", exist_ok=True)

# Initialize connections
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
ingredient_model = Word2Vec.load(MODEL_PATH)


def get_similar_ingredients(tx):
    query = """
        MATCH (a:Ingredient)-[r:SIMILAR_TO]->(b:Ingredient)
        RETURN a.name AS source, collect(b.name) AS substitutes;
    """
    return tx.run(query).data()


def main():
    with driver.session() as session:
        print("Fetching substitution pairs from Neo4j...")
        data = session.execute_read(get_similar_ingredients)

    features = []
    labels = []

    for record in tqdm(data):
        source = record['source']
        substitutes = record['substitutes']

        if source in ingredient_model.wv:
            features.append(ingredient_model.wv[source])
            labels.append(substitutes)  # Keep as list

    print("Saving substitution training features and labels...")

    # Save features normally
    np.save(OUTPUT_FEATURES_PATH, np.array(features))

    # Save labels with pickle
    with open(OUTPUT_LABELS_PATH, 'wb') as f:
        pickle.dump(labels, f)

    print("✅ Substitution training dataset prepared successfully!")


if __name__ == "__main__":
    main()
