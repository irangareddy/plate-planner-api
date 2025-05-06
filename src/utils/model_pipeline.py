import logging
import os

import pandas as pd
from gensim.models import Word2Vec
from neo4j import GraphDatabase

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------
# 1. Load and Prepare Dataset
# ---------------------
def load_dataset(filepath: str) -> pd.DataFrame:
    logging.info(f"Loading dataset from {filepath}")
    df = pd.read_csv(filepath)

    if "NER" not in df.columns:
        raise ValueError("Dataset must have 'NER' column.")

    df["ingredients_list"] = df["NER"].apply(eval)  # assuming NER is a stringified list

    # Clean ingredients: lowercased, stripped
    df["ingredients_list"] = df["ingredients_list"].apply(lambda lst: [ing.lower().strip() for ing in lst])

    # Remove recipes with <2 ingredients (optional)
    df = df[df["ingredients_list"].apply(len) > 1]

    return df


# ---------------------
# 2. Train Word2Vec Model
# ---------------------
def train_word2vec(ingredient_sentences: list[list[str]], vector_size: int = 128, window: int = 5,
                   min_count: int = 5) -> Word2Vec:
    logging.info("Training Word2Vec model...")
    model = Word2Vec(
        sentences=ingredient_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=os.cpu_count(),
        sg=1  # skip-gram model
    )
    logging.info("Word2Vec training completed.")
    return model


# ---------------------
# 3. Find Similar Ingredients (filtered, no self loops)
# ---------------------
def find_similar_ingredients(model: Word2Vec, topn: int = 5, similarity_threshold: float = 0.75) -> list[
    tuple[str, str, float]]:
    logging.info(f"Finding top-{topn} similar ingredients with similarity > {similarity_threshold}")
    substitution_pairs = []
    for ingredient in model.wv.index_to_key:
        try:
            similars = model.wv.most_similar(ingredient, topn=topn)
            for similar_ing, score in similars:
                if score >= similarity_threshold and ingredient != similar_ing:
                    substitution_pairs.append((ingredient, similar_ing, score))
        except KeyError:
            continue

    # Deduplicate (no (A, B) and (B, A))
    seen = set()
    unique_pairs = []
    for ing1, ing2, score in substitution_pairs:
        key = tuple(sorted((ing1, ing2)))
        if key not in seen:
            seen.add(key)
            unique_pairs.append((ing1, ing2, score))

    logging.info(f"Filtered to {len(unique_pairs)} unique substitution pairs after deduplication.")
    return unique_pairs


# ---------------------
# 4. Push Substitutes to Neo4j (Batch UNWIND)
# ---------------------
def batch_push_to_neo4j(substitution_pairs: list[tuple[str, str, float]], uri: str, user: str, password: str,
                        batch_size: int = 500):
    logging.info(f"Pushing {len(substitution_pairs)} substitution relationships to Neo4j in batches of {batch_size}...")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    def execute_batch(tx, batch):
        tx.run("""
        UNWIND $rows AS row
        MATCH (a:Ingredient {name: row.ing1}), (b:Ingredient {name: row.ing2})
        MERGE (a)-[r:SUBSTITUTES_WITH]->(b)
        SET r.context = "general", r.similarityScore = row.score;
        """, rows=batch)

    with driver.session() as session:
        for i in range(0, len(substitution_pairs), batch_size):
            batch = [{"ing1": ing1, "ing2": ing2, "score": float(score)} for ing1, ing2, score in
                     substitution_pairs[i:i + batch_size]]
            session.write_transaction(execute_batch, batch)

    driver.close()
    logging.info("Finished pushing substitutions to Neo4j.")


# ---------------------
# 5. Full Pipeline Runner
# ---------------------
def run_pipeline(
        recipe_filepath: str,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        vector_size: int = 128,
        window: int = 5,
        min_count: int = 5,
        topn: int = 5,
        similarity_threshold: float = 0.75,
        batch_size: int = 500
):
    df = load_dataset(recipe_filepath)
    logging.info(f"Loaded {len(df)} recipes.")

    ingredient_sentences = df["ingredients_list"].tolist()

    w2v_model = train_word2vec(ingredient_sentences, vector_size=vector_size, window=window, min_count=min_count)
    logging.info(f"Trained Word2Vec model on {len(w2v_model.wv.index_to_key)} unique ingredients.")

    substitution_pairs = find_similar_ingredients(w2v_model, topn=topn, similarity_threshold=similarity_threshold)

    batch_push_to_neo4j(substitution_pairs, neo4j_uri, neo4j_user, neo4j_password, batch_size=batch_size)

    logging.info("Pipeline completed successfully.")


# ---------------------
# 6. Main Entry
# ---------------------
if __name__ == "__main__":
    run_pipeline(
        recipe_filepath="/data/raw/RecipeNLG_dataset.csv",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="12345678"
    )
