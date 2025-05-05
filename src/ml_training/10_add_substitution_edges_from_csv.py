import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
from tqdm import tqdm

# ------------------ Config ------------------
CSV_PATH = '/Users/rangareddy/Development/OSS/plate-planner-api/src/data/processed/substitution_edges_colab_20250505_085919.csv'  # 🔁 Change this
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
BATCH_SIZE = 1000
MIN_SCORE = 0.90
NOISE = {"level", "spray", "kellogg"}

# ------------------ Neo4j Driver ------------------
load_dotenv()
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def batch_insert(tx, rows):
    tx.run("""
        UNWIND $batch AS row
        MATCH (a:Ingredient {name: row.source})
        MATCH (b:Ingredient {name: row.target})
        MERGE (a)-[r:SUBSTITUTES_WITH]->(b)
        SET r.score = row.score
    """, batch=rows)

# ------------------ Main ------------------
def main():
    print("📦 Loading substitution CSV...")
    df = pd.read_csv(CSV_PATH)

    # Filter low scores and noisy terms
    print("🧹 Filtering poor or noisy substitutions...")
    df = df[df['score'] >= MIN_SCORE]
    df = df[~df['target'].isin(NOISE)]

    print(f"✅ {len(df)} edges remaining after filtering")

    rows = df.to_dict(orient='records')
    with driver.session() as session:
        for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="🔁 Uploading"):
            batch = rows[i:i + BATCH_SIZE]
            session.execute_write(batch_insert, batch)

    print("✅ All substitution edges uploaded to Neo4j.")

if __name__ == "__main__":
    main()
