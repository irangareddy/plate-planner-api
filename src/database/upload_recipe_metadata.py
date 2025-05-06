import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm
from datetime import datetime
import os

# === CONFIG ===
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

CSV_PATH = "/Users/rangareddy/Development/OSS/plate-planner-api/src/data/raw/recipe_dataset_200k.csv"
BATCH_SIZE = 500

# === Neo4j Driver ===
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# === Utility ===
def create_recipe_nodes(tx, batch):
    for recipe in batch:
        tx.run("""
            MERGE (r:Recipe {recipe_id: $recipe_id})
            SET r.title = $title,
                r.directions = $directions,
                r.link = $link,
                r.source = $source
        """, **recipe)

def batch_iter(data, size):
    for i in range(0, len(data), size):
        yield data[i:i + size]

# === Main ===
def main():
    start = datetime.now()
    print(f"🚀 Starting upload at {start.strftime('%H:%M:%S')}")

    print("📖 Reading CSV...")
    df = pd.read_csv(CSV_PATH)

    print("🧹 Cleaning and preparing records...")
    df = df[['title', 'directions', 'link', 'source']].copy()
    df = df.dropna(subset=['title'])
    df = df.reset_index().rename(columns={'index': 'recipe_id'})  # create unique ID if not available

    records = df.to_dict(orient="records")

    print(f"📦 Inserting {len(records):,} recipes into Neo4j...")
    with driver.session() as session:
        for batch in tqdm(batch_iter(records, BATCH_SIZE), total=(len(records) // BATCH_SIZE + 1), desc="📤 Uploading"):
            session.execute_write(create_recipe_nodes, batch)

    end = datetime.now()
    duration = end - start
    print(f"✅ Upload complete at {end.strftime('%H:%M:%S')} (Duration: {duration})")

if __name__ == "__main__":
    main()
