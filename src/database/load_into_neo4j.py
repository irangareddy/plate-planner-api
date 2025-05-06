import os

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

INGREDIENTS_PATH = "/Users/rangareddy/Development/OSS/plate-planner-api/src/data/processed/ingredients.csv"
RECIPES_PATH = "/Users/rangareddy/Development/OSS/plate-planner-api/src/data/processed/recipes.csv"
RELATIONS_PATH = "/Users/rangareddy/Development/OSS/plate-planner-api/src/data/processed/recipe_ingredients.csv"

BATCH_SIZE = 500

def create_indexes(tx):
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.name IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Recipe) REQUIRE r.recipe_id IS UNIQUE")

def create_ingredients(tx, ingredients):
    for ing in ingredients:
        tx.run("MERGE (i:Ingredient {name: $name})", name=ing)

def create_recipes(tx, recipes):
    for idx, row in recipes.iterrows():
        tx.run("MERGE (r:Recipe {recipe_id: $rid, title: $title})", rid=int(row["recipe_id"]), title=row["title"])

def create_relations(tx, relations):
    for idx, row in relations.iterrows():
        tx.run("""
            MATCH (r:Recipe {recipe_id: $rid})
            MATCH (i:Ingredient {name: $ing})
            MERGE (r)-[:HAS_INGREDIENT]->(i)
        """, rid=int(row["recipe_id"]), ing=row["ingredient"])

def batch(iterable, size):
    l = len(iterable)
    for idx in range(0, l, size):
        yield iterable[idx:min(idx + size, l)]

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            print("Creating indexes...")
            session.execute_write(create_indexes)

            print("Loading ingredients...")
            ingredients_df = pd.read_csv(INGREDIENTS_PATH)
            ingredients = ingredients_df["ingredient"].dropna().unique().tolist()
            num_batches_ingredients = (len(ingredients) + BATCH_SIZE - 1) // BATCH_SIZE
            for batch_ings in tqdm(batch(ingredients, BATCH_SIZE), total=num_batches_ingredients, desc="Ingredients"):
                session.execute_write(create_ingredients, batch_ings)

            print("Loading recipes...")
            recipes_df = pd.read_csv(RECIPES_PATH)
            num_batches_recipes = (len(recipes_df) + BATCH_SIZE - 1) // BATCH_SIZE
            for batch_recipes in tqdm(batch(recipes_df, BATCH_SIZE), total=num_batches_recipes, desc="Recipes"):
                session.execute_write(create_recipes, batch_recipes)

            print("Creating recipe-ingredient relationships...")
            relations_df = pd.read_csv(RELATIONS_PATH)
            num_batches_relations = (len(relations_df) + BATCH_SIZE - 1) // BATCH_SIZE
            for batch_rels in tqdm(batch(relations_df, BATCH_SIZE), total=num_batches_relations, desc="Relations"):
                session.execute_write(create_relations, batch_rels)
        print("✅ Done loading into Neo4j!")
    finally:
        driver.close()

if __name__ == "__main__":
    main()
