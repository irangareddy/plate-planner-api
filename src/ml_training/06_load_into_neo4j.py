# 06_load_into_neo4j.py

import pandas as pd
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Load environment variables (assuming you store Neo4j connection in .env)
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

# Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Data paths
INGREDIENTS_PATH = "/Users/rangareddy/Development/Projects/plate-planner-api/src/data/processed/ingredients.csv"
RECIPES_PATH = "/Users/rangareddy/Development/Projects/plate-planner-api/src/data/processed/recipes.csv"
RELATIONS_PATH = "/Users/rangareddy/Development/Projects/plate-planner-api/src/data/processed/recipe_ingredients.csv"

# Batch size for transaction optimization
BATCH_SIZE = 500


def create_indexes(tx):
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.name IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Recipe) REQUIRE r.recipe_id IS UNIQUE")


def create_ingredients(tx, ingredients):
    for ing in ingredients:
        tx.run("MERGE (i:Ingredient {name: $name})", name=ing)


def create_recipes(tx, recipes):
    for idx, row in recipes.iterrows():
        tx.run("MERGE (r:Recipe {recipe_id: $rid, title: $title})", rid=int(row['recipe_id']), title=row['title'])


def create_relations(tx, relations):
    for idx, row in relations.iterrows():
        tx.run("""
            MATCH (r:Recipe {recipe_id: $rid})
            MATCH (i:Ingredient {name: $ing})
            MERGE (r)-[:HAS_INGREDIENT]->(i)
        """, rid=int(row['recipe_id']), ing=row['ingredient'])


def batch(iterable, size):
    l = len(iterable)
    for idx in range(0, l, size):
        yield iterable[idx:min(idx + size, l)]


def main():
    with driver.session() as session:
        # 1. Create Indexes
        print("Creating indexes...")
        session.write_transaction(create_indexes)

        # 2. Load Ingredients
        print("Loading ingredients...")
        ingredients_df = pd.read_csv(INGREDIENTS_PATH)
        ingredients = ingredients_df['ingredient'].dropna().unique().tolist()

        for batch_ings in batch(ingredients, BATCH_SIZE):
            session.write_transaction(create_ingredients, batch_ings)

        # 3. Load Recipes
        print("Loading recipes...")
        recipes_df = pd.read_csv(RECIPES_PATH)
        for batch_recipes in batch(recipes_df, BATCH_SIZE):
            session.write_transaction(create_recipes, batch_recipes)

        # 4. Load Recipe-Ingredient Relationships
        print("Creating recipe-ingredient relationships...")
        relations_df = pd.read_csv(RELATIONS_PATH)
        for batch_rels in batch(relations_df, BATCH_SIZE):
            session.write_transaction(create_relations, batch_rels)

    print("✅ Done loading into Neo4j!")


if __name__ == "__main__":
    main()
