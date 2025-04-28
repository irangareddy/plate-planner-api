from neo4j import GraphDatabase
import pandas as pd
from itertools import combinations

# Connect to Neo4j
uri = "bolt://localhost:7687"
username = "database"
password = "12345678"
driver = GraphDatabase.driver(uri, auth=(username, password))

# Load your recipes and ingredients
recipes_df = pd.read_csv('/src/data/RecipeNLG_dataset.csv')  # Make sure you load the recipes dataset

# Batch size for inserting relationships
BATCH_SIZE = 1000

# Ensure you have an index on Ingredient name for faster lookups
def create_ingredient_index(tx):
    tx.run("CREATE INDEX IF NOT EXISTS FOR (i:Ingredient) ON (i.name)")

# Create the OCCURS_WITH relationships using UNWIND and MERGE
def create_occurs_with_batch(tx, batch):
    ingredients = []
    for recipe_ingredients in batch:
        for ingredient_pair in combinations(recipe_ingredients, 2):  # Get pairs of ingredients
            ingredient1, ingredient2 = ingredient_pair
            ingredients.append((ingredient1, ingredient2))

    # Using UNWIND to batch process relationships
    tx.run(
        """
        UNWIND $ingredients AS pair
        MATCH (i1:Ingredient {name: pair[0]}), (i2:Ingredient {name: pair[1]})
        MERGE (i1)-[r:OCCURS_WITH]->(i2)
        MERGE (i2)-[r2:OCCURS_WITH]->(i1)
        """,
        ingredients=ingredients
    )

# Function for batch inserting the relationships
def batch_insert_occurs_with_relationships(recipes, batch_size=BATCH_SIZE):
    with driver.session() as session:
        # Ensure ingredient index is created before running the bulk operations
        session.write_transaction(create_ingredient_index)

        total_recipes = len(recipes)
        for i in range(0, total_recipes, batch_size):
            batch = recipes[i:i + batch_size]
            session.execute_write(create_occurs_with_batch, batch)
            print(f"Processed batch {i} to {i + len(batch)} out of {total_recipes}")

    print("✅ All 'OCCURS_WITH' relationships inserted.")

# Prepare the recipes data
# Assuming each recipe has a column 'NER' with ingredients as a list of strings
recipes_df['ingredients'] = recipes_df['NER'].apply(eval)  # Convert string list to actual list

# Get a list of ingredients for each recipe
recipes_ingredients = recipes_df['ingredients'].tolist()

# Insert relationships in batches
batch_insert_occurs_with_relationships(recipes_ingredients)
