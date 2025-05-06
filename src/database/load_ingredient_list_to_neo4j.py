from neo4j import GraphDatabase
import pandas as pd

# Connect to your local Neo4j
# (adjust your URI, username, password if needed)
uri = "bolt://localhost:7687"
username = "database"
password = "12345678"

driver = GraphDatabase.driver(uri, auth=(username, password))

# Load ingredients
ingredients_df = pd.read_csv('../data/support/ingredients_list.csv')


def create_ingredient_node(tx, ingredient_name):
    tx.run(
        """
        MERGE (i:Ingredient {name: $name})
        """,
        name=ingredient_name
    )


# Optional but recommended: create uniqueness constraint
def create_unique_constraint(tx):
    tx.run(
        """
        CREATE CONSTRAINT unique_ingredient_name IF NOT EXISTS
        FOR (i:Ingredient)
        REQUIRE i.name IS UNIQUE
        """
    )


# Start inserting
with driver.session() as session:
    # First create constraint
    session.execute_write(create_unique_constraint)

    # Now insert ingredients
    for idx, row in ingredients_df.iterrows():
        ingredient = row['ingredient']
        session.execute_write(create_ingredient_node, ingredient)

        # Progress logging
        if idx % 1000 == 0:
            print(f"Inserted {idx} ingredients...")

driver.close()

print("✅ All ingredients inserted into Neo4j!")
