from neo4j import GraphDatabase

# Neo4j connection details
uri = "bolt://localhost:7687"  # default port
username = "database"
password = "12345678"  # <-- update this

driver = GraphDatabase.driver(uri, auth=(username, password))

# Ingredients and substitutions
ingredients = [
    "Milk", "Almond Milk", "Butter", "Coconut Oil", "Sour Cream", "Yogurt"
]

substitutions = [
    ("Milk", "Almond Milk", 0.9),
    ("Butter", "Coconut Oil", 0.8),
    ("Sour Cream", "Yogurt", 0.85),
]

def create_ingredients(tx, ingredients):
    for ingredient in ingredients:
        tx.run("MERGE (:Ingredient {name: $name})", name=ingredient)

def create_substitutions(tx, substitutions):
    for from_ing, to_ing, confidence in substitutions:
        tx.run("""
            MATCH (from:Ingredient {name: $from_name})
            MATCH (to:Ingredient {name: $to_name})
            MERGE (from)-[:SUBSTITUTES_FOR {confidence: $confidence}]->(to)
        """, from_name=from_ing, to_name=to_ing, confidence=confidence)

def find_substitutes(tx, ingredient_name):
    result = tx.run("""
        MATCH (i:Ingredient {name: $name})-[:SUBSTITUTES_FOR]->(sub)
        RETURN sub.name AS substitute, i.name AS original
    """, name=ingredient_name)
    return [record["substitute"] for record in result]

# Open a session
with driver.session() as session:
    # Create data
    session.write_transaction(create_ingredients, ingredients)
    session.write_transaction(create_substitutions, substitutions)

    # Query a few items
    butter_substitutes = session.read_transaction(find_substitutes, "Butter")
    milk_substitutes = session.read_transaction(find_substitutes, "Milk")

    print(f"Substitutes for Butter: {butter_substitutes}")
    print(f"Substitutes for Milk: {milk_substitutes}")

driver.close()
