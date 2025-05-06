
from neo4j import GraphDatabase
from tqdm import tqdm

# --- Config ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
INGREDIENTS_TO_TEST = [
    "cognac", "espresso", "craisins", "jalapenos",
    "angel food cake", "dried apricots", "red chili",
    "cherry pie filling", "strawberry preserves", "green cabbage"
]

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def check_node_exists(tx, name):
    result = tx.run("MATCH (i:Ingredient {name: $name}) RETURN count(i) AS count", name=name)
    return result.single()["count"] > 0

def get_substitutes(tx, ingredient):
    query = """
        MATCH (a:Ingredient {name: $ingredient})-[r:SUBSTITUTES_WITH]->(b:Ingredient)
        RETURN b.name AS substitute, r.score AS score
        ORDER BY r.score DESC
        LIMIT 5
    """
    return list(tx.run(query, ingredient=ingredient))

def main():
    with driver.session() as session:
        print("=== SUBSTITUTION DIAGNOSTIC REPORT ===\n")
        for name in tqdm(INGREDIENTS_TO_TEST, desc="Checking ingredients"):
            print(f">>> Testing: {name}")
            node_exists = session.execute_read(check_node_exists, name)
            if not node_exists:
                print("  ❌ Ingredient node NOT found in graph.")
                continue

            results = session.execute_read(get_substitutes, name)
            if not results:
                print("  ⚠️  No SUBSTITUTES_WITH relationships found.")
                continue

            for res in results:
                row = res.data()
                substitute = row.get("substitute", "N/A")
                score = row.get("score", 0.0)
                print(f"  ✅ {substitute:25} (score: {score:.4f})")
            print()

if __name__ == "__main__":
    main()
