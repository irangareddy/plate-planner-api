import re
from difflib import get_close_matches

from neo4j import GraphDatabase

# --- Config ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
NUM_INGREDIENTS = 10
SCORE_THRESHOLD = 0.85  # Optional filter
OUTPUT_FILE = "/data/results/substitution/random_substitution_test_results.txt"

# Optional: fallback to fuzzy match on failed lookup
ENABLE_FUZZY_MATCH = True

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ------------------ Utilities ------------------

def normalize(ingredient):
    ing = ingredient.lower()
    ing = re.sub(r"[^a-z ]", "", ing)
    ing = re.sub(r"\b(brand|fresh|deli|chunky|layered|lowfat|nonfat|plain|pack)\b", "", ing)
    return ing.strip()

def get_random_ingredients(tx, limit):
    result = tx.run("""
        MATCH (i:Ingredient)<-[:HAS_INGREDIENT]-(r:Recipe)
        WITH i.name AS name, count(r) AS uses
        WHERE uses > 50
        WITH name, rand() AS r ORDER BY r
        RETURN name LIMIT $limit
    """, limit=limit)
    return [record["name"] for record in result]


def get_all_ingredients(tx):
    result = tx.run("MATCH (i:Ingredient) RETURN i.name AS name")
    return [r["name"] for r in result]

def get_substitutes(tx, ingredient):
    query = """
        MATCH (a:Ingredient {name: $ingredient})-[r:SUBSTITUTES_WITH]->(b:Ingredient)
        WHERE r.score >= $threshold
        RETURN b.name AS substitute, r.score
        ORDER BY r.score DESC
        LIMIT 5
    """
    return list(tx.run(query, ingredient=ingredient, threshold=SCORE_THRESHOLD))


# ------------------ Main ------------------

def main():
    with driver.session() as session, open(OUTPUT_FILE, "w") as f:
        all_ingredients = session.execute_read(get_all_ingredients)
        ingredients = session.execute_read(get_random_ingredients, NUM_INGREDIENTS)

        f.write("=== Random Ingredient Substitution Test ===\n\n")

        for ingredient in ingredients:
            norm_ing = normalize(ingredient)

            # Fallback with fuzzy matching if needed
            if norm_ing not in all_ingredients and ENABLE_FUZZY_MATCH:
                match = get_close_matches(norm_ing, all_ingredients, n=1)
                if match:
                    norm_ing = match[0]

            f.write(f">>> {ingredient} → {norm_ing}\n")
            results = session.execute_read(get_substitutes, norm_ing)

            if not results:
                f.write("  No substitutions found.\n\n")
                continue

            for res in results:
                row = res.data()
                substitute = row.get("substitute", "N/A")
                score = row.get("score", 0.0)
                f.write(f"  {substitute:25} (score: {score:.4f})\n")
            f.write("\n")

    print(f"✅ Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
