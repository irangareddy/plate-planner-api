
from datetime import datetime

from evaluation import normalize_ingredient
from neo4j import GraphDatabase

# --- Configuration ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
OUTPUT_FILE = "ingredients_no_substitutes_normalized.txt"

# --- Neo4j Access ---
def get_ingredients_without_subs(tx):
    result = tx.run("""
        MATCH (i:Ingredient)
        WHERE NOT (i)-[:SUBSTITUTES_WITH]->()
        RETURN i.name AS name
        ORDER BY rand()
        LIMIT 100
    """)
    return [record["name"] for record in result]

# --- Reporting ---
def generate_report(ingredients):
    report = []
    report.append("=== INGREDIENTS WITH NO SUBSTITUTES ===")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for name in ingredients:
        normalized, score = normalize_ingredient(name, return_score=True)
        if normalized != name:
            mapping_status = f"✅ Normalized to: {normalized} [{score}]"
        else:
            mapping_status = f"❌ No normalization applied [{score}]"
        report.append(f"{name:40} → {mapping_status}")

    return "\n".join(report)

# --- Main ---
def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            ingredients = session.execute_read(get_ingredients_without_subs)

            if not ingredients:
                print("No ingredients found without substitutes.")
                return

            report_content = generate_report(ingredients)
            print(report_content)

            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                f.write(report_content)

            print(f"\n✅ Report saved to: {OUTPUT_FILE}")

    except Exception as e:
        print(f"❌ Error: {e!s}")
    finally:
        driver.close()

if __name__ == "__main__":
    main()
