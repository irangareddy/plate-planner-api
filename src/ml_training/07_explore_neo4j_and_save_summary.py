# 07_explore_neo4j_and_save_summary.py

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

OUTPUT_FILE = "data/results/graph_exploration_summary.txt"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

queries = {
    "Total Ingredients": "MATCH (i:Ingredient) RETURN count(i) AS total_ingredients;",
    "Total Recipes": "MATCH (r:Recipe) RETURN count(r) AS total_recipes;",
    "Sample Recipe-Ingredient Relationships": """
        MATCH (r:Recipe)-[:HAS_INGREDIENT]->(i:Ingredient)
        RETURN r.title, i.name
        LIMIT 10;
    """,
    "Recipes Using Butter": """
        MATCH (i:Ingredient {name: 'butter'})<-[:HAS_INGREDIENT]-(r:Recipe)
        RETURN r.title
        LIMIT 10;
    """,
    "Common Ingredients with Butter": """
        MATCH (r:Recipe)-[:HAS_INGREDIENT]->(i:Ingredient),
              (r)-[:HAS_INGREDIENT]->(other:Ingredient)
        WHERE i.name = "butter" AND i <> other
        RETURN other.name, count(*) AS together_count
        ORDER BY together_count DESC
        LIMIT 10;
    """,
    "Visualize Small Graph": """
        MATCH (r:Recipe)-[:HAS_INGREDIENT]->(i:Ingredient)
        RETURN r.title, i.name
        LIMIT 50;
    """
}

def run_query(tx, query):
    result = tx.run(query)
    return [record.data() for record in result]

def main():
    with driver.session() as session, open(OUTPUT_FILE, 'w') as f:
        for section, query in queries.items():
            f.write(f"=== {section} ===\n")
            print(f"Running query: {section}")

            records = session.execute_read(run_query, query)

            for record in records:
                line = ", ".join([f"{k}: {v}" for k, v in record.items()])
                f.write(line + "\n")
            f.write("\n\n")

    print(f"✅ All queries completed. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
