# 11_run_substitution_queries_and_analyze.py

import os
from datetime import datetime

from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

OUTPUT_FILE = f"data/results/substitution_graph_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# --- Queries ---
queries = {
    "Total Ingredients in Graph": """
        MATCH (i:Ingredient) RETURN count(i) AS total_ingredients;
    """,
    "Total Recipes in Graph": """
        MATCH (r:Recipe) RETURN count(r) AS total_recipes;
    """,
    "Find Best Substitutes for 'butter'": """
        MATCH (a:Ingredient {name: 'butter'})-[r:SIMILAR_TO]->(b:Ingredient)
        RETURN b.name AS substitute, r.score AS similarity
        ORDER BY r.score DESC
        LIMIT 5;
    """,
    "Recipes Using 'butter' or its Substitutes (up to 2 hops)": """
        MATCH (start:Ingredient {name: 'butter'})-[:SIMILAR_TO*0..2]->(sub:Ingredient)
        MATCH (r:Recipe)-[:HAS_INGREDIENT]->(sub)
        RETURN r.title AS recipe, collect(sub.name) AS matched_ingredients
        LIMIT 10;
    """,
    "Pantry Direct Match (flour, milk, eggs)": """
        WITH ['flour', 'milk', 'eggs'] AS pantry
        MATCH (i:Ingredient)
        WHERE i.name IN pantry
        MATCH (r:Recipe)-[:HAS_INGREDIENT]->(i)
        WITH r, count(i) AS matched_ingredients
        ORDER BY matched_ingredients DESC
        RETURN r.title AS recipe, matched_ingredients
        LIMIT 10;
    """,
    "Pantry Smart Match (allow substitutes for flour, milk, eggs)": """
        WITH ['flour', 'milk', 'eggs'] AS pantry
        MATCH (i:Ingredient)
        WHERE i.name IN pantry
        MATCH (i)-[:SIMILAR_TO*0..1]->(sub:Ingredient)
        MATCH (r:Recipe)-[:HAS_INGREDIENT]->(sub)
        WITH r, collect(DISTINCT sub.name) AS matched_ingredients
        ORDER BY size(matched_ingredients) DESC
        RETURN r.title AS recipe, matched_ingredients
        LIMIT 10;
    """
}

def run_query(tx, query):
    result = tx.run(query)
    return [record.data() for record in result]

def main():
    analysis_points = []

    with driver.session() as session, open(OUTPUT_FILE, "w") as f:
        f.write(f"Plate Planner Neo4j Graph Exploration Summary\nGenerated on {datetime.now()}\n\n")

        for title, query in queries.items():
            f.write(f"=== {title} ===\n")
            print(f"Running query: {title}")

            records = session.execute_read(run_query, query)

            if not records:
                f.write("No results found.\n\n")
                analysis_points.append(f"Issue detected: No results for {title}")
                continue

            for record in records:
                line = ", ".join([f"{k}: {v}" for k, v in record.items()])
                f.write(line + "\n")
            f.write("\n\n")

    print(f"✅ All queries completed. Results saved to {OUTPUT_FILE}")

    # Basic Analysis
    print("\n=== Analysis Summary ===")
    if not analysis_points:
        print("✅ Graph is healthy. Substitutions, recipes, and traversals working!")
    else:
        for point in analysis_points:
            print(f"⚠️ {point}")

if __name__ == "__main__":
    main()
