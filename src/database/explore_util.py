# 07_explore_neo4j_util.py

import os

from config.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from config.paths import DataPaths
from neo4j import GraphDatabase

paths = DataPaths()
OUTPUT_FILE = paths.graph_summary

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

queries = {
    "Ingredient Count": "MATCH (i:Ingredient) RETURN count(i) AS total",
    "Recipe Count": "MATCH (r:Recipe) RETURN count(r) AS total",
    "HAS_INGREDIENT Relationship Count": "MATCH ()-[r:HAS_INGREDIENT]->() RETURN count(r) AS total",
    "SIMILAR_TO Relationship Count": "MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) AS total",
    "SUBSTITUTES_WITH Relationship Count": "MATCH ()-[r:SUBSTITUTES_WITH]->() RETURN count(r) AS total",

    "Total Ingredients": "MATCH (i:Ingredient) RETURN count(i) AS total_ingredients",
    "Total Recipes": "MATCH (r:Recipe) RETURN count(r) AS total_recipes",

    "Sample Recipe-Ingredient Relationships": """
        MATCH (r:Recipe)-[:HAS_INGREDIENT]->(i:Ingredient)
        RETURN r.recipe_id, r.title, i.name
        LIMIT 10
    """,

    "Sample Recipe Metadata": """
        MATCH (r:Recipe)
        WHERE r.directions IS NOT NULL AND r.link IS NOT NULL AND r.source IS NOT NULL
        RETURN r.recipe_id, r.title, r.source, left(r.directions, 100) AS short_directions, r.link
        LIMIT 10
    """,

    "Recipes Using Butter": """
        MATCH (i:Ingredient {name: 'butter'})<-[:HAS_INGREDIENT]-(r:Recipe)
        RETURN r.title
        LIMIT 10
    """,

    "Common Ingredients with Butter": """
        MATCH (r:Recipe)-[:HAS_INGREDIENT]->(i:Ingredient),
              (r)-[:HAS_INGREDIENT]->(other:Ingredient)
        WHERE i.name = "butter" AND i <> other
        RETURN other.name, count(*) AS together_count
        ORDER BY together_count DESC
        LIMIT 10
    """,

    "Top SIMILAR_TO Edges": """
        MATCH (a:Ingredient)-[r:SIMILAR_TO]->(b:Ingredient)
        RETURN a.name AS source, b.name AS target, r.score, r.context
        ORDER BY r.score DESC
        LIMIT 10
    """,

    "SIMILAR_TO Neighbors of Butter": """
        MATCH (a:Ingredient {name: 'butter'})-[r:SIMILAR_TO]->(b:Ingredient)
        RETURN b.name AS substitute, r.score, r.context
        ORDER BY r.score DESC
        LIMIT 10
    """,

    "Top SUBSTITUTES_WITH Edges": """
        MATCH (a:Ingredient)-[r:SUBSTITUTES_WITH]->(b:Ingredient)
        RETURN a.name AS source, b.name AS substitute, r.score, r.context
        ORDER BY r.score DESC
        LIMIT 10
    """,

    "SUBSTITUTES_WITH Neighbors of Butter": """
        MATCH (a:Ingredient {name: 'butter'})-[r:SUBSTITUTES_WITH]->(b:Ingredient)
        RETURN b.name AS substitute, r.score, r.context
        ORDER BY r.score DESC
        LIMIT 10
    """,

    "Visualize Small Graph": """
        MATCH (r:Recipe)-[:HAS_INGREDIENT]->(i:Ingredient)
        RETURN r.title, i.name
        LIMIT 50
    """
}


def run_query(tx, query):
    result = tx.run(query)
    return [record.data() for record in result]

def main():
    with driver.session() as session, open(OUTPUT_FILE, "w") as f:
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
