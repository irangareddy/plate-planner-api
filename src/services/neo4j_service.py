from neo4j import GraphDatabase
from src.config.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def find_substitutes(ingredient_name: str, top_k: int = 5):
    query = """
        MATCH (a:Ingredient {name: $name})-[r:SIMILAR_TO]->(b:Ingredient)
        RETURN b.name AS substitute, r.score AS similarity
        ORDER BY r.score DESC
        LIMIT $top_k;
    """
    with driver.session() as session:
        results = session.execute_read(lambda tx: tx.run(query, name=ingredient_name, top_k=top_k).data())
    return results

def find_recipes_from_pantry(pantry: list, top_k: int = 10):
    query = """
        WITH $pantry AS pantry
        MATCH (i:Ingredient)
        WHERE i.name IN pantry
        MATCH (i)-[:SIMILAR_TO*0..1]->(sub:Ingredient)
        MATCH (r:Recipe)-[:HAS_INGREDIENT]->(sub)
        WITH r, collect(DISTINCT sub.name) AS matched_ingredients
        ORDER BY size(matched_ingredients) DESC
        RETURN r.title AS recipe, matched_ingredients
        LIMIT $top_k;
    """
    with driver.session() as session:
        results = session.execute_read(lambda tx: tx.run(query, pantry=pantry, top_k=top_k).data())
    return results
