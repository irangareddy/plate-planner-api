from neo4j import GraphDatabase

from src.config.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from src.evaluation.hybrid_substitution import (
    get_direct_subs,
    get_hybrid_subs,
    normalize_ingredient,
)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def get_hybrid_substitutes(
    ingredient: str,
    context: str | None = None,
    top_k: int = 5,
    alpha: float = 0.9,
    use_hybrid: bool = True
):
    norm_ing = normalize_ingredient(ingredient)

    with driver.session() as session:
        if use_hybrid:
            return session.execute_read(get_hybrid_subs, norm_ing, context, top_k, alpha)
        else:
            return session.execute_read(_direct_only, norm_ing, context, top_k)

# Helper: direct-only fallback
def _direct_only(tx, ingredient, context=None, top_k=5):
    direct, _ = get_direct_subs(tx, ingredient, context, top_k)
    return sorted(direct, key=lambda x: -x["score"])[:top_k]


def recipe_details(title: str):
    def _fetch_recipe(tx, title):
        result = tx.run("""
            MATCH (r:Recipe)
            WHERE toLower(r.title) = toLower($title)
            OPTIONAL MATCH (r)-[:HAS_INGREDIENT]->(i:Ingredient)
            RETURN r.title AS title,
                   r.directions AS directions,
                   r.link AS link,
                   r.source AS source,
                   collect(i.name) AS ingredients
        """, title=title)
        return result.single()

    with driver.session() as session:
        return session.execute_read(_fetch_recipe, title)

