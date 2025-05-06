from evaluation import get_direct_subs, get_hybrid_subs, normalize_ingredient
from neo4j import GraphDatabase

from src.config.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

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

