import json

import pandas as pd
import spacy
from neo4j import GraphDatabase

# --- Config ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
TOP_K = 5

# --- NLP Normalizer ---
nlp = spacy.load("en_core_web_sm")
def normalize_ingredient(name):
    doc = nlp(name)
    lemma = " ".join([token.lemma_ for token in doc if token.pos_ != "DET"])
    return lemma.replace(" ", "_").lower().strip()

# --- Neo4j Driver ---
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def get_direct_subs(tx, ingredient, context=None, top_k=5):
    matched = True
    if context:
        result = tx.run("""
            MATCH (a:Ingredient {name: $ingredient})-[r:SUBSTITUTES_WITH]->(b)
            WHERE r.context = $context
            RETURN b.name AS substitute, r.score AS score
            ORDER BY score DESC
            LIMIT $top_k
        """, ingredient=ingredient, context=context, top_k=top_k)
        subs = [{"name": r["substitute"], "score": r["score"], "context": context, "source": "direct"} for r in result]
        if subs:
            return subs, "matched"

    result = tx.run("""
        MATCH (a:Ingredient {name: $ingredient})-[r:SUBSTITUTES_WITH]->(b)
        RETURN b.name AS substitute, r.score AS score, r.context AS context
        ORDER BY score DESC
        LIMIT $top_k
    """, ingredient=ingredient, top_k=top_k)
    return [{"name": r["substitute"], "score": r["score"], "context": r.get("context", "general"), "source": "direct"} for r in result], "fallback"

def get_cooccurrence_subs(tx, ingredient, top_k=5):
    result = tx.run("""
        MATCH (r:Recipe)-[:HAS_INGREDIENT]->(i:Ingredient {name: $ingredient})
              <-[:HAS_INGREDIENT]-(r)-[:HAS_INGREDIENT]->(sub:Ingredient)
        WHERE sub.name <> $ingredient
        RETURN sub.name AS substitute, COUNT(*) AS score
        ORDER BY score DESC
        LIMIT $top_k
    """, ingredient=ingredient, top_k=top_k)
    return [{"name": r["substitute"], "score": round(r["score"] / 50.0, 4), "context": None, "source": "cooccurrence"} for r in result]

def get_hybrid_subs(tx, ingredient, context=None, top_k=5, alpha=0.9):
    direct_subs, mode = get_direct_subs(tx, ingredient, context, top_k=top_k * 2)
    cooc_subs = get_cooccurrence_subs(tx, ingredient, top_k=top_k * 2)

    direct_dict = {s["name"]: s for s in direct_subs}
    cooc_dict = {s["name"]: s for s in cooc_subs}

    merged = []
    all_names = set(direct_dict) | set(cooc_dict)

    for name in all_names:
        d = direct_dict.get(name)
        c = cooc_dict.get(name)
        d_score = d["score"] if d else 0.0
        c_score = c["score"] if c else 0.0
        total = round(alpha * d_score + (1 - alpha) * c_score, 4)
        merged.append({
            "name": name,
            "score": total,
            "context": context if d else None,
            "source": "hybrid"
        })

    return sorted(merged, key=lambda x: -x["score"])[:top_k]

# --- Evaluation Runner ---
def run_eval(input_csv, output_json, use_hybrid=False):
    df = pd.read_csv(input_csv)
    results = []

    with driver.session() as session:
        for _, row in df.iterrows():
            raw_ing = row["query_ingredient"]
            context = str(row.get("context", "")).strip().lower() or None
            norm_ing = normalize_ingredient(raw_ing)

            if use_hybrid:
                subs = session.execute_read(get_hybrid_subs, norm_ing, context)
                matched = any(s["context"] == context and s["source"] == "direct" for s in subs)
                results.append({
                    "ingredient": raw_ing,
                    "context": context or "fallback",
                    "hybrid": True,
                    "matched": matched,
                    "substitutes": subs
                })
            else:
                direct_subs, mode = session.execute_read(get_direct_subs, norm_ing, context)
                results.append({
                    "ingredient": raw_ing,
                    "context": context or "fallback",
                    "hybrid": False,
                    "matched": (mode == "matched"),
                    "substitutes": direct_subs
                })

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Substitution results saved to: {output_json}")


if __name__ == "__main__":
    run_eval("/Users/rangareddy/Development/OSS/plate-planner-api/src/data/processed/ingredient_substitution/eval_queries.csv", "substitution_eval_results.json", use_hybrid=False)
