import random
from ast import literal_eval
import pandas as pd
from ranx import Qrels, Run, evaluate
from utils.recipesuggestionmodel import metadata_df, suggest_recipes

# --- Parameters that match API
TOP_N = 10
RERANK_WEIGHT = 0.6
RAW_K = 50
MIN_OVERLAP = 2

# --- Generate test queries dynamically
def generate_test_queries(df, n=10, min_ing=2, max_ing=3):
    queries = []
    for _ in range(n):
        row = df.sample(1).iloc[0]
        try:
            ingredients = literal_eval(row["NER"])
            if len(ingredients) >= min_ing:
                sample = random.sample(ingredients, k=min(max_ing, len(ingredients)))
                queries.append((sample, row["title"]))
        except Exception:
            continue
    return queries

# --- Evaluate using the same suggestion logic as your API
def build_qrels_and_run(queries):
    qrels, run = {}, {}
    for i, (ingredients, expected_title) in enumerate(queries):
        qid = f"q{i}"
        qrels[qid] = {expected_title: 1}

        results = suggest_recipes(
            ingredients=ingredients,
            top_n=TOP_N,
            rerank_weight=RERANK_WEIGHT,
            raw_k=RAW_K,
            min_overlap=MIN_OVERLAP
        )
        run[qid] = {r["title"]: r["combined_score"] for r in results}
    return Qrels(qrels), Run(run)

# --- Evaluation runner per ingredient count
def evaluate_with_fixed_query_size(n_queries=10, num_ingredients=2):
    print(f"\n🔍 Testing with {num_ingredients} ingredient(s)...")
    test_queries = generate_test_queries(metadata_df, n=n_queries, min_ing=num_ingredients, max_ing=num_ingredients)
    qrels, run = build_qrels_and_run(test_queries)

    metrics = ["precision@5", "recall@5", "ndcg@5", "mrr"]
    results = evaluate(qrels=qrels, run=run, metrics=metrics)
    df = pd.DataFrame([results])
    print(df.to_string(index=False))

# --- Main entry point
def main():
    evaluate_with_fixed_query_size(n_queries=10, num_ingredients=2)
    evaluate_with_fixed_query_size(n_queries=10, num_ingredients=3)

if __name__ == "__main__":
    main()
