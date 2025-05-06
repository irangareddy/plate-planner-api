from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    project_root: Path = Path(__file__).resolve().parents[2]
    data_root: Path = project_root / "src" / "data"

    # === Subdirectories ===
    models: Path = data_root / "models"
    processed: Path = data_root / "processed"
    raw: Path = data_root / "raw"
    results: Path = data_root / "results"
    support: Path = data_root / "support"

    # === Models ===
    recipe_faiss_index: Path = models / "recipe_suggestion" / "recipe_index.faiss"
    action_w2v: Path = models / "ingredient_substitution" / "action_w2v.model"
    ingredient_w2v: Path = models / "ingredient_substitution" / "ingredient_w2v.model"
    faiss_context_index: Path = models / "ingredient_substitution" / "faiss_context.index"

    # === Processed (ingredient substitution) ===
    cleaned_ner: Path = processed / "ingredient_substitution" / "cleaned_ner.csv"
    cleaned_ner_actions: Path = processed / "ingredient_substitution" / "cleaned_ner_actions.csv"
    context_metadata: Path = processed / "ingredient_substitution" / "context_metadata.csv"
    context_vectors: Path = processed / "ingredient_substitution" / "context_vectors.npy"
    eval_queries: Path = processed / "ingredient_substitution" / "eval_queries.csv"
    substitution_edges: Path = processed / "ingredient_substitution" / "substitution_edges.csv"
    substitution_edges_cleaned: Path = processed / "ingredient_substitution" / "substitution_edges_cleaned.csv"
    substitution_edges_with_context: Path = processed / "ingredient_substitution" / "substitution_edges_with_context.csv"
    substitution_edges_with_context_cleaned: Path = processed / "ingredient_substitution" / "substitution_edges_with_context_cleaned.csv"

    # === Processed (recipe suggestion) ===
    recipe_embeddings: Path = processed / "recipe_suggestion" / "recipe_embeddings.npy"
    recipe_metadata: Path = processed / "recipe_suggestion" / "recipe_metadata.csv"

    # === Other processed
    ingredients: Path = processed / "ingredients.csv"
    recipe_ingredients: Path = processed / "recipe_ingredients.csv"
    recipes: Path = processed / "recipes.csv"

    # === Raw Datasets
    recipe_nlg: Path = raw / "RecipeNLG_dataset.csv"
    recipe_dataset_10k: Path = raw / "recipe_dataset_10k.csv"
    recipe_dataset_50k: Path = raw / "recipe_dataset_50k.csv"
    recipe_dataset_100k: Path = raw / "recipe_dataset_100k.csv"
    recipe_dataset_200k: Path = raw / "recipe_dataset_200k.csv"

    # === Results: Exploration
    dataset_understanding: Path = results / "exploration" / "dataset_understanding.txt"
    graph_summary: Path = results / "exploration" / "graph_exploration_summary.txt"

    # === Results: Substitution
    substitution_eval_report: Path = results / "substitution" / "substitution_eval_report.txt"
    substitution_hybrid_results: Path = results / "substitution" / "substitution_hybrid_results.json"
    substitution_domain_review: Path = results / "substitution" / "substitution_domain_review.json"
    substitution_domain_with_scores: Path = results / "substitution" / "substitution_domain_review_with_scores.json"
    ingredients_no_substitutes: Path = results / "substitution" / "ingredients_no_substitutes.txt"
    substitution_target_diagnostics: Path = results / "substitution" / "substitution_target_diagnostics.txt"
    random_substitution_test_results: Path = results / "substitution" / "random_substitution_test_results.txt"

    # === Support
    db_snapshot: Path = support / "db_snapshot.txt"
    support_ingredients: Path = support / "ingredients_list.csv"
