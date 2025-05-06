#!/bin/bash

set -e

echo "Starting data folder reorganization..."

BASE_DIR="src/data"

# Create target directories
mkdir -p $BASE_DIR/{raw,processed/ingredient_substitution,processed/recipe_suggestion,models/ingredient_substitution,models/recipe_suggestion,results/substitution,results/exploration,support}

# Safe move with existence check
move_if_exists () {
    for file in "$@"; do
        if [ -e "$file" ]; then
            echo "Moving $file → $2"
            mv "$file" "$2"
        fi
    done
}

# Move raw datasets
move_if_exists $BASE_DIR/recipe_dataset_*.csv $BASE_DIR/raw/
move_if_exists $BASE_DIR/RecipeNLG_dataset.csv $BASE_DIR/raw/

# Move processed ingredient substitution files
move_if_exists $BASE_DIR/cleaned_ner.csv $BASE_DIR/processed/ingredient_substitution/
move_if_exists $BASE_DIR/cleaned_ner_actions.csv $BASE_DIR/processed/ingredient_substitution/
move_if_exists $BASE_DIR/context_metadata.csv $BASE_DIR/processed/ingredient_substitution/
move_if_exists $BASE_DIR/context_vectors.npy $BASE_DIR/processed/ingredient_substitution/
move_if_exists $BASE_DIR/eval_queries.csv $BASE_DIR/processed/ingredient_substitution/
move_if_exists $BASE_DIR/substitution_edges*.csv $BASE_DIR/processed/ingredient_substitution/

# Move recipe suggestion processed files
move_if_exists $BASE_DIR/recipe_embeddings.npy $BASE_DIR/processed/recipe_suggestion/
move_if_exists $BASE_DIR/recipe_metadata.csv $BASE_DIR/processed/recipe_suggestion/

# Move general processed files
move_if_exists $BASE_DIR/ingredients.csv $BASE_DIR/processed/
move_if_exists $BASE_DIR/recipes.csv $BASE_DIR/processed/
move_if_exists $BASE_DIR/recipe_ingredients.csv $BASE_DIR/processed/

# Move model artifacts
move_if_exists $BASE_DIR/models/action_w2v.model $BASE_DIR/models/ingredient_substitution/
move_if_exists $BASE_DIR/models/ingredient_w2v.model $BASE_DIR/models/ingredient_substitution/
move_if_exists $BASE_DIR/models/faiss_context.index $BASE_DIR/models/ingredient_substitution/
move_if_exists $BASE_DIR/models/recipe_index.faiss $BASE_DIR/models/recipe_suggestion/

# Move evaluation results
move_if_exists $BASE_DIR/results/hit_at_5k_eval_*.txt $BASE_DIR/results/substitution/
move_if_exists $BASE_DIR/results/ingredients_no_substitutes.txt $BASE_DIR/results/substitution/
move_if_exists $BASE_DIR/results/random_substitution_test_results.txt $BASE_DIR/results/substitution/
move_if_exists $BASE_DIR/results/substitution_domain_review*.json $BASE_DIR/results/substitution/
move_if_exists $BASE_DIR/results/substitution_eval_report.txt $BASE_DIR/results/substitution/
move_if_exists $BASE_DIR/results/substitution_hybrid_results.json $BASE_DIR/results/substitution/
move_if_exists $BASE_DIR/results/substitution_target_diagnostics.txt $BASE_DIR/results/substitution/

move_if_exists $BASE_DIR/results/dataset_understanding.txt $BASE_DIR/results/exploration/
move_if_exists $BASE_DIR/results/graph_exploration_summary.txt $BASE_DIR/results/exploration/

# Move support/meta files
move_if_exists $BASE_DIR/dataset_understanding.txt $BASE_DIR/support/
move_if_exists $BASE_DIR/db_snapshot.txt $BASE_DIR/support/
move_if_exists $BASE_DIR/ingredients_list.csv $BASE_DIR/support/

echo "✅ Data folder reorganization complete!"
