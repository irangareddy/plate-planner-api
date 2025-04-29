# src/ml_training/prepare_recipe_training_data.py
import pandas as pd
import numpy as np
import ast
import os
from tqdm import tqdm
from scipy.sparse import lil_matrix, save_npz


tqdm.pandas()

# Paths
RECIPES_PATH = "/Users/rangareddy/Development/Projects/plate-planner-api/src/data/processed/recipes.csv"
RELATIONS_PATH = "/Users/rangareddy/Development/Projects/plate-planner-api/src/data/processed/recipe_ingredients.csv"
OUTPUT_FEATURES_PATH = "/Users/rangareddy/Development/Projects/plate-planner-api/src/ml_models/recipe_features.npy"
OUTPUT_LABELS_PATH = "/Users/rangareddy/Development/Projects/plate-planner-api/src/ml_models/recipe_labels.npy"

os.makedirs("/Users/rangareddy/Development/Projects/plate-planner-api/src/ml_models", exist_ok=True)

def main():
    print("Loading recipes and relations...")
    recipes_df = pd.read_csv(RECIPES_PATH)
    relations_df = pd.read_csv(RELATIONS_PATH)

    print("Building ingredient vocabulary...")
    all_ingredients = relations_df['ingredient'].unique().tolist()
    ingredient_to_idx = {ingredient: idx for idx, ingredient in enumerate(all_ingredients)}
    vocab_size = len(ingredient_to_idx)
    print(f"Total unique ingredients: {vocab_size}")

    print("Building sparse recipe feature vectors...")
    num_recipes = len(recipes_df)
    features = lil_matrix((num_recipes, vocab_size), dtype=np.float32)

    recipe_labels = []

    for idx, recipe_id in tqdm(enumerate(recipes_df['recipe_id'].tolist()), total=num_recipes):
        ingredients = relations_df[relations_df['recipe_id'] == recipe_id]['ingredient'].tolist()
        for ing in ingredients:
            if ing in ingredient_to_idx:
                features[idx, ingredient_to_idx[ing]] = 1
        recipe_labels.append(recipe_id)

    print("Saving sparse features and labels...")
    save_npz(OUTPUT_FEATURES_PATH, features.tocsr())
    np.save(OUTPUT_LABELS_PATH, np.array(recipe_labels))

    print("✅ Sparse recipe training data prepared successfully!")

if __name__ == "__main__":
    main()
