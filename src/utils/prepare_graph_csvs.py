# 05_process_for_neo4j.py

import ast
import os
import re

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

RAW_DATA_PATH = "/Users/rangareddy/Development/Projects/plate-planner-api/app/src/data/raw/recipe_dataset.csv"
CLEANED_INGREDIENTS_PATH = "/Users/rangareddy/Development/Projects/plate-planner-api/app/src/data/processed/ingredients.csv"
CLEANED_RECIPES_PATH = "/Users/rangareddy/Development/Projects/plate-planner-api/app/src/data/processed/recipes.csv"
CLEANED_RELATIONS_PATH = "/Users/rangareddy/Development/Projects/plate-planner-api/app/src/data/processed/recipe_ingredients.csv"

os.makedirs("/Users/rangareddy/Development/Projects/plate-planner-api/app/src/data/processed", exist_ok=True)

# Utility: Clean single ingredient
def clean_ingredient(name):
    if not isinstance(name, str):
        return None
    name = name.lower()
    name = re.sub(r"\([^)]*\)", "", name)  # Remove brackets (like "(1/2 cup)")
    name = re.sub(r"[^a-zA-Z\s]", "", name)  # Remove non-letter characters
    name = re.sub(r"\s+", " ", name).strip()  # Remove extra spaces
    return name if 2 <= len(name) <= 50 else None  # Filter out too short/long junk

def safe_literal_eval(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

def main():
    print("Loading raw dataset...")
    df = pd.read_csv(RAW_DATA_PATH)

    print("Parsing NER ingredients...")
    df["ner_list"] = df["NER"].progress_apply(safe_literal_eval)

    print("Cleaning ingredients...")
    df["ner_list_cleaned"] = df["ner_list"].progress_apply(
        lambda lst: list(filter(None, [clean_ingredient(x) for x in lst]))
    )

    print("Filtering recipes with no cleaned ingredients...")
    df = df[df["ner_list_cleaned"].apply(lambda x: len(x) > 0)]

    print("Assigning recipe IDs...")
    df = df.reset_index(drop=True)
    df["recipe_id"] = df.index

    # --- Create Ingredients Table ---
    print("Building ingredients list...")
    all_ingredients = set(ing for lst in df["ner_list_cleaned"] for ing in lst)
    ingredients_df = pd.DataFrame({"ingredient": list(all_ingredients)})
    ingredients_df = ingredients_df.dropna().drop_duplicates()
    ingredients_df.to_csv(CLEANED_INGREDIENTS_PATH, index=False)

    # --- Create Recipes Table ---
    print("Building recipes table...")
    recipes_df = df[["recipe_id", "title"]].dropna().drop_duplicates()
    recipes_df.to_csv(CLEANED_RECIPES_PATH, index=False)

    # --- Create Recipe-Ingredients Relations ---
    print("Building recipe-ingredient relations...")
    relations = []
    for idx, row in df.iterrows():
        recipe_id = row["recipe_id"]
        for ing in row["ner_list_cleaned"]:
            relations.append((recipe_id, ing))

    relations_df = pd.DataFrame(relations, columns=["recipe_id", "ingredient"])
    relations_df.to_csv(CLEANED_RELATIONS_PATH, index=False)

    print("✅ Processing complete. Cleaned datasets ready!")

if __name__ == "__main__":
    main()
