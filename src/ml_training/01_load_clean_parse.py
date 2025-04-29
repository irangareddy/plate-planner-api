# 01_load_clean_parse.py

import pandas as pd
import ast
import os

# Paths
RAW_DATA_PATH = '/Users/rangareddy/Development/Projects/plate-planner-api/src/data/raw/recipe_dataset.csv'          # <-- adjust this to your file
CLEANED_DATA_PATH = '/Users/rangareddy/Development/Projects/plate-planner-api/src/data/processed/cleaned_ner.csv'

# Step 1: Load dataset
print("Loading dataset...")
df = pd.read_csv(RAW_DATA_PATH)

# Step 2: Parse NER column into list
print("Parsing NER column...")
df['ner_list'] = df['NER'].apply(ast.literal_eval)

# Step 3: Clean ingredients (lowercase + strip spaces)
print("Cleaning ingredients...")
df['ner_list_cleaned'] = df['ner_list'].apply(lambda lst: [x.lower().strip() for x in lst])

# Step 4: Save to processed folder
print(f"Saving cleaned data to {CLEANED_DATA_PATH}...")
os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)
# We need to convert lists to strings to save in CSV
df[['title', 'ner_list_cleaned', 'directions']].to_csv(CLEANED_DATA_PATH, index=False)

print("✅ Done! Saved cleaned dataset as CSV.")