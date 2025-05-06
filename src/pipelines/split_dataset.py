import pandas as pd

# Load your full dataset
df = pd.read_csv("app/src/data/raw/RecipeNLG_dataset.csv")

# Take a 1% sample (adjust depending on your Mac's capability)
sample_df = df.sample(frac=0.05, random_state=42)  # ~2.5% of 2M = 200k

# Save the sample to a file (optional)
sample_df.to_csv("app/src/data/raw/recipe_dataset_100k.csv", index=False)
