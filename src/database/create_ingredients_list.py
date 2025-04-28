import pandas as pd

# Load your dataset
df = pd.read_csv('/src/data/RecipeNLG_dataset.csv')  # Replace with your actual path

# 'NER' column is a string that looks like a list — we need to convert it
import ast

# Parse the 'NER' column safely
df['NER'] = df['NER'].apply(ast.literal_eval)

# Now flatten all ingredients into one big list
all_ingredients = []

for ingredients_list in df['NER']:
    all_ingredients.extend(ingredients_list)

print(f"Total ingredients collected: {len(all_ingredients)}")

# Optional: Make them lowercase and strip whitespace
all_ingredients = [ingredient.strip().lower() for ingredient in all_ingredients]

# Optional: Get only unique ingredients
unique_ingredients = list(set(all_ingredients))

print(f"Unique ingredients found: {len(unique_ingredients)}")

# Save to a simple CSV for later use
ingredients_df = pd.DataFrame(unique_ingredients, columns=['ingredient'])
ingredients_df.to_csv('ingredients_list.csv', index=False)

print("Ingredient list saved to 'ingredients_list.csv'")
