"""Read dataset"""
import pandas as pd
import numpy as np
from textwrap import dedent

# 1. Data Loading with Error Handling
try:
    df = pd.read_csv('/src/data/RecipeNLG_dataset.csv')  # Use relative path
    print("Dataset loaded successfully!\n")

    # 2. Initial Exploration
    metrics = []
    metrics.append("=== Dataset Overview ===\n")
    metrics.append(f"Shape: {df.shape}\n")
    metrics.append("\nFirst 5 rows:\n" + df.head().to_string())

    # 3. Detailed Analysis
    metrics.append("\n\n=== Data Types ===")
    metrics.append(df.dtypes.to_string())

    metrics.append("\n\n=== Missing Values ===")
    metrics.append(df.isnull().sum().to_string())

    metrics.append("\n\n=== Unique Values ===")
    metrics.append(f"Unique Ingredients: {df['ingredients'].nunique()}")

    # 4. Save Metrics
    with open('../data/dataset_understanding.txt', 'w') as f:
        f.write("\n".join(metrics))

    print("Metrics saved to dataset_understanding.txt")

except FileNotFoundError:
    print("Error: File not found. Please verify:")
    print("- File exists in current directory")
    print("- Correct filename: 'RecipeNLG_dataset.csv'")
    print("- File is not open in other programs")
except Exception as e:
    print(f"Unexpected error: {str(e)}")
