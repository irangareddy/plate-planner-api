import pandas as pd
import numpy as np

# 1. Data Loading with Error Handling
try:
    df = pd.read_csv('/Users/rangareddy/Development/OSS/plate-planner-api/src/data/processed/substitution_edges_colab_20250505_085919.csv')
    print("Dataset loaded successfully!\n")

    metrics = []
    metrics.append("=== DATASET OVERVIEW ===\n")
    metrics.append(f"Shape: {df.shape}")
    metrics.append(f"Columns: {list(df.columns)}\n")
    metrics.append("Data Types:\n" + df.dtypes.to_string())
    metrics.append("\nFirst 5 rows:\n" + df.head().to_string())

    # 2. Missing Values
    metrics.append("\n=== MISSING VALUES ===")
    metrics.append(df.isnull().sum().to_string())

    # 3. Unique Values
    metrics.append("\n=== UNIQUE VALUES ===")
    for col in df.columns:
        metrics.append(f"{col}: {df[col].nunique()} unique")

    # 4. Top Frequent Values for Categorical Columns
    metrics.append("\n=== TOP FREQUENT VALUES (Categorical Columns) ===")
    for col in df.select_dtypes(include=['object', 'category']):
        top_vals = df[col].value_counts().head(3)
        metrics.append(f"\nTop in '{col}':\n{top_vals.to_string()}")

    # 5. Numeric Summary
    metrics.append("\n=== NUMERIC SUMMARY ===")
    metrics.append(df.describe().to_string())

    # 6. Score Column Detailed Analysis
    if 'score' in df.columns:
        score = df['score'].dropna()
        metrics.append("\n=== SCORE COLUMN STATS ===")
        metrics.append(f"Min: {score.min()}")
        metrics.append(f"Max: {score.max()}")
        metrics.append(f"Mean: {score.mean()}")
        metrics.append(f"Median: {score.median()}")
        metrics.append(f"Std: {score.std()}")
        metrics.append(f"Range: {score.max() - score.min()}")
        # Histogram (as text)
        hist, bin_edges = np.histogram(score, bins=10)
        hist_lines = ["Histogram:"]
        for i in range(len(hist)):
            hist_lines.append(f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}: {hist[i]}")
        metrics.extend(hist_lines)
    else:
        metrics.append("\nNo 'score' column found in dataset.")

    # 7. Correlation Matrix (if more than one numeric column)
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        metrics.append("\n=== NUMERIC CORRELATION MATRIX ===")
        metrics.append(df[num_cols].corr().to_string())

    # 8. Save Metrics
    with open('../data/dataset_understanding.txt', 'w') as f:
        f.write("\n".join(metrics))

    print("Metrics saved to dataset_understanding.txt")

except FileNotFoundError:
    print("Error: File not found. Please verify:")
    print("- File exists in current directory")
    print("- Correct filename")
    print("- File is not open in other programs")
except Exception as e:
    print(f"Unexpected error: {str(e)}")
