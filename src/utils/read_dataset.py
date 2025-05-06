import numpy as np
import pandas as pd

# 1. Data Loading with Error Handling
try:
    df = pd.read_csv("/app/src/data/processed/ingredient_substitution/substitution_edges_with_context_cleaned.csv")
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
    for col in df.select_dtypes(include=["object", "category"]):
        top_vals = df[col].value_counts().head(3)
        metrics.append(f"\nTop in '{col}':\n{top_vals.to_string()}")

    # 5. Numeric Summary
    metrics.append("\n=== NUMERIC SUMMARY ===")
    metrics.append(df.describe().to_string())

    # 6. Score Column Detailed Analysis
    if "score" in df.columns:
        score = df["score"].dropna()
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

    # 7. Context-Aware Substitution Analysis
    if "context" in df.columns:
        metrics.append("\n=== CONTEXT-AWARE SUBSTITUTION STATS ===")
        metrics.append(f"Context values (unique): {df['context'].nunique()}")
        context_counts = df["context"].value_counts().head(10)
        metrics.append("\nTop 10 Contexts by Frequency:\n" + context_counts.to_string())

        # Score distribution per context
        metrics.append("\nContext-Based Score Breakdown:")
        grouped = df.groupby("context")["score"].agg(["count", "mean", "std", "min", "max"]).sort_values("count", ascending=False)
        metrics.append(grouped.to_string())

    # 8. Correlation Matrix (if more than one numeric column)
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        metrics.append("\n=== NUMERIC CORRELATION MATRIX ===")
        metrics.append(df[num_cols].corr().to_string())

    # 9. Save Metrics
    with open("/app/src/data/results/exploration/dataset_understanding.txt", "w") as f:
        f.write("\n".join(metrics))

    print("✅ Metrics saved to dataset_understanding.txt")

except FileNotFoundError:
    print("❌ Error: File not found. Please check your path.")
except Exception as e:
    print(f"❌ Unexpected error: {e!s}")
