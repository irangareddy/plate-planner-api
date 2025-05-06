import pandas as pd

# --- Config ---
CSV_PATH = "/data/processed/ingredient_substitution/substitution_edges_with_context.csv"
TOP_N = 100
REPORT_PATH = "/data/results/substitution/substitution_target_diagnostics.txt"

# --- Load CSV ---
print("📦 Loading substitution CSV...")
df = pd.read_csv(CSV_PATH)

# --- Generate Top Target Report ---
print(f"📊 Getting top {TOP_N} most frequent targets...")
top_targets = df["target"].value_counts().head(TOP_N)

with open(REPORT_PATH, "w") as f:
    f.write("=== TOP TARGETS IN SUBSTITUTION GRAPH ===\n\n")
    for i, (term, count) in enumerate(top_targets.items(), 1):
        f.write(f"{i:3}. {term:25} → {count} occurrences\n")

    # --- Context Summary ---
    if "context" in df.columns:
        f.write("\n\n=== UNIQUE CONTEXT VALUES ===\n\n")
        context_counts = df["context"].fillna("None").value_counts()
        for context, count in context_counts.items():
            f.write(f"- {context:15} → {count} edges\n")
    else:
        f.write("\n⚠️ No 'context' column found in the dataset.\n")

print(f"✅ Report saved to: {REPORT_PATH}")
