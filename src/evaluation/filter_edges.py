import pandas as pd

# --- Config ---
CSV_IN = "app/src/data/processed/substitution_edges_with_context.csv"
CSV_OUT = "app/src/data/processed/substitution_edges_with_context_cleaned.csv"

BAD_TARGETS = {
    "level", "equal", "read", "just", "combination", "directions",
    "angel", "starter", "sharp", "pods", "pink", "yellow"
}
SCORE_THRESHOLD = 0.90

# --- Load & Filter ---
print("📦 Loading substitution CSV...")
df = pd.read_csv(CSV_IN)

if "context" in df.columns:
    print("🧠 Context column detected — will be preserved.")
else:
    print("⚠️ No context column found — substitution will be generic.")

print("🧹 Filtering bad targets and low scores...")
df_cleaned = df[~df["target"].isin(BAD_TARGETS)]
df_cleaned = df_cleaned[df_cleaned["score"] >= SCORE_THRESHOLD]

# --- Save Output ---
df_cleaned.to_csv(CSV_OUT, index=False)
print(f"✅ Cleaned CSV saved to: {CSV_OUT}")
