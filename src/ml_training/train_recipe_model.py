# src/ml_training/train_recipe_model.py

import numpy as np
import pickle
import os
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from tqdm import tqdm

# Paths
FEATURES_PATH = "/Users/rangareddy/Development/Projects/plate-planner-api/src/ml_models/recipe_features.npy.npz"
LABELS_PATH = "/Users/rangareddy/Development/Projects/plate-planner-api/src/ml_models/recipe_labels.npy"
OUTPUT_MODEL_PATH = "/Users/rangareddy/Development/Projects/plate-planner-api/src/ml_models/recipe_suggestion_model.pkl"

os.makedirs("/Users/rangareddy/Development/Projects/plate-planner-api/src/ml_models", exist_ok=True)

def main():
    print("Loading sparse features and labels...")
    X = load_npz(FEATURES_PATH)
    y = np.load(LABELS_PATH)

    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    print("Training LightGBM model...")
    model = LGBMClassifier(
        objective='multiclass',
        num_class=len(np.unique(y)),
        max_depth=10,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")

    print("Saving model...")
    with open(OUTPUT_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    print("✅ LightGBM recipe suggestion model trained and saved!")

if __name__ == "__main__":
    main()
