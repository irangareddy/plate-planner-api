# src/ml_training/train_substitution_model.py

import numpy as np
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Paths
FEATURES_PATH = "src/ml_models/substitution_features.npy"
LABELS_PATH = "src/ml_models/substitution_labels.pkl"
OUTPUT_MODEL_PATH = "src/ml_models/ingredient_substitution_model.pkl"

os.makedirs("src/ml_models", exist_ok=True)

def main():
    print("Loading substitution features and labels...")
    X = np.load(FEATURES_PATH)

    with open(LABELS_PATH, 'rb') as f:
        labels = pickle.load(f)

    print("Preparing multi-label binarizer...")
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(labels)

    print(f"Total classes: {len(mlb.classes_)}")

    print("Splitting data...")
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

    print("Training KNN model...")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, Y_train)

    print("Evaluating model...")
    score = model.score(X_val, Y_val)
    print(f"Validation Accuracy: {score:.4f}")

    print("Saving model and label binarizer...")
    with open(OUTPUT_MODEL_PATH, 'wb') as f:
        pickle.dump((model, mlb), f)

    print("✅ Ingredient substitution model trained and saved!")

if __name__ == "__main__":
    main()
