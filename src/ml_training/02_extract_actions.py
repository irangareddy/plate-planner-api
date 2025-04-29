import pandas as pd
import ast
import spacy
import os
from tqdm import tqdm

# ----------------- Paths -----------------
CLEANED_DATA_PATH = '/Users/rangareddy/Development/Projects/plate-planner-api/src/data/processed/cleaned_ner.csv'
ACTIONS_DATA_PATH = '/Users/rangareddy/Development/Projects/plate-planner-api/src/data/processed/cleaned_ner_actions.csv'

# ----------------- Setup -----------------
tqdm.pandas()

nlp = spacy.load("en_core_web_sm")

def safe_literal_eval(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

def extract_verbs_batch(directions_series):
    actions_list = []
    texts = []

    print("Preparing texts for spaCy pipe...")

    for directions in tqdm(directions_series, desc="Preparing texts"):
        if isinstance(directions, str):
            try:
                steps = ast.literal_eval(directions)
                if isinstance(steps, list):
                    full_text = ' '.join(steps)
                else:
                    full_text = directions
            except Exception:
                full_text = directions
        else:
            full_text = ' '.join(directions) if isinstance(directions, list) else str(directions)

        texts.append(full_text)

    print("Running spaCy pipe for action extraction...")

    for doc in tqdm(nlp.pipe(texts, batch_size=32, n_process=4), total=len(texts), desc="spaCy processing"):
        actions = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        actions_list.append(list(set(actions)))

    return actions_list

# ----------------- Main Safe Start -----------------
if __name__ == "__main__":
    print("Loading cleaned ingredients dataset...")
    df = pd.read_csv(CLEANED_DATA_PATH)

    print("Parsing ner_list_cleaned with progress bar...")
    df['ner_list_cleaned'] = df['ner_list_cleaned'].progress_apply(safe_literal_eval)

    print("Extracting actions...")
    df['actions'] = extract_verbs_batch(df['directions'])

    print(f"Saving updated dataset with actions to {ACTIONS_DATA_PATH}...")
    os.makedirs(os.path.dirname(ACTIONS_DATA_PATH), exist_ok=True)
    df[['title', 'ner_list_cleaned', 'directions', 'actions']].to_csv(ACTIONS_DATA_PATH, index=False)

    print("✅ Done! Actions extracted and saved successfully.")
