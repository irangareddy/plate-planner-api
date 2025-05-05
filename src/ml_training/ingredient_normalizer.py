
import re
import wordninja

DESCRIPTORS = {
    "fresh", "frozen", "dried", "blanched", "steamed", "sweetened", "unsweetened",
    "canned", "instant", "chunky", "sliced", "grated", "crushed", "whole", "nonfat",
    "lowfat", "lean", "fatfree", "reduced", "condensed", "prepared", "thawed", "peeled"
}

UNITS = {
    "cup", "cups", "tbsp", "tablespoon", "tablespoons", "tsp", "teaspoon", "grams",
    "sticks", "ounces", "oz", "pound", "liters", "ml", "g", "lb", "can", "jar", "slice",
    "pack", "package", "stick", "serving"
}

STOPWORDS = {"with", "and", "from", "your", "for", "of", "only", "some", "a", "the", "to", "in"}
BLACKLIST = {"brand", "type", "style", "version", "blend", "classic", "premium"}

def normalize_ingredient_v2(text, fallback=True, return_score=False):
    original = text
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)

    # Step 1: WordNinja split
    split_tokens = []
    for word in text.split():
        split_tokens.extend(wordninja.split(word))

    # Step 2: Filter
    filtered = [
        t for t in split_tokens
        if t not in DESCRIPTORS and t not in UNITS and t not in STOPWORDS and t not in BLACKLIST and len(t) > 2
    ]

    # Step 3: Fallback logic
    if not filtered and fallback:
        last_token = split_tokens[-1] if split_tokens else ""
        return (last_token, "fallback") if return_score else last_token

    # Score based on token count
    score = "strong" if len(filtered) >= 2 else "weak"
    normalized = " ".join(filtered)

    return (normalized, score) if return_score else normalized

inputs = [
    "miracle whip salad dressing",
    "tblsp honey",
    "yellow sweet pepper",
    "kit kat fingers",
    "button garlic",
    "some brand butter",
]

for ing in inputs:
    norm, score = normalize_ingredient_v2(ing, return_score=True)
    print(f"{ing:35} → {norm:25} [{score}]")

