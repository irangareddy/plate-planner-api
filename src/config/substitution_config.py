# config.py

class SubstitutionConfig:
    """Configuration constants for ingredient substitution modeling.
    Access as SubstitutionConfig.INGREDIENT_WEIGHT, etc.
    """

    # Vector weights
    INGREDIENT_WEIGHT = 0.85
    ACTION_WEIGHT = 0.15

    # Advanced hybrid/heuristic weights (for DIISH-like approaches)
    ONTOLOGY_WEIGHT = 0.35
    EMBEDDING_WEIGHT = 0.45
    NUTRITION_WEIGHT = 0.20  # For explicit nutrition similarity

    # Nutritional penalty (e.g. for exceeding calories)
    NUTRITION_PENALTY = 0.2

    # Filtering thresholds
    MAX_CALORIE_RATIO = 1.2  # Substitute can't exceed 120% of original calories
