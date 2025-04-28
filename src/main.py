import ast  # To parse the string into a list

from fastapi import FastAPI

app = FastAPI()


@app.get("/api/v1/suggest")
def suggest_meals(pantry: str, cuisine: str):
    """Endpoint to suggest meals based on the pantry items and cuisine type.
    """
    # Parse the pantry string into a list
    pantry_list = ast.literal_eval(pantry)

    if cuisine.lower() == "italian":
        suggested_meals = [
            {
                "name": "Tomato Pasta",
                "cuisine": "Italian",
                "ingredients": ["pasta", "tomato", "cheese"],
                "instructions": "Boil the pasta, then mix with tomato sauce and top with cheese."
            },
            {
                "name": "Caprese Salad",
                "cuisine": "Italian",
                "ingredients": ["tomato", "cheese", "basil"],
                "instructions": "Slice the tomatoes and cheese, add basil leaves, and drizzle with olive oil."
            }
        ]
    else:
        suggested_meals = []

    return {"suggested_meals": suggested_meals}
