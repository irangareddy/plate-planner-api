from fastapi import FastAPI
from src.services.neo4j_service import find_substitutes, find_recipes_from_pantry
from src.models.request_response_models import SubstituteRequest, PantryRequest

app = FastAPI(title="Plate Planner Backend", version="0.1")

@app.get("/")
def root():
    return {"message": "Plate Planner API is running."}

@app.post("/substitute_ingredient")
def substitute_ingredient(request: SubstituteRequest):
    results = find_substitutes(request.ingredient, request.top_k)
    return {"substitutes": results}

@app.post("/find_recipes_from_pantry")
def find_recipes(request: PantryRequest):
    results = find_recipes_from_pantry(request.pantry, request.top_k)
    return {"recipes": results}
