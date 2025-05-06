
from fastapi import FastAPI
from pydantic import BaseModel

from src.services.neo4j_service import get_hybrid_substitutes
from src.utils.recipesuggestionmodel import suggest_recipes

app = FastAPI(title="Plate Planner Backend", version="0.1")

# ✅ Request schema for recipe suggestion
class RecipeRequest(BaseModel):
    ingredients: list[str]
    top_n: int = 5
    rerank_weight: float = 0.6



@app.get("/")
def root():
    return {"message": "Plate Planner API is running."}


@app.post("/suggest_recipes")
def suggest_recipes_endpoint(request: RecipeRequest):
    results = suggest_recipes(
        ingredients=request.ingredients,
        top_n=request.top_n,
        rerank_weight=request.rerank_weight
    )
    return {
        "input_ingredients": request.ingredients,
        "top_n": request.top_n,
        "results": results
    }

@app.get("/substitute")
def substitute(
    ingredient: str,
    context: str = None,
    hybrid: bool = False  # 👈 now toggleable from query param
):
    substitutes = get_hybrid_substitutes(ingredient, context, use_hybrid=hybrid)
    return {
        "ingredient": ingredient,
        "context": context or "fallback",
        "hybrid": hybrid,
        "substitutes": substitutes
    }
