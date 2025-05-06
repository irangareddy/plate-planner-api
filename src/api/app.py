import asyncio
import logging
from typing import List, Optional
from ast import literal_eval

from fastapi import FastAPI, HTTPException, Query, status, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

from src.services.neo4j_service import get_hybrid_substitutes, recipe_details as fetch_recipe_details
from src.utils.recipesuggestionmodel import suggest_recipes, metadata_df

# ——— FastAPI app setup ———
app = FastAPI(
    title="Plate Planner Backend",
    version="0.1",
    openapi_tags=[
        {"name": "health", "description": "Health check"},
        {"name": "recipes", "description": "Recipe suggestion operations"},
        {"name": "substitution", "description": "Ingredient substitution operations"},
    ],
)

# ——— CORS ———
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # <-- lock this down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ——— Logging ———
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("plate_planner")


# ——— Models ———
class RecipeRequest(BaseModel):
    """Incoming request payload for recipe suggestions."""
    ingredients: List[str] = Field(
        ...,
        description="List of ingredients to base the recipe suggestions on",
        example=["butter", "sugar", "flour"],
    )
    top_n: int = Field(
        5,
        description="Number of top recipes to return",
        example=5,
    )
    rerank_weight: float = Field(
        0.6,
        description="Balance between semantic similarity and ingredient overlap (0–1)",
        example=0.6,
    )

    class Config:
        schema_extra = {
            "examples": {
                "🍪 Basic Baking": {
                    "summary": "Common baking ingredients",
                    "value": {"ingredients": ["butter", "sugar", "flour"]}
                },
                "🥗 Light Salad": {
                    "summary": "Simple salad base",
                    "value": {"ingredients": ["lettuce", "tomato", "olive oil"]}
                },
                "🍝 Pasta Dinner": {
                    "summary": "Pasta plus flavorings",
                    "value": {"ingredients": ["pasta", "garlic", "parmesan"]}
                },
                "🍲 Cozy Soup": {
                    "summary": "Soup essentials",
                    "value": {"ingredients": ["chicken", "carrot", "onion"]}
                },
                "🥞 Quick Breakfast": {
                    "summary": "Breakfast staples",
                    "value": {"ingredients": ["egg", "milk", "banana"]}
                },
            }
        }


class RecipeResult(BaseModel):
    """Schema for a single suggested recipe."""
    title: str
    # only those that overlapped with the query
    ingredients: List[str]
    semantic_score: float
    overlap_score: float
    combined_score: float
    rank: int


class RecipeSuggestionResponse(BaseModel):
    """Outgoing response for suggested recipes."""
    input_ingredients: List[str]
    top_n: int
    results: List[RecipeResult]


class SubstituteItem(BaseModel):
    name: str = Field(..., description="Substitute ingredient name", example="oleo")
    score: float = Field(..., description="Normalized similarity score (0–1)", example=0.83)
    context: Optional[str] = Field(None, description="Context in which this substitute applies", example="baking")
    source: str = Field(..., description="Whether this came from 'direct', 'cooccurrence' or 'hybrid'", example="direct")


class SubstituteResponse(BaseModel):
    ingredient: str = Field(..., description="Original ingredient you looked up", example="butter")
    context: Optional[str] = Field(None, description="Optional usage context", example="baking")
    hybrid: bool = Field(..., description="Whether hybrid lookup was used", example=False)
    substitutes: List[SubstituteItem] = Field(..., description="List of candidate substitutions")


class RecipeDetails(BaseModel):
    title: str = Field(..., example="Marinated Flank Steak Recipe")
    directions: list[str] = Field(
        ..., description="Step-by-step cooking instructions"
    )
    link: str = Field(
        ..., description="URL to the original recipe"
    )
    source: str = Field(..., example="Recipes1M")
    ingredients: list[str] = Field(
        ...,
        description="Full ingredient list, deduplicated & in original order",
    )


# ——— Endpoints ———
@app.get("/", tags=["health"], summary="Health check")
async def root() -> dict:
    return {"message": "Plate Planner API is running."}


@app.post(
    "/suggest_recipes",
    response_model=List[RecipeResult],
    status_code=status.HTTP_200_OK,
    summary="Suggest recipes (only overlapping ingredients returned)",
)
async def suggest_recipes_endpoint(request: RecipeRequest):
    try:
        results = await asyncio.to_thread(
            suggest_recipes,
            request.ingredients,
            request.top_n,
            request.rerank_weight,
        )
    except Exception:
        logger.exception("Failed to suggest recipes")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not generate recipe suggestions",
        )

    return results


@app.get(
    "/substitute",
    response_model=SubstituteResponse,
    status_code=status.HTTP_200_OK,
    tags=["substitution"],
    summary="Get possible substitutes for an ingredient",
)
async def substitute(
    ingredient: str = Query(..., description="Ingredient to substitute", example="butter"),
    context: Optional[str] = Query(None, description="Usage context (e.g. baking)", example="baking"),
    hybrid: bool = Query(False, description="Use hybrid substitution (direct + cooccurrence)", example=False),
    top_k: int = Query(5, description="Number of substitutes to return", example=5),
):
    """
    Query Neo4j for direct and/or hybrid substitutes.

    - ingredient: required ingredient name  
    - context: optional use-case filter  
    - hybrid: if true, merges direct + co-occurrence methods  
    - top_k: how many substitutes to return  
    """
    try:
        raw_subs = await asyncio.to_thread(
            get_hybrid_substitutes,
            ingredient,
            context,
            top_k,
            use_hybrid=hybrid,
        )
    except Exception:
        logger.error("Substitution lookup failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve substitutes",
        )

    return SubstituteResponse(
        ingredient=ingredient,
        context=context,
        hybrid=hybrid,
        substitutes=raw_subs,
    )


@app.get(
    "/recipes/{recipe_title}",
    response_model=RecipeDetails,
    status_code=status.HTTP_200_OK,
    summary="Fetch full recipe details by title",
    tags=["recipes"],
)
async def get_recipe_details(
    recipe_title: str = Path(
        ...,
        description="Exact recipe title (case-insensitive)",
        example="Marinated Flank Steak Recipe",
    )
):
    """
    Returns detailed recipe data from Neo4j.
    """
    record = await asyncio.to_thread(fetch_recipe_details, recipe_title)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recipe '{recipe_title}' not found."
        )

    # Parse directions
    raw_dirs = record.get("directions", [])
    if isinstance(raw_dirs, str):
        try:
            raw_dirs = literal_eval(raw_dirs)
        except Exception:
            raw_dirs = [raw_dirs]

    # Parse & dedupe ingredients
    raw_ings = record.get("ingredients", []) or []
    seen = set()
    unique_ings: list[str] = []
    for ing in raw_ings:
        if ing not in seen:
            unique_ings.append(ing)
            seen.add(ing)

    return RecipeDetails(
        title=record.get("title", ""),
        directions=raw_dirs,
        link=record.get("link", ""),      # plain str, no HttpUrl validation
        source=record.get("source", ""),
        ingredients=unique_ings,
    )
