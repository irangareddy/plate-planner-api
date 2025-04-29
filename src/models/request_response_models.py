from pydantic import BaseModel
from typing import List

class SubstituteRequest(BaseModel):
    ingredient: str
    top_k: int = 5

class PantryRequest(BaseModel):
    pantry: List[str]
    top_k: int = 10
