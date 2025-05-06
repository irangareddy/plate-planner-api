
from pydantic import BaseModel


class SubstituteRequest(BaseModel):
    ingredient: str
    top_k: int = 5

class PantryRequest(BaseModel):
    pantry: list[str]
    top_k: int = 10
