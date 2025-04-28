from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)

def test_suggest_meals_italian():
    response = client.get(
        "/api/v1/suggest",
        params={"pantry": "['pasta', 'tomato', 'cheese']", "cuisine": "Italian"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "suggested_meals" in data
    assert len(data["suggested_meals"]) > 0
    assert data["suggested_meals"][0]["cuisine"] == "Italian"

def test_suggest_meals_empty():
    response = client.get(
        "/api/v1/suggest",
        params={"pantry": "['rice']", "cuisine": "Mexican"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "suggested_meals" in data
    assert len(data["suggested_meals"]) == 0
