from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_scaffold():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "scaffold"


def test_invalid_question():
    assert client.post("/ask", json={"question": "x"}).status_code == 422


def test_not_yet_implemented():
    assert client.post("/ask", json={"question": "Which policy is current?"}).status_code == 501
