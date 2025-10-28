from fastapi.testclient import TestClient
import pytest
from demo_langchain_vec import app  # import your FastAPI app

client = TestClient(app)

def test_search():
    response = client.get("/search?sentenct=foo")
    assert response.status_code == 200
