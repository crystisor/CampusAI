import pytest
from fastapi.testclient import TestClient
from src.web.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/system/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ollama" in data
    assert "qdrant" in data
    assert "gpu" in data

def test_subjects_listing():
    response = client.get("/api/subjects")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1

def test_create_subject():
    response = client.post("/api/subjects", data={"subject_id": "test_calculus", "name": "Test Calculus"})
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "test_calculus"
    assert data["name"] == "Test Calculus"

def test_dashboard_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "CampusAI" in response.text
    assert "Inspection Studio" in response.text
