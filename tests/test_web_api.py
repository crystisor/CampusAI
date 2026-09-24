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
    assert "btnDeleteCurrentSubject" in response.text
    assert "modalDeleteSubject" in response.text

def test_delete_subject():
    # 1. Create a subject to delete
    create_resp = client.post("/api/subjects", data={"subject_id": "subject_to_delete_test", "name": "Subject To Delete"})
    assert create_resp.status_code == 200

    # 2. Check it appears in listing
    list_resp = client.get("/api/subjects")
    assert list_resp.status_code == 200
    ids = [s["id"] for s in list_resp.json()]
    assert "subject_to_delete_test" in ids

    # 3. Delete the subject
    del_resp = client.delete("/api/subjects/subject_to_delete_test")
    assert del_resp.status_code == 200
    del_data = del_resp.json()
    assert del_data["status"] == "deleted"
    assert del_data["id"] == "subject_to_delete_test"

    # 4. Verify it's no longer in listing
    list_resp_after = client.get("/api/subjects")
    after_ids = [s["id"] for s in list_resp_after.json()]
    assert "subject_to_delete_test" not in after_ids

    # 5. Delete non-existent subject should return 404
    del_404 = client.delete("/api/subjects/non_existent_subject_xyz")
    assert del_404.status_code == 404

