import pytest
from fastapi.testclient import TestClient

from patent_agent.api import app as app_module
from patent_agent.schema import SearchResult


@pytest.fixture
def client(monkeypatch):
    # Patch out the real pipeline (real embedding + cross-encoder models) so these tests
    # verify the API's contract, not the retrieval stack's accuracy — that's covered by
    # test_hybrid.py / test_reranker.py already.
    monkeypatch.setattr(
        app_module,
        "run_prior_art_search",
        lambda disclosure_text: [SearchResult(patent_id="US10000001B2", score=0.9, retrieval_method="reranked")],
    )
    app_module.job_store._jobs.clear()
    return TestClient(app_module.app)


def test_analyze_returns_job_id(client):
    response = client.post("/disclosure/analyze", json={"disclosure_text": "a neural network that uses dropout"})
    assert response.status_code == 200
    assert "job_id" in response.json()


def test_analyze_rejects_empty_disclosure(client):
    response = client.post("/disclosure/analyze", json={"disclosure_text": "   "})
    assert response.status_code == 422


def test_get_job_returns_404_for_unknown_job(client):
    response = client.get("/jobs/does-not-exist")
    assert response.status_code == 404


def test_analyze_then_get_job_reaches_completed(client):
    job_id = client.post("/disclosure/analyze", json={"disclosure_text": "a neural network"}).json()["job_id"]
    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["candidate_patents"][0]["patent_id"] == "US10000001B2"


def test_report_returns_501_with_candidates_for_completed_job(client):
    job_id = client.post("/disclosure/analyze", json={"disclosure_text": "a neural network"}).json()["job_id"]
    response = client.get(f"/report/{job_id}")
    assert response.status_code == 501
    detail = response.json()["detail"]
    assert "comparison and risk-report agents" in detail["message"]
    assert detail["candidate_patents"][0]["patent_id"] == "US10000001B2"


def test_report_returns_404_for_unknown_job(client):
    response = client.get("/report/does-not-exist")
    assert response.status_code == 404
