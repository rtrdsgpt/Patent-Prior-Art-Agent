import pytest
from fastapi.testclient import TestClient

from api import app as app_module
from schema import FTOReport, InventionDisclosure

FAKE_DISCLOSURE = InventionDisclosure(raw_text="a neural network", technical_field="x", key_elements=[], candidate_cpc_classes=[])
FAKE_REPORT = FTOReport(disclosure=FAKE_DISCLOSURE, assessments=[], summary="The disclosure appears novel relative to the searched corpus.")


@pytest.fixture
def client(monkeypatch):
    # Patch out the real pipeline (real Groq calls + embedding/cross-encoder models) so
    # these tests verify the API's contract, not agent/retrieval accuracy — that's covered
    # by the agents' and retrieval stack's own test suites already.
    monkeypatch.setattr(app_module, "run_fto_analysis", lambda disclosure_text: FAKE_REPORT)
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
    assert body["report"]["summary"] == FAKE_REPORT.summary


def test_report_returns_full_fto_report_for_completed_job(client):
    job_id = client.post("/disclosure/analyze", json={"disclosure_text": "a neural network"}).json()["job_id"]
    response = client.get(f"/report/{job_id}")
    assert response.status_code == 200
    assert response.json()["summary"] == FAKE_REPORT.summary


def test_report_returns_404_for_unknown_job(client):
    response = client.get("/report/does-not-exist")
    assert response.status_code == 404


def test_report_returns_409_while_job_still_running(client):
    # Bypass POST /disclosure/analyze (whose background task runs synchronously under
    # TestClient, reaching "completed" before this test could observe "running") and drive
    # the job store directly to get a job genuinely stuck in a non-terminal state.
    job = app_module.job_store.create("a neural network")
    app_module.job_store.mark_running(job.job_id)

    response = client.get(f"/report/{job.job_id}")

    assert response.status_code == 409
    assert "running" in response.json()["detail"]


def test_report_returns_500_for_failed_job(client, monkeypatch):
    def _raise(disclosure_text):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module, "run_fto_analysis", _raise)

    job_id = client.post("/disclosure/analyze", json={"disclosure_text": "a neural network"}).json()["job_id"]
    response = client.get(f"/report/{job_id}")
    assert response.status_code == 500
    assert "boom" in response.json()["detail"]
