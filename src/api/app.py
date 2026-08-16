"""FastAPI layer — todo.md section 3.

`POST /disclosure/analyze` → job id, `GET /jobs/{id}` → status, `GET /report/{id}` → FTO
report. Now that all five section-2 agents exist, a completed job holds a real `FTOReport` —
see `pipeline.py`'s `run_fto_analysis`.
"""

from __future__ import annotations

import logging

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from api.jobs import Job, JobStore
from api.pipeline import run_fto_analysis
from schema import FTOReport

logger = logging.getLogger(__name__)

app = FastAPI(title="Patent Prior-Art / FTO Agent API")
job_store = JobStore()


class AnalyzeRequest(BaseModel):
    disclosure_text: str


class AnalyzeResponse(BaseModel):
    job_id: str


def _run_job(job_id: str, disclosure_text: str) -> None:
    job_store.mark_running(job_id)
    try:
        report = run_fto_analysis(disclosure_text)
        job_store.mark_completed(job_id, report)
    except Exception as exc:  # noqa: BLE001 - a job failure must never crash the background task silently
        logger.exception("Job %s failed", job_id)
        job_store.mark_failed(job_id, str(exc))


def _get_job_or_404(job_id: str) -> Job:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job with id {job_id!r}")
    return job


@app.post("/disclosure/analyze", response_model=AnalyzeResponse)
def analyze_disclosure(request: AnalyzeRequest, background_tasks: BackgroundTasks) -> AnalyzeResponse:
    if not request.disclosure_text.strip():
        raise HTTPException(status_code=422, detail="disclosure_text must not be empty")

    job = job_store.create(request.disclosure_text)
    background_tasks.add_task(_run_job, job.job_id, request.disclosure_text)
    return AnalyzeResponse(job_id=job.job_id)


@app.get("/jobs/{job_id}", response_model=Job)
def get_job(job_id: str) -> Job:
    return _get_job_or_404(job_id)


@app.get("/report/{job_id}", response_model=FTOReport)
def get_report(job_id: str) -> FTOReport:
    job = _get_job_or_404(job_id)

    if job.status in ("pending", "running"):
        raise HTTPException(status_code=409, detail=f"Job {job_id} is not finished yet (status: {job.status.value})")
    if job.status == "failed":
        raise HTTPException(status_code=500, detail=f"Job {job_id} failed: {job.error}")

    return job.report
