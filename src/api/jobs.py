"""In-memory job store for the analyze/status/report API (todo.md section 3).

In-memory and process-local deliberately, not a placeholder for a database that got
skipped — there's no multi-worker deployment yet, and adding persistence now would be
speculative infrastructure for a requirement that doesn't exist yet. Swap for a real store
(Redis/Postgres) if/when the API needs to survive a restart or run behind more than one
worker process.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from threading import Lock

from pydantic import BaseModel

from schema import FTOReport


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(BaseModel):
    job_id: str
    status: JobStatus
    disclosure_text: str
    created_at: datetime
    report: FTOReport | None = None
    error: str | None = None


class JobStore:
    """Thread-safe in-memory job store. A lock guards the dict because FastAPI runs
    synchronous route handlers in a thread pool, so concurrent requests are real here even
    without a multi-process deployment."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()

    def create(self, disclosure_text: str) -> Job:
        job = Job(
            job_id=str(uuid.uuid4()),
            status=JobStatus.PENDING,
            disclosure_text=disclosure_text,
            created_at=datetime.now(timezone.utc),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            self._jobs[job_id].status = JobStatus.RUNNING

    def mark_completed(self, job_id: str, report: FTOReport) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.COMPLETED
            job.report = report

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.FAILED
            job.error = error
