from api.jobs import JobStatus, JobStore
from schema import FTOReport, InventionDisclosure

FAKE_REPORT = FTOReport(
    disclosure=InventionDisclosure(raw_text="x", technical_field="x", key_elements=[], candidate_cpc_classes=[]),
    assessments=[],
    summary="summary",
)


def test_create_job_starts_pending():
    store = JobStore()
    job = store.create("a disclosure")
    assert job.status == JobStatus.PENDING
    assert job.disclosure_text == "a disclosure"


def test_get_returns_none_for_unknown_job():
    store = JobStore()
    assert store.get("does-not-exist") is None


def test_mark_running_updates_status():
    store = JobStore()
    job = store.create("x")
    store.mark_running(job.job_id)
    assert store.get(job.job_id).status == JobStatus.RUNNING


def test_mark_completed_sets_status_and_report():
    store = JobStore()
    job = store.create("x")
    store.mark_completed(job.job_id, FAKE_REPORT)
    updated = store.get(job.job_id)
    assert updated.status == JobStatus.COMPLETED
    assert updated.report == FAKE_REPORT


def test_mark_failed_sets_status_and_error():
    store = JobStore()
    job = store.create("x")
    store.mark_failed(job.job_id, "boom")
    updated = store.get(job.job_id)
    assert updated.status == JobStatus.FAILED
    assert updated.error == "boom"


def test_each_job_gets_a_unique_id():
    store = JobStore()
    job_a = store.create("a")
    job_b = store.create("b")
    assert job_a.job_id != job_b.job_id
