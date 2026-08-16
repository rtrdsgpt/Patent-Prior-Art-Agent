"""Structural checks for dags/ingest_corpus_dag.py.

Airflow is deliberately NOT installed in this project's main venv/requirements (it runs in
its own container — see Dockerfile.airflow and that DAG's own docstring for why: avoiding
exposing the main app's dependency set to Airflow's own pins). So these tests
`pytest.importorskip` and skip locally rather than fail — they exist to validate DAG
structure wherever Airflow *is* available (the Airflow container itself, or a CI job that
installs requirements-airflow.txt), not to force an unrelated dependency into local dev.
The DAG's actual task logic is exercised for real via `airflow dags test` inside the built
Airflow image (see log.md) — a stronger check than importing the module could give anyway,
since it runs the real callables, not just imports them.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("airflow")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dags"))

from ingest_corpus_dag import dag  # noqa: E402


def test_dag_has_expected_id():
    assert dag.dag_id == "ingest_corpus_dag"


def test_dag_has_two_tasks():
    assert set(dag.task_ids) == {"ingest_corpus", "embed_and_index"}


def test_ingest_runs_before_embed_and_index():
    ingest_task = dag.get_task("ingest_corpus")
    assert "embed_and_index" in ingest_task.downstream_task_ids


def test_dag_does_not_backfill():
    assert dag.catchup is False
