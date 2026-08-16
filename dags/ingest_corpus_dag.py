"""Airflow DAG (todo.md section 4): scheduled ingestion of patents in the target CPC class
from Google Patents Public Data — download → clean/chunk → embed/index.

Both tasks call the *same* functions the manual CLI paths use
(`ingestion.ingest_corpus.main()`, `retrieval.embedding_index.build_embedding_index()`) —
this DAG is a scheduling wrapper around the one real ingestion/indexing path, not a second,
DAG-specific reimplementation of it.

**Why "scheduled" here means "re-run the same bounded seed+expand fetch periodically," not a
true delta/watermark-based incremental load:** `bigquery_client.py`'s seed query is a plain
`LIMIT`, with no `ORDER BY` and therefore no stable cursor to track "new patents since last
run" against. Re-running the same bounded fetch on a schedule is what "incremental" can
honestly mean without first adding a `publication_date`-ordered cursor to the seed query —
real future work if this ever needs to track true deltas instead of periodically refreshing
a bounded sample, not implemented now since nothing downstream needs it yet.

**`embed_and_index` writes to the same on-disk Chroma path (`chroma_persist_directory`) the
API's `PersistentClient` reads from** (see `api/pipeline.py`'s `_build_chroma_client`) — but
`_get_indexes()` only builds/loads that once per API process (`functools.lru_cache`), so a
freshly re-indexed corpus on disk isn't picked up by an already-running API process without
a restart. Hot-reloading the API's indexes is out of scope here; this DAG's job is to keep
the on-disk artifact current, not to push it into a running process.
"""

from __future__ import annotations

from datetime import datetime

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator


def _run_ingestion() -> None:
    from ingestion.ingest_corpus import main

    main()


def _embed_and_index() -> None:
    import chromadb

    from config.settings import get_settings
    from ingestion.corpus import load_corpus
    from retrieval.embedding_index import build_embedding_index

    settings = get_settings()
    patents = load_corpus(settings)
    client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
    build_embedding_index(patents, settings=settings, client=client)


with DAG(
    dag_id="ingest_corpus_dag",
    description="Ingest patents in the target CPC class from Google Patents Public Data (BigQuery), then refresh the on-disk embedding index.",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["patent-agent", "ingestion"],
) as dag:
    ingest_task = PythonOperator(task_id="ingest_corpus", python_callable=_run_ingestion)
    embed_task = PythonOperator(task_id="embed_and_index", python_callable=_embed_and_index)

    ingest_task >> embed_task
