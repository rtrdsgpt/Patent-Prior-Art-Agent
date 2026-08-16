"""MLflow experiment tracking (todo.md section 4): retrieval/reranking eval runs, and
FTOReport generation runs.

Defaults to a local SQLite-backed tracking store (`sqlite:///mlflow.db`) rather than a
hosted MLflow server — no separate service to stand up, and `mlflow ui --backend-store-uri
sqlite:///mlflow.db` reads the same local store directly for a real, inspectable dashboard.
**Not** the plain `./mlruns` file store MLflow examples usually default to — confirmed by
actually running this against a real MLflow install (not assumed from older docs): MLflow's
filesystem tracking backend is in maintenance mode as of this project's MLflow version and
raises on first use unless explicitly opted back into, so SQLite is the real default now, not
just a nicer one. Point `MLFLOW_TRACKING_URI` at a real server later without changing any
call site here — this only sets a default when that env var isn't already set, so an
explicitly configured environment (including these tests, which point it at an isolated
per-test SQLite file) always wins.

Named `experiment_tracking.py`, not `tracking.py`, specifically to avoid confusion with
`tracing.py` (OpenTelemetry request/span tracing) — related-sounding but different concerns:
one is "how did this eval/run score," the other is "what happened during this one pipeline
call."
"""

from __future__ import annotations

import logging
import os

import mlflow

from config.settings import Settings
from evaluation.recall_eval import RecallEvalResult
from schema import FTOReport

logger = logging.getLogger(__name__)

if not os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

_EVAL_EXPERIMENT = "patent-agent-retrieval-eval"
_REPORT_EXPERIMENT = "patent-agent-report-generation"


def log_eval_run(result: RecallEvalResult, settings: Settings, num_patents_in_corpus: int) -> None:
    """Log one `run_recall_eval()` result as an MLflow run — params identify what was
    evaluated (embedding/reranker model, corpus size, k), metrics are the actual scores, so
    changing the embedding model or chunking strategy and re-running the eval produces a
    directly comparable row in the MLflow UI, which is the whole point of tracking this at
    all (todo.md: "track retrieval/reranking experiments").
    """
    mlflow.set_experiment(_EVAL_EXPERIMENT)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "k": result.k,
                "embedding_model": settings.embedding_model,
                "reranker_model": settings.reranker_model,
                "target_cpc_class": settings.target_cpc_class,
                "num_patents_in_corpus": num_patents_in_corpus,
                "num_eval_cases": result.num_cases,
                "num_in_corpus_cases": result.num_in_corpus_cases,
            }
        )
        mlflow.log_metrics(
            {
                "overall_recall_at_k": result.overall.mean_recall_at_k,
                "overall_mrr": result.overall.mrr,
                "overall_ndcg_at_k": result.overall.mean_ndcg_at_k,
            }
        )
        if result.in_corpus:
            mlflow.log_metrics(
                {
                    "in_corpus_recall_at_k": result.in_corpus.mean_recall_at_k,
                    "in_corpus_mrr": result.in_corpus.mrr,
                    "in_corpus_ndcg_at_k": result.in_corpus.mean_ndcg_at_k,
                }
            )


def log_report_run(report: FTOReport, settings: Settings) -> None:
    """Log one `run_fto_pipeline()` result as an MLflow run (todo.md: "log report-generation
    runs"). Never lets a tracking failure break the actual pipeline — this is an
    observability side effect, not something a user's `/disclosure/analyze` call should ever
    fail over.
    """
    try:
        mlflow.set_experiment(_REPORT_EXPERIMENT)
        with mlflow.start_run():
            verified = [a for a in report.assessments if a.citation_verified]
            mlflow.log_params(
                {
                    "groq_model": settings.groq_model,
                    "disclosure_text_length": len(report.disclosure.raw_text),
                }
            )
            mlflow.log_metrics(
                {
                    "num_candidates_assessed": len(report.assessments),
                    "num_citation_verified": len(verified),
                    "citation_verified_rate": (len(verified) / len(report.assessments)) if report.assessments else 1.0,
                    "num_overlap_assessed_true": sum(
                        1 for a in report.assessments for c in a.element_comparisons if c.overlap_assessed
                    ),
                }
            )
    except Exception:
        logger.exception("MLflow logging failed for a report-generation run; continuing without it.")
