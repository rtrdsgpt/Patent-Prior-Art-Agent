import mlflow
import pytest
from grounded_evals import RetrievalEvalReport

from config.settings import Settings
from evaluation.recall_eval import RecallEvalResult
from experiment_tracking import log_eval_run, log_report_run
from schema import ClaimElementComparison, FTOReport, InventionDisclosure, NoveltyAssessment


@pytest.fixture(autouse=True)
def isolated_mlflow_store(tmp_path, monkeypatch):
    # Real local (SQLite-backed -- see experiment_tracking.py's docstring for why not a
    # plain file store) MLflow tracking store under tmp_path, not the project's own
    # sqlite:///mlflow.db -- these tests verify our logging code against the real mlflow
    # library (read back what was actually recorded), just without polluting real
    # experiment history.
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    yield


def _latest_run(experiment_name: str):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
    return runs.iloc[0]


def test_log_eval_run_logs_params_and_overall_metrics():
    result = RecallEvalResult(
        k=10,
        num_cases=5,
        overall=RetrievalEvalReport(k=10, mean_recall_at_k=0.4, mrr=0.5, mean_ndcg_at_k=0.6),
        num_in_corpus_cases=0,
        in_corpus=None,
    )
    settings = Settings(embedding_model="test-embed-model", reranker_model="test-rerank-model", target_cpc_class="G06N3")

    log_eval_run(result, settings, num_patents_in_corpus=42)

    run = _latest_run("patent-agent-retrieval-eval")
    assert run["params.embedding_model"] == "test-embed-model"
    assert run["params.k"] == "10"
    assert run["params.num_patents_in_corpus"] == "42"
    assert run["metrics.overall_recall_at_k"] == 0.4
    assert run["metrics.overall_mrr"] == 0.5
    assert run["metrics.overall_ndcg_at_k"] == 0.6


def test_log_eval_run_omits_in_corpus_metrics_when_none():
    result = RecallEvalResult(
        k=5, num_cases=3, overall=RetrievalEvalReport(k=5, mean_recall_at_k=0.0, mrr=0.0, mean_ndcg_at_k=0.0), num_in_corpus_cases=0, in_corpus=None
    )

    log_eval_run(result, Settings(), num_patents_in_corpus=10)

    run = _latest_run("patent-agent-retrieval-eval")
    assert "metrics.in_corpus_recall_at_k" not in run or run.get("metrics.in_corpus_recall_at_k") != run.get("metrics.in_corpus_recall_at_k")  # NaN if absent


def test_log_eval_run_includes_in_corpus_metrics_when_present():
    result = RecallEvalResult(
        k=5,
        num_cases=3,
        overall=RetrievalEvalReport(k=5, mean_recall_at_k=0.1, mrr=0.1, mean_ndcg_at_k=0.1),
        num_in_corpus_cases=2,
        in_corpus=RetrievalEvalReport(k=5, mean_recall_at_k=0.8, mrr=0.9, mean_ndcg_at_k=0.85),
    )

    log_eval_run(result, Settings(), num_patents_in_corpus=10)

    run = _latest_run("patent-agent-retrieval-eval")
    assert run["metrics.in_corpus_recall_at_k"] == 0.8


def _report(comparisons_by_patent: dict) -> FTOReport:
    disclosure = InventionDisclosure(raw_text="x" * 50, technical_field="x", key_elements=[], candidate_cpc_classes=[])
    assessments = [
        NoveltyAssessment(candidate_patent_id=pid, element_comparisons=comparisons, citation_verified=verified)
        for pid, (comparisons, verified) in comparisons_by_patent.items()
    ]
    return FTOReport(disclosure=disclosure, assessments=assessments, summary="summary")


def _comparison(overlap_assessed: bool) -> ClaimElementComparison:
    return ClaimElementComparison(
        disclosure_element="x", candidate_patent_id="P", candidate_claim_number=1, cited_claim_text="x", overlap_explanation="x", overlap_assessed=overlap_assessed
    )


def test_log_report_run_logs_correct_counts():
    report = _report(
        {
            "P1": ([_comparison(True), _comparison(False)], True),
            "P2": ([_comparison(True)], False),
        }
    )

    log_report_run(report, Settings(groq_model="test-model"))

    run = _latest_run("patent-agent-report-generation")
    assert run["params.groq_model"] == "test-model"
    assert run["params.disclosure_text_length"] == "50"
    assert run["metrics.num_candidates_assessed"] == 2
    assert run["metrics.num_citation_verified"] == 1
    assert run["metrics.citation_verified_rate"] == 0.5
    assert run["metrics.num_overlap_assessed_true"] == 2


def test_log_report_run_handles_zero_assessments():
    report = _report({})
    log_report_run(report, Settings())

    run = _latest_run("patent-agent-report-generation")
    assert run["metrics.num_candidates_assessed"] == 0
    assert run["metrics.citation_verified_rate"] == 1.0


def test_log_report_run_swallows_mlflow_errors(monkeypatch):
    monkeypatch.setattr(mlflow, "start_run", lambda: (_ for _ in ()).throw(RuntimeError("mlflow is down")))
    report = _report({})

    log_report_run(report, Settings())  # should not raise
