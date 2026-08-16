"""CLI entry point: run the recall@k eval against the ingested corpus and print results.

    python -m evaluation.run_eval [--k K] [--sample-size N]

No Groq calls (see `recall_eval.py`'s docstring for why), so this is safe/cheap to rerun
whenever the corpus, embedding model, chunking, or reranker changes — exactly the
"retrieval/reranking experiments" todo.md section 4's MLflow bullet wants tracked. MLflow
run-logging isn't wired in yet (section 4 still open); this prints a plain report for now.
"""

from __future__ import annotations

import argparse
import logging

from config.settings import get_settings
from evaluation.recall_eval import build_eval_set, run_recall_eval
from ingestion.corpus import load_corpus
from retrieval.bm25_index import build_bm25_index
from retrieval.embedding_index import build_embedding_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall@k eval against real examiner-cited prior art.")
    parser.add_argument("--k", type=int, default=10, help="k for recall@k / nDCG@k (default: 10)")
    parser.add_argument("--sample-size", type=int, default=None, help="Cap the number of eval cases (default: all)")
    args = parser.parse_args()

    settings = get_settings()
    patents = load_corpus(settings)
    logger.info("Loaded %d patents from the corpus.", len(patents))

    eval_cases = build_eval_set(patents)
    logger.info("Built %d eval cases (patents with at least one real examiner citation).", len(eval_cases))
    if not eval_cases:
        logger.warning("No eval cases — no patents in the corpus have examiner citations. Nothing to score.")
        return

    logger.info("Indexing corpus for retrieval...")
    bm25_index = build_bm25_index(patents)
    embedding_collection = build_embedding_index(patents, settings=settings)
    patents_by_id = {p.patent_id: p for p in patents}

    result = run_recall_eval(eval_cases, bm25_index, embedding_collection, patents_by_id, settings=settings, k=args.k, sample_size=args.sample_size)

    print(f"\n=== Recall@{result.k} eval — {result.num_cases} cases ===")
    print(f"Overall (against ALL real examiner citations, including ones outside this {len(patents)}-patent corpus):")
    print(f"  recall@{result.k}: {result.overall.mean_recall_at_k:.3f}")
    print(f"  MRR:       {result.overall.mrr:.3f}")
    print(f"  nDCG@{result.k}:   {result.overall.mean_ndcg_at_k:.3f}")

    if result.in_corpus:
        print(f"\nIn-corpus only ({result.num_in_corpus_cases} cases with a citation actually present in the index):")
        print(f"  recall@{result.k}: {result.in_corpus.mean_recall_at_k:.3f}")
        print(f"  MRR:       {result.in_corpus.mrr:.3f}")
        print(f"  nDCG@{result.k}:   {result.in_corpus.mean_ndcg_at_k:.3f}")
    else:
        print("\nNo eval case had a citation present in the indexed corpus — in-corpus metric unavailable.")


if __name__ == "__main__":
    main()
