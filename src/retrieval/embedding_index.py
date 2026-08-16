"""Dense (embedding) retrieval over the claim-level chunk index.

Uses Chroma as the vector store and a local sentence-transformers model for embeddings —
no external API calls, so this works fully offline once the model weights are cached
locally (see log.md for why nothing here depends on the paused Groq/BigQuery credentials).

One chunk per claim (`chunking.claim_to_index_chunk`), not per patent — see
`ingestion/chunking.py`'s module docstring for why claim-level is the chunk boundary used
throughout the pipeline.
"""

from __future__ import annotations

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from config.settings import Settings, get_settings
from ingestion.chunking import claim_to_index_chunk
from schema import Patent, SearchResult

_COLLECTION_NAME = "patent_claims"


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """Wraps a local sentence-transformers model as a Chroma embedding function.

    Explicit wrapper (rather than Chroma's built-in default embedding function) so the
    model is the one named in `Settings.embedding_model`, not whatever Chroma defaults to.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 - Chroma's required param name
        return self._model.encode(list(input)).tolist()

    @staticmethod
    def name() -> str:
        return "sentence-transformers"

    def get_config(self) -> dict:
        return {"model_name": self._model_name}

    @staticmethod
    def build_from_config(config: dict) -> "SentenceTransformerEmbeddingFunction":
        return SentenceTransformerEmbeddingFunction(config["model_name"])


def _chunk_id(patent_id: str, claim_number: int) -> str:
    return f"{patent_id}::claim-{claim_number}"


def build_embedding_index(
    patents: list[Patent],
    settings: Settings | None = None,
    client: chromadb.ClientAPI | None = None,
) -> Collection:
    """Embed every claim of every patent and index it in Chroma.

    `client` defaults to an in-memory ephemeral client — pass a `PersistentClient` (using
    `settings.chroma_persist_directory`) to persist across runs. Kept as a parameter rather
    than hardcoded so tests don't write to disk.
    """
    settings = settings or get_settings()
    client = client or chromadb.EphemeralClient()
    embedding_fn = SentenceTransformerEmbeddingFunction(settings.embedding_model)

    collection = client.get_or_create_collection(name=_COLLECTION_NAME, embedding_function=embedding_fn)

    ids, documents, metadatas = [], [], []
    for patent in patents:
        for claim in patent.claims:
            ids.append(_chunk_id(patent.patent_id, claim.claim_number))
            documents.append(claim_to_index_chunk(claim, patent.title))
            metadatas.append({"patent_id": patent.patent_id, "claim_number": claim.claim_number})

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    return collection


def dense_search(collection: Collection, query: str, top_k: int) -> list[SearchResult]:
    """Query the embedding index, returning one `SearchResult` per patent.

    A patent can have multiple claims match; only the best-scoring claim per patent is
    kept, since search results are ranked at the patent level (a candidate patent is either
    worth comparing against the disclosure or it isn't — which claim tipped it off is the
    comparison agent's concern, not the search agent's).
    """
    # Over-fetch chunks since multiple claims from the same patent can appear before dedup.
    results = collection.query(query_texts=[query], n_results=max(top_k * 5, top_k))

    best_by_patent: dict[str, float] = {}
    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    for _id, distance, metadata in zip(ids, distances, metadatas):
        patent_id = metadata["patent_id"]
        score = 1.0 / (1.0 + distance)  # convert distance to a similarity-style score, higher is better
        if patent_id not in best_by_patent or score > best_by_patent[patent_id]:
            best_by_patent[patent_id] = score

    ranked = sorted(best_by_patent.items(), key=lambda kv: kv[1], reverse=True)
    return [
        SearchResult(patent_id=patent_id, score=score, retrieval_method="dense")
        for patent_id, score in ranked[:top_k]
    ]
