"""
esg_rag/bm25_search.py
----------------------
bm25_search(query, k=30, filters=None) -> list[SearchHit]

Wraps Qdrant sparse (BM25) search. Good at:
  - Exact terms: "Scope 1", "ESRS E1", tickers, fiscal years
  - Acronyms, standard names, precise numbers
  - Complementing dense search (which misses exact matches)
"""

from __future__ import annotations

from esg_rag.schemas import SearchHit
from esg_rag.store import QdrantStore

_store: QdrantStore | None = None


def _get_store() -> QdrantStore:
    global _store
    if _store is None:
        _store = QdrantStore()
    return _store


def bm25_search(
    query: str,
    k: int = 30,
    filters: dict[str, str] | None = None,
    doc_ids: list[str] | None = None,
) -> list[SearchHit]:
    """
    BM25 sparse search over the Qdrant index.

    Args:
        query:   raw query string (not embedded — BM25 uses token overlap)
        k:       number of results to return
        filters: e.g. {"company": "Apple", "year": "2024"}
        doc_ids: restrict to specific doc_ids (overrides filters)

    Returns:
        list[SearchHit] sorted by BM25 score descending, with rank set.
    """
    store = _get_store()
    hits = store.search_sparse(query, top_k=k, doc_ids=doc_ids, filters=filters)

    for i, h in enumerate(hits):
        h.rank = i + 1
        h.search_type = "bm25"

    return hits
