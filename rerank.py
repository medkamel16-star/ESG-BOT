"""
esg_rag/rerank.py
-----------------
rerank_albert(query, candidates, top_n=8) -> list[SearchHit]

Takes the top-30 fused candidates and asks Albert's /rerank endpoint
to score them with a cross-encoder. Returns the top_n best hits with
scores min-max normalised to [0, 1].

Cross-encoders read query + document together, which is much more
accurate than the embedding similarity used in retrieval — but too slow
to run on the full corpus, so we only rerank the 30 fused candidates.
"""

from __future__ import annotations

import esg_rag.albert as albert
from esg_rag.schemas import SearchHit


def rerank_albert(
    query: str,
    candidates: list[SearchHit],
    top_n: int = 8,
) -> list[SearchHit]:
    """
    Rerank candidates using Albert's cross-encoder.

    Args:
        query:      the original user question
        candidates: list[SearchHit] from fusion (typically 30 items)
        top_n:      how many to keep after reranking

    Returns:
        top_n SearchHits sorted by rerank score descending.
        Scores are min-max normalised to [0, 1].
        hit.search_type is set to "reranked".
    """
    if not candidates:
        return []

    # Use full_text for reranking — includes contextual prefix
    docs = [c.full_text[:1000] for c in candidates]

    ranked = albert.rerank(query, docs, top_n=top_n, caller="rerank")
    # albert.rerank already returns normalised scores sorted by score desc

    result: list[SearchHit] = []
    for new_rank, r in enumerate(ranked):
        hit = candidates[r["index"]]
        hit.score = r["score"]
        hit.search_type = "reranked"
        hit.rank = new_rank + 1
        result.append(hit)

    return result
