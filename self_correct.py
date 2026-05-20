"""
esg_rag/self_correct.py
-----------------------
self_correcting_retrieve(query, filters, base_fn) -> list[SearchHit]

State machine that wraps any retrieve function and tries to improve
results if the initial retrieval evidence is weak.

States:
  1. Retrieve with full filters
  2. Score evidence quality (avg rerank score)
     - HIGH  (>0.5): return immediately
     - MEDIUM (0.2-0.5): relax filters (drop company/year constraints) and retry
     - LOW   (<0.2): rewrite query via Albert and retry with relaxed filters
     - STILL LOW: flag for web search (return with low_confidence=True)

This directly targets the RAGAS context_recall metric — if the corpus
has the answer but retrieval missed it, self-correction finds it.
"""

from __future__ import annotations

from typing import Callable

import esg_rag.albert as albert
from esg_rag.schemas import SearchHit

HIGH_THRESHOLD   = 0.5
MEDIUM_THRESHOLD = 0.2


def _evidence_score(hits: list[SearchHit]) -> float:
    """Average rerank score of the top hits (or raw score if not reranked)."""
    if not hits:
        return 0.0
    return sum(h.score for h in hits[:5]) / min(len(hits), 5)


def _rewrite_query(query: str) -> str:
    """Ask Albert to rephrase the query to improve retrieval."""
    try:
        return albert.chat_text(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a search query optimizer for ESG reports. "
                        "Rewrite the question as a concise search query using "
                        "different keywords. Output the rewritten query only."
                    ),
                },
                {"role": "user", "content": f"Original question: {query}"},
            ],
            model=albert.CHAT_MODEL,
            max_tokens=60,
            temperature=0.3,
            caller="self_correct",
        ).strip()
    except Exception:
        return query


def self_correcting_retrieve(
    query: str,
    filters: dict[str, str],
    base_fn: Callable[[str, dict[str, str]], list[SearchHit]],
) -> tuple[list[SearchHit], bool]:
    """
    Retrieve with automatic self-correction.

    Args:
        query:   original user question
        filters: initial filter dict (e.g. {"company": "Apple"})
        base_fn: the retrieve function to wrap — signature (query, filters) -> list[SearchHit]

    Returns:
        (hits, needs_web_search)
        needs_web_search=True signals the agent should call the web_search tool.
    """
    # ── Round 1: full filters ─────────────────────────────────────────────────
    hits = base_fn(query, filters)
    score = _evidence_score(hits)

    if score >= HIGH_THRESHOLD:
        return hits, False

    # ── Round 2: relax filters ────────────────────────────────────────────────
    relaxed_filters: dict[str, str] = {}  # drop company/year constraints
    hits2 = base_fn(query, relaxed_filters)
    score2 = _evidence_score(hits2)

    if score2 >= MEDIUM_THRESHOLD:
        print(f"  [self_correct] relaxed filters improved score "
              f"{score:.3f} → {score2:.3f}")
        return hits2, False

    # ── Round 3: rewrite query ────────────────────────────────────────────────
    rewritten = _rewrite_query(query)
    if rewritten != query:
        print(f"  [self_correct] rewriting query: '{query}' → '{rewritten}'")
        hits3 = base_fn(rewritten, relaxed_filters)
        score3 = _evidence_score(hits3)

        if score3 >= MEDIUM_THRESHOLD:
            return hits3, False

    # ── Still low — flag for web search ──────────────────────────────────────
    print(f"  [self_correct] evidence still weak ({score:.3f}) — flagging for web search")
    best = hits3 if 'hits3' in dir() and hits3 else (hits2 if hits2 else hits)
    return best, True
