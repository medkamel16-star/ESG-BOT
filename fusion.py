"""
esg_rag/fusion.py
-----------------
rrf_fuse(results_by_method, k=60, weights=None) -> list[SearchHit]

Merges ranked lists from BM25, dense (and optionally ColBERT) using
Reciprocal Rank Fusion (RRF).

RRF formula for each candidate:
  score = sum over methods: weight_m / (k + rank_m)

Where rank_m is 1-based position in method m's result list.
Candidates not appearing in a method get rank = len(that list) + 1.

weights: dict of method name → float, e.g. {"bm25": 0.3, "dense": 0.7}
         Default: equal weights for all methods present.

The router (3.6) passes adaptive weights based on query class:
  factual_lookup      → bm25 upweighted  (exact terms matter)
  cross_doc_compare   → equal
  numeric_computation → bm25 upweighted
  regulatory_check    → dense upweighted (concept matching)
  out_of_corpus       → dense only (no point BM25 on nothing)
"""

from __future__ import annotations

from collections import defaultdict
from esg_rag.schemas import SearchHit


def rrf_fuse(
    results_by_method: dict[str, list[SearchHit]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[SearchHit]:
    """
    Fuse multiple ranked lists into one using RRF.

    Args:
        results_by_method: {"bm25": [...], "dense": [...], ...}
        k:       RRF constant (higher → less reward for top ranks)
        weights: per-method multipliers. Default: 1.0 for all methods.

    Returns:
        Deduplicated list[SearchHit] sorted by RRF score descending.
        The returned hits carry search_type="fused" and rank set.
    """
    if not results_by_method:
        return []

    methods = list(results_by_method.keys())
    if weights is None:
        weights = {m: 1.0 for m in methods}
    else:
        # Fill missing methods with 1.0
        for m in methods:
            weights.setdefault(m, 1.0)

    # Build a lookup: (doc_id, chunk_index) → best SearchHit object
    # We keep the hit from the method where it ranked highest
    best_hit: dict[tuple[str, int], SearchHit] = {}
    rrf_scores: dict[tuple[str, int], float] = defaultdict(float)

    for method, hits in results_by_method.items():
        w = weights.get(method, 1.0)
        n = len(hits)
        # Assign "infinity rank" to items not present in this list
        # (handled implicitly — we only iterate what's present)
        for rank_0, hit in enumerate(hits):
            key = (hit.doc_id, hit.chunk_index)
            rank_1 = rank_0 + 1
            rrf_scores[key] += w / (k + rank_1)

            # Keep whichever hit has the highest raw score for display
            if key not in best_hit or hit.score > best_hit[key].score:
                best_hit[key] = hit

    # Sort by RRF score
    ranked_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    fused: list[SearchHit] = []
    for rank_0, key in enumerate(ranked_keys):
        hit = best_hit[key]
        hit.score = round(rrf_scores[key], 6)
        hit.search_type = "fused"
        hit.rank = rank_0 + 1
        fused.append(hit)

    return fused


# ── Adaptive weights from query class ────────────────────────────────────────

WEIGHTS_BY_CLASS: dict[str, dict[str, float]] = {
    "factual_lookup":      {"bm25": 0.6, "dense": 0.4},
    "numeric_computation": {"bm25": 0.7, "dense": 0.3},
    "cross_doc_compare":   {"bm25": 0.5, "dense": 0.5},
    "regulatory_check":    {"bm25": 0.3, "dense": 0.7},
    "out_of_corpus":       {"bm25": 0.0, "dense": 1.0},
}


def weights_for_class(query_class_label: str) -> dict[str, float]:
    """Return fusion weights appropriate for a given query class label."""
    return WEIGHTS_BY_CLASS.get(query_class_label, {"bm25": 0.5, "dense": 0.5})
