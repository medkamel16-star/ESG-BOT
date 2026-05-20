"""
esg_rag/retrieve.py
-------------------
retrieve(query, filters, config) -> RetrievalResult

The single entry point for the search layer. Reads pipeline_config.yaml
and runs the configured pipeline:

  router classify
    → parallel BM25 + dense (asyncio.gather)
    → RRF fusion (adaptive weights from query class)
    → Albert rerank
    → optional self-correcting wrap

Every module in this pipeline is independently togglable via pipeline_config.yaml.
This is how A/B tests work: flip a flag, run the eval harness, check scoreboard.

Usage:
  from esg_rag.retrieve import retrieve
  result = retrieve("What are Apple's Scope 1 emissions?")
  for hit in result.hits:
      print(hit.page, hit.text[:100])
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import yaml

from esg_rag.schemas import QueryClass, RetrievalResult, SearchHit
from esg_rag.bm25_search import bm25_search
from esg_rag.dense_search import dense_search
from esg_rag.fusion import rrf_fuse, weights_for_class
from esg_rag.rerank import rerank_albert
from esg_rag.router import classify_query
from esg_rag.self_correct import self_correcting_retrieve

CONFIG_PATH = Path("pipeline_config.yaml")


def _load_cfg() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return {}


# ── Async parallel search ─────────────────────────────────────────────────────

async def _parallel_search(
    query: str,
    filters: dict[str, str],
    doc_ids: list[str] | None,
    dense_k: int,
    sparse_k: int,
    use_bm25: bool,
    use_dense: bool,
) -> dict[str, list[SearchHit]]:
    """Run BM25 and dense search in parallel via asyncio."""
    loop = asyncio.get_event_loop()
    results: dict[str, list[SearchHit]] = {}

    tasks = []
    labels = []

    if use_dense:
        tasks.append(loop.run_in_executor(
            None, lambda: dense_search(query, k=dense_k, filters=filters, doc_ids=doc_ids)
        ))
        labels.append("dense")

    if use_bm25:
        tasks.append(loop.run_in_executor(
            None, lambda: bm25_search(query, k=sparse_k, filters=filters, doc_ids=doc_ids)
        ))
        labels.append("bm25")

    completed = await asyncio.gather(*tasks)
    for label, hits in zip(labels, completed):
        results[label] = hits

    return results


# ── Core pipeline (non-self-correcting) ──────────────────────────────────────

def _core_retrieve(
    query: str,
    filters: dict[str, str],
    cfg: dict[str, Any],
    query_class: QueryClass,
) -> list[SearchHit]:
    """Run the configured pipeline once."""
    r_cfg = cfg.get("retrieval", {})
    use_dense   = r_cfg.get("use_dense", True)
    use_bm25    = r_cfg.get("use_bm25", False)
    use_rerank  = r_cfg.get("use_reranker", False)
    dense_k     = r_cfg.get("dense_top_k", 30)
    sparse_k    = r_cfg.get("sparse_top_k", 30)
    rerank_k    = r_cfg.get("rerank_top_k", 8)
    rrf_k_val   = r_cfg.get("rrf_k", 60)
    fusion_mode = r_cfg.get("fusion", "rrf")

    doc_ids = None  # future: agent can pass explicit doc_ids

    # ── Parallel search
    results_by_method = asyncio.run(
        _parallel_search(query, filters, doc_ids, dense_k, sparse_k, use_bm25, use_dense)
    )

    # ── Fusion
    if len(results_by_method) > 1:
        adaptive_weights = weights_for_class(query_class.label)
        hits = rrf_fuse(results_by_method, k=rrf_k_val, weights=adaptive_weights)
    elif results_by_method:
        hits = list(results_by_method.values())[0]
    else:
        return []

    # ── Rerank
    if use_rerank and hits:
        hits = rerank_albert(query, hits, top_n=rerank_k)
    else:
        hits = hits[:rerank_k]

    return hits


# ── Public entry point ────────────────────────────────────────────────────────

def retrieve(
    query: str,
    filters: dict[str, str] | None = None,
    config: dict[str, Any] | None = None,
) -> RetrievalResult:
    """
    Main retrieval entry point.

    Args:
        query:   user question
        filters: optional {"company": "Apple", "year": "2024"}
        config:  pipeline config dict (loaded from pipeline_config.yaml if None)

    Returns:
        RetrievalResult with .hits (best chunks) and .telemetry
    """
    t0 = time.perf_counter()
    cfg = config or _load_cfg()
    r_cfg = cfg.get("retrieval", {})
    filters = filters or {}

    # ── Route
    use_router = r_cfg.get("use_router", False)
    if use_router:
        query_class = classify_query(query)
        # Merge router-suggested filters with caller-supplied filters
        merged_filters = {**query_class.suggested_filters, **filters}
    else:
        query_class = QueryClass("factual_lookup", 1.0, filters, 8)
        merged_filters = filters

    # ── Self-correcting retrieve or direct
    use_self_correct = r_cfg.get("use_self_correct", False)
    needs_web = False

    if use_self_correct:
        def _base(q, f):
            return _core_retrieve(q, f, cfg, query_class)
        hits, needs_web = self_correcting_retrieve(query, merged_filters, _base)
    else:
        hits = _core_retrieve(query, merged_filters, cfg, query_class)

    elapsed = (time.perf_counter() - t0) * 1000

    return RetrievalResult(
        question=query,
        hits=hits,
        query_class=query_class,
        telemetry={
            "latency_ms": round(elapsed, 1),
            "n_hits": len(hits),
            "needs_web_search": needs_web,
            "filters_used": merged_filters,
            "query_class": query_class.label,
            "use_bm25": r_cfg.get("use_bm25", False),
            "use_reranker": r_cfg.get("use_reranker", False),
        },
    )
