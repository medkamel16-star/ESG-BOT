"""
esg_rag/schemas.py
------------------
Single source of truth for all dataclasses / typed dicts in the project.
Every module imports from here — never define data shapes elsewhere.

Hierarchy:
  Chunk           → produced by parsing.py / chunking.py
  SearchHit       → produced by bm25_search.py / dense_search.py
  RetrievalResult → produced by retrieve.py (fused + reranked)
  QueryClass      → produced by router.py
  AgentResult     → produced by agents/
  CSRDDatapoint   → produced by csrd_autofill.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Document / Chunk ─────────────────────────────────────────────────────────

@dataclass
class Chunk:
    # Identity
    doc_id: str            # sha256[:12] of the PDF bytes
    chunk_index: int       # sequential index within doc

    # Document metadata
    company: str
    year: str
    report_type: str       # "sustainability" | "annual" | "csrd" | "unknown"
    source_pdf: str        # filename only

    # Content
    text: str
    page: int
    section: str = ""      # heading path e.g. "3. Environment > 3.2 Emissions"

    # Structure flags
    has_table: bool = False
    table_data: list[dict[str, Any]] = field(default_factory=list)
    is_figure: bool = False
    figure_caption: str = ""
    image_path: str = ""   # path to cropped figure PNG (for VLM)
    vlm_description: str = ""  # filled by vlm_figures.py (GPU, optional)

    # Quality / enrichment (filled by later pipeline stages)
    quality_score: float = 0.0    # 2.3 chunk_quality.py
    context_prefix: str = ""      # 2.4 contextual.py

    # Embedding (filled just before upsert)
    embedding: list[float] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Text as embedded: context prefix + content."""
        if self.context_prefix:
            return f"{self.context_prefix}\n\n{self.text}"
        return self.text

    @property
    def char_count(self) -> int:
        return len(self.text)


# ── Search ────────────────────────────────────────────────────────────────────

@dataclass
class SearchHit:
    """A single result from BM25 or dense search, before fusion/reranking."""
    doc_id: str
    chunk_index: int
    company: str
    year: str
    report_type: str
    source_pdf: str
    page: int
    section: str
    text: str
    context_prefix: str
    has_table: bool
    table_data: list[dict[str, Any]]
    quality_score: float

    score: float           # raw search score (BM25 or cosine)
    search_type: str       # "bm25" | "dense" | "colbert"
    rank: int = 0          # rank in its own result list

    @property
    def full_text(self) -> str:
        if self.context_prefix:
            return f"{self.context_prefix}\n\n{self.text}"
        return self.text

    @classmethod
    def from_payload(cls, payload: dict[str, Any], score: float, search_type: str) -> "SearchHit":
        """Build a SearchHit from a Qdrant point payload."""
        import json
        raw_td = payload.get("table_data", "[]")
        table_data = json.loads(raw_td) if isinstance(raw_td, str) else raw_td
        return cls(
            doc_id=payload.get("doc_id", ""),
            chunk_index=payload.get("chunk_index", 0),
            company=payload.get("company", ""),
            year=payload.get("year", ""),
            report_type=payload.get("report_type", ""),
            source_pdf=payload.get("source_pdf", ""),
            page=payload.get("page", 0),
            section=payload.get("section", ""),
            text=payload.get("text", ""),
            context_prefix=payload.get("context_prefix", ""),
            has_table=payload.get("has_table", False),
            table_data=table_data,
            quality_score=payload.get("quality_score", 0.0),
            score=score,
            search_type=search_type,
        )


@dataclass
class RetrievalResult:
    """Final output of retrieve.py — fused, reranked, ready for the agent."""
    question: str
    hits: list[SearchHit]                    # top-k chunks, best first
    query_class: "QueryClass | None" = None
    telemetry: dict[str, Any] = field(default_factory=dict)
    # telemetry keys: bm25_candidates, dense_candidates, rerank_score_top,
    #                 latency_ms, albert_calls


# ── Query routing ─────────────────────────────────────────────────────────────

@dataclass
class QueryClass:
    """Output of router.py — classifies what kind of question this is."""
    label: str              # "factual_lookup" | "cross_doc_compare" |
                            # "numeric_computation" | "regulatory_check" |
                            # "out_of_corpus"
    confidence: float       # 0.0–1.0
    suggested_filters: dict[str, str] = field(default_factory=dict)
    suggested_top_k: int = 8


# ── Agent ─────────────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    """Return value of agents/*.run_agent()."""
    final_output: str
    tool_trace: list[dict[str, Any]]    # [{tool, args, result, latency_ms}]
    iterations_used: int
    tokens: dict[str, int] = field(default_factory=dict)  # {in, out}
    retrieved_contexts: list[SearchHit] = field(default_factory=list)
    verifier_result: dict[str, Any] = field(default_factory=dict)


# ── CSRD ──────────────────────────────────────────────────────────────────────

@dataclass
class CSRDDatapoint:
    """One filled cell in the CSRD auto-fill form."""
    esrs_id: str           # e.g. "E1-6_GHG_Scope1"
    value: str             # extracted value (or "not disclosed")
    unit: str              # e.g. "tCO2e", "%", "EUR"
    source_page: int
    source_doc_id: str
    confidence: float      # 0.0–1.0
