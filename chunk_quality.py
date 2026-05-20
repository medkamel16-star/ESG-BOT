"""
esg_rag/chunk_quality.py
------------------------
score_chunks(chunks) -> chunks

Scores each Chunk 0.0–1.0 and sets chunk.quality_score in place.
Used as a tiebreaker during retrieval and to filter near-empty chunks.

Scoring rubric (matches build guide spec exactly):
  entity_density   → up to +0.4  (ESG entities: companies, metrics, standards)
  section_depth    → up to +0.2  (deeper heading hierarchy = better signal)
  table_adjacency  → up to +0.2  (chunk IS a table, or contains table-like text)
  length_band      → up to +0.2  (800–3000 chars is the sweet spot)
"""

from __future__ import annotations

import re
from esg_rag.schemas import Chunk

# ── Patterns ──────────────────────────────────────────────────────────────────

# ESG-domain entities and units
_ESG_ENTITY = re.compile(
    r'\b(Scope [123]|GHG|CO2|tCO2e?|CSRD|ESRS|ESG|CDP|TCFD|SBTi|EU Taxonomy|'
    r'DNSH|Net Zero|carbon neutral|RE100|SDG\s*\d+|GRI\s*\d+|SASB|UNGC)\b',
    re.IGNORECASE,
)
_NUMBERS = re.compile(r'\b\d[\d,.\-/%]*\b')
_UNITS   = re.compile(r'\b(tCO2e?|GWh|MWh|kWh|Mt|kt|m³|€|USD|\$|%|GJ|TJ|MW|TWh)\b')
_NAMED_ENT = re.compile(r'\b([A-Z][a-z]+ ){1,3}[A-Z][a-z]+\b')  # "Total Energies"


def _entity_density(text: str) -> float:
    """
    Fraction of "signal" tokens (ESG terms + numbers + units + named entities)
    normalised so that a chunk with ~1 signal token per 50 chars scores ~1.0.
    """
    if not text:
        return 0.0
    esg_hits    = len(_ESG_ENTITY.findall(text))
    number_hits = len(_NUMBERS.findall(text))
    unit_hits   = len(_UNITS.findall(text))
    entity_hits = len(_NAMED_ENT.findall(text))
    total = esg_hits * 2 + number_hits + unit_hits + entity_hits
    # Normalise: ~1 signal per 50 chars → score 1.0
    expected = max(len(text) / 50, 1)
    return min(total / expected, 1.0)


def _section_depth(section: str) -> float:
    """Deeper heading path → higher score. 0 depth → 0, 3+ levels → 1.0."""
    depth = section.count(">") if section else 0
    return min(depth / 3.0, 1.0)


def _table_adjacency(chunk: Chunk) -> float:
    """
    1.0 if chunk IS a table.
    0.5 if chunk text looks like it contains table-like content (many | or tabs).
    0.0 otherwise.
    """
    if chunk.has_table:
        return 1.0
    pipe_density = chunk.text.count("|") / max(len(chunk.text), 1)
    if pipe_density > 0.02:
        return 0.5
    return 0.0


def _length_score(text: str) -> float:
    """
    Sweet spot is 800–3000 chars.
    Below 800: linear ramp from 0 → 1.
    800–3000:  1.0.
    Above 3000: linear decay.
    """
    n = len(text)
    if n < 80:
        return 0.0
    if n < 800:
        return n / 800.0
    if n <= 3000:
        return 1.0
    # Decay for very long chunks (shouldn't appear after chunking.py)
    return max(0.3, 1.0 - (n - 3000) / 5000.0)


# ── Public API ────────────────────────────────────────────────────────────────

def score_chunk(chunk: Chunk) -> float:
    """
    Compute and set chunk.quality_score. Returns the score.

    Weights from build guide:
      entity_density  × 0.4
      section_depth   × 0.2
      table_adjacency × 0.2
      length_band     × 0.2
    """
    score = (
        _entity_density(chunk.text)   * 0.4 +
        _section_depth(chunk.section) * 0.2 +
        _table_adjacency(chunk)       * 0.2 +
        _length_score(chunk.text)     * 0.2
    )
    chunk.quality_score = round(min(max(score, 0.0), 1.0), 4)
    return chunk.quality_score


def score_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Score all chunks in place. Returns the same list."""
    for c in chunks:
        score_chunk(c)
    if chunks:
        scores = [c.quality_score for c in chunks]
        avg = sum(scores) / len(scores)
        print(f"  [quality] {len(chunks)} chunks — "
              f"avg={avg:.3f}  min={min(scores):.3f}  max={max(scores):.3f}")
    return chunks


def filter_low_quality(chunks: list[Chunk], threshold: float = 0.05) -> list[Chunk]:
    """Drop chunks below threshold. Call after score_chunks."""
    before = len(chunks)
    kept = [c for c in chunks if c.quality_score >= threshold]
    removed = before - len(kept)
    if removed:
        print(f"  [quality] removed {removed} chunks below {threshold} → {len(kept)} kept")
    return kept
