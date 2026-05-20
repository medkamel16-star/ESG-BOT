"""
esg_rag/chunking.py
-------------------
Splits raw Chunk objects (from parsing.py) into indexable pieces.

Strategy: sentence-boundary sliding window.
  - Target size: 800–1200 chars (configurable)
  - Overlap: 150 chars (one or two sentences carried forward)
  - Tables and figures: NEVER split — kept as single chunks regardless of size
  - Very short chunks (<80 chars): kept as-is (will be filtered by quality scorer)

Input:  list[Chunk] from parsing.py (one chunk = one structural element)
Output: list[Chunk] with long text blocks split into smaller pieces,
        each with its own chunk_index.
"""

from __future__ import annotations

import re
from esg_rag.schemas import Chunk


# ── Text splitter ─────────────────────────────────────────────────────────────

def _sentence_split(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """
    Split text on sentence boundaries, respecting max_chars.
    Carries overlap chars forward to preserve context across chunks.
    """
    # Split on sentence boundaries: ". ", "! ", "? ", ".\n"
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    pieces: list[str] = []
    current = ""

    for sent in sentences:
        if not sent.strip():
            continue
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip() if current else sent
        else:
            if current:
                pieces.append(current)
            # Carry overlap from end of previous piece
            tail = current[-overlap:] if len(current) > overlap else current
            # Find a word boundary in the tail
            boundary = tail.rfind(" ")
            tail = tail[boundary + 1:] if boundary > 0 else tail
            current = (tail + " " + sent).strip() if tail else sent

    if current:
        pieces.append(current)

    return pieces if pieces else [text[:max_chars]]


# ── Main function ─────────────────────────────────────────────────────────────

def chunk_elements(
    elements: list[Chunk],
    max_chars: int = 1200,
    overlap: int = 150,
) -> list[Chunk]:
    """
    Split long text Chunk objects into indexable pieces.
    Tables and figures pass through unchanged.
    Re-assigns chunk_index sequentially across the document.

    Args:
        elements:  list[Chunk] from parsing.py
        max_chars: target maximum characters per chunk
        overlap:   overlap between consecutive chunks (chars)

    Returns:
        list[Chunk] ready for quality scoring and embedding.
    """
    output: list[Chunk] = []
    idx = 0

    for el in elements:
        # ── Tables: never split
        if el.has_table or el.is_figure:
            el.chunk_index = idx
            output.append(el)
            idx += 1
            continue

        # ── Short text: pass through
        if len(el.text) <= max_chars:
            el.chunk_index = idx
            output.append(el)
            idx += 1
            continue

        # ── Long text: split into pieces
        pieces = _sentence_split(el.text, max_chars=max_chars, overlap=overlap)
        for piece in pieces:
            if len(piece.strip()) < 20:
                continue
            output.append(Chunk(
                doc_id=el.doc_id,
                chunk_index=idx,
                company=el.company,
                year=el.year,
                report_type=el.report_type,
                source_pdf=el.source_pdf,
                text=piece.strip(),
                page=el.page,
                section=el.section,
                has_table=False,
                table_data=[],
                is_figure=False,
            ))
            idx += 1

    print(f"  [chunking] {len(elements)} elements → {len(output)} chunks "
          f"(max_chars={max_chars}, overlap={overlap})")
    return output
