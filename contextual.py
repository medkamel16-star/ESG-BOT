"""
esg_rag/contextual.py
---------------------
add_contextual_prefixes(parsed_doc, chunks)

For each chunk, asks Albert (cheap model) to write 1-2 sentences that situate
the chunk in its document. Sets chunk.context_prefix.

Uses asyncio.gather with concurrency=8 (5-10 per guide spec).
Results cached to data/parsed_cache/{doc_id}_ctx_{chunk_index}.txt
so re-indexing after a code change doesn't re-spend tokens.

One-time cost at ingest. Expected: ~120 tokens per chunk.
For 500 chunks: ~60k tokens total.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import esg_rag.albert as albert
from esg_rag.schemas import Chunk

CTX_CACHE_DIR = Path("data/parsed_cache/ctx")
CTX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

CONCURRENCY = 8   # asyncio semaphore limit

SYSTEM = (
    "You are a precise document analyst. "
    "Write exactly 1-2 sentences that situate the following chunk within its document. "
    "Include: company name, report year, report type, section name, and the kind of "
    "information (metric values, narrative, table data, commitment/target). "
    "Be factual. Do NOT paraphrase the chunk. Do NOT add opinions."
)


def _cache_key(chunk: Chunk) -> str:
    raw = f"{chunk.doc_id}:{chunk.chunk_index}:{chunk.text[:200]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load(key: str) -> str | None:
    p = CTX_CACHE_DIR / f"{key}.txt"
    return p.read_text(encoding="utf-8") if p.exists() else None


def _save(key: str, prefix: str) -> None:
    (CTX_CACHE_DIR / f"{key}.txt").write_text(prefix, encoding="utf-8")


def _user_msg(chunk: Chunk) -> str:
    return (
        f"Company: {chunk.company}\n"
        f"Report: {chunk.report_type} {chunk.year}\n"
        f"Section: {chunk.section or 'unknown'}\n"
        f"Page: {chunk.page}\n\n"
        f"Chunk:\n{chunk.text[:600]}\n\n"
        "Write the 1-2 sentence context prefix:"
    )


# ── Async worker ──────────────────────────────────────────────────────────────

async def _add_one(chunk: Chunk, sem: asyncio.Semaphore, force: bool) -> None:
    """Add context prefix to a single chunk, respecting the semaphore."""
    key = _cache_key(chunk)
    if not force:
        cached = _load(key)
        if cached is not None:
            chunk.context_prefix = cached
            return

    async with sem:
        # albert.chat_text is synchronous — run in thread pool so we don't block
        loop = asyncio.get_event_loop()
        prefix = await loop.run_in_executor(
            None,
            lambda: albert.chat_text(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": _user_msg(chunk)},
                ],
                model=albert.CHAT_MODEL,
                max_tokens=100,
                temperature=0.0,
                caller="contextual",
            ).strip()
        )
        _save(key, prefix)
        chunk.context_prefix = prefix


# ── Public API ────────────────────────────────────────────────────────────────

def add_contextual_prefixes(
    chunks: list[Chunk],
    *,
    enabled: bool = True,
    force: bool = False,
) -> list[Chunk]:
    """
    Add contextual prefixes to all non-figure chunks using asyncio.gather.
    Figures are skipped (their caption already provides context).

    Args:
        chunks:  list of chunks to enrich (modified in place)
        enabled: set to False to skip (reads from pipeline_config)
        force:   re-generate even if cached

    Returns:
        Same list (modified in place).
    """
    if not enabled:
        print("  [contextual] disabled — skipping")
        return chunks

    targets = [c for c in chunks if not c.is_figure]
    figures_skipped = len(chunks) - len(targets)
    print(f"  [contextual] enriching {len(targets)} chunks "
          f"({figures_skipped} figures skipped, concurrency={CONCURRENCY})…")

    async def _run_all():
        sem = asyncio.Semaphore(CONCURRENCY)
        tasks = [_add_one(c, sem, force) for c in targets]
        await asyncio.gather(*tasks)

    asyncio.run(_run_all())

    cached = sum(1 for c in targets if c.context_prefix and not force)
    new    = len(targets) - cached
    print(f"  [contextual] done — {cached} from cache, {new} new Albert calls")
    return chunks
