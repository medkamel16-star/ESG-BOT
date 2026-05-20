"""
esg_rag/ingest.py
-----------------
2.2  ingest_pdf(path, company, year, report_type) -> list[Chunk]
     Parse a single PDF. Cache parsed JSON at data/parsed_cache/{doc_id}.json.
     Skip parse if cached.

2.7  full_ingest(pdf_paths, config) -> summary dict
     parse → chunk → quality_score → vlm_figures → contextual_prefix → embed → upsert
     Shows tqdm progress bars.

CLI:
     python -m esg_rag.ingest data/pdfs/*.pdf
     python -m esg_rag.ingest --manifest ingest_manifest.yaml
     python -m esg_rag.ingest --pdf path/to/report.pdf --company Engie --year 2024
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

import esg_rag.albert as albert
from esg_rag.schemas import Chunk
from esg_rag.parsing import parse_pdf, doc_id_from_path, detect_report_type
from esg_rag.chunking import chunk_elements
from esg_rag.chunk_quality import score_chunks, filter_low_quality
from esg_rag.contextual import add_contextual_prefixes
from esg_rag.store import QdrantStore

CACHE_DIR = Path("data/parsed_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Chunk serialisation ───────────────────────────────────────────────────────

def _chunk_to_dict(c: Chunk) -> dict[str, Any]:
    d = asdict(c)
    d.pop("embedding", None)   # don't cache embeddings — re-compute always
    return d


def _dict_to_chunk(d: dict[str, Any]) -> Chunk:
    d.setdefault("embedding", [])
    d.setdefault("vlm_description", "")
    d.setdefault("image_path", "")
    return Chunk(**{k: v for k, v in d.items() if k in Chunk.__dataclass_fields__})


# ── 2.2 — ingest_pdf ─────────────────────────────────────────────────────────

def ingest_pdf(
    path: Path | str,
    company: str,
    year: str | int,
    report_type: str = "",
    *,
    force_reparse: bool = False,
) -> list[Chunk]:
    """
    Parse a single PDF → list[Chunk] (elements, not yet chunked/scored).
    Caches parsed JSON at data/parsed_cache/{doc_id}.json.
    """
    pdf_path = Path(path)
    doc_id = doc_id_from_path(pdf_path)
    cache_file = CACHE_DIR / f"{doc_id}.json"

    if cache_file.exists() and not force_reparse:
        print(f"[ingest] cache hit {pdf_path.name} ({doc_id})")
        raw = json.loads(cache_file.read_text())
        return [_dict_to_chunk(d) for d in raw]

    print(f"[ingest] parsing {pdf_path.name} …")
    if not report_type:
        report_type = detect_report_type(pdf_path.name)

    chunks = parse_pdf(pdf_path, company=company, year=str(year))

    # Overwrite report_type and company in case parse_pdf guessed differently
    for c in chunks:
        c.company = company
        c.year = str(year)
        if report_type:
            c.report_type = report_type

    cache_file.write_text(json.dumps([_chunk_to_dict(c) for c in chunks], ensure_ascii=False))
    print(f"[ingest] cached {len(chunks)} elements → {cache_file.name}")
    return chunks


# ── 2.7 — full_ingest ────────────────────────────────────────────────────────

def full_ingest(
    pdf_specs: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    *,
    store: QdrantStore | None = None,
    force_reparse: bool = False,
    force_reembed: bool = False,
) -> dict[str, Any]:
    """
    Full pipeline for a list of PDFs:
      parse → chunk → quality_score → (vlm_figures) → contextual_prefix → embed → upsert

    pdf_specs: [{"path": ..., "company": ..., "year": ..., "report_type": ...}, ...]
    config:    dict from pipeline_config.yaml (or None for defaults)

    Returns a summary dict with per-doc stats and total elapsed time.
    """
    import time
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x   # noqa: E731

    cfg = config or {}
    store = store or QdrantStore()
    t_start = time.perf_counter()

    # Config flags
    do_contextual   = cfg.get("ingest", {}).get("contextual_prefixes", True)
    do_vlm          = cfg.get("ingest", {}).get("vlm_figures", False)
    quality_thresh  = cfg.get("ingest", {}).get("quality_filter_threshold", 0.05)
    chunk_max_chars = cfg.get("ingest", {}).get("chunk_max_chars", 1200)
    chunk_overlap   = cfg.get("ingest", {}).get("chunk_overlap", 150)
    embed_model     = cfg.get("ingest", {}).get("embed_model", albert.EMBED_MODEL)

    summary: dict[str, Any] = {"docs": [], "errors": [], "total_chunks": 0}
    all_chunks: list[Chunk] = []

    already_indexed = {d["doc_id"] for d in store.list_docs()}

    # ── Step 1: Parse ─────────────────────────────────────────────────────────
    print("\n══ 1/5  Parsing PDFs ══")
    for spec in tqdm(pdf_specs, desc="parsing"):
        try:
            elements = ingest_pdf(
                spec["path"], spec["company"], spec["year"],
                spec.get("report_type", ""),
                force_reparse=force_reparse,
            )
            chunks = chunk_elements(elements, max_chars=chunk_max_chars, overlap=chunk_overlap)
            all_chunks.extend(chunks)
            summary["docs"].append({
                "path": str(spec["path"]),
                "company": spec["company"],
                "year": str(spec["year"]),
                "elements": len(elements),
                "chunks": len(chunks),
            })
        except Exception as e:
            msg = f"FAILED {spec['path']}: {e}"
            print(f"\n  [ingest] {msg}")
            summary["errors"].append(msg)

    print(f"  → {len(all_chunks)} chunks from {len(summary['docs'])} docs")

    # ── Step 2: Quality score + filter ────────────────────────────────────────
    print("\n══ 2/5  Scoring chunk quality ══")
    score_chunks(all_chunks)
    all_chunks = filter_low_quality(all_chunks, threshold=quality_thresh)

    # ── Step 3: VLM figure descriptions (GPU, optional) ───────────────────────
    if do_vlm:
        print("\n══ 3/5  VLM figure descriptions ══")
        try:
            from esg_rag.vlm_figures import describe_figures
            describe_figures(all_chunks)
        except ImportError:
            print("  [ingest] vlm_figures.py not found — skipping")
    else:
        print("\n══ 3/5  VLM figures (disabled) ══")

    # ── Step 4: Contextual prefixes ───────────────────────────────────────────
    print("\n══ 4/5  Contextual prefixes ══")
    add_contextual_prefixes(all_chunks, enabled=do_contextual)

    # ── Step 5: Embed + upsert ────────────────────────────────────────────────
    print("\n══ 5/5  Embedding and upserting ══")
    by_doc: dict[str, list[Chunk]] = {}
    for c in all_chunks:
        by_doc.setdefault(c.doc_id, []).append(c)

    for doc_id, doc_chunks in tqdm(by_doc.items(), desc="upserting"):
        if doc_id in already_indexed and not force_reembed:
            print(f"  [ingest] {doc_id} already indexed — skipping "
                  f"(use --force-reembed to override)")
            continue

        texts = [c.full_text for c in doc_chunks]
        embeddings = albert.embed_texts(texts, model=embed_model, caller="ingest_embed")
        for chunk, emb in zip(doc_chunks, embeddings):
            chunk.embedding = emb

        store.upsert_chunks(doc_chunks)

    elapsed = time.perf_counter() - t_start
    summary["total_chunks"] = len(all_chunks)
    summary["elapsed_s"] = round(elapsed, 1)

    cost = albert.cost_summary()
    print(f"\n✓ Ingest complete in {elapsed:.1f}s")
    print(f"  Total Albert tokens in: {cost.get('total_tokens_in', 0)}")
    print(f"  Indexed docs: {len(store.list_docs())}")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESG ingest pipeline")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", help="Single PDF path")
    src.add_argument("--manifest", help="YAML manifest of PDFs")
    src.add_argument("pdfs", nargs="*", help="Glob of PDF paths (e.g. data/pdfs/*.pdf)")

    parser.add_argument("--company", help="Company name (for --pdf)")
    parser.add_argument("--year",    help="Report year (for --pdf)")
    parser.add_argument("--report-type", default="", help="Report type override")
    parser.add_argument("--config",  default="pipeline_config.yaml")
    parser.add_argument("--force-reparse",  action="store_true")
    parser.add_argument("--force-reembed",  action="store_true")
    args = parser.parse_args()

    # Load config
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

    if args.pdf:
        if not args.company or not args.year:
            parser.error("--pdf requires --company and --year")
        specs = [{"path": args.pdf, "company": args.company,
                  "year": args.year, "report_type": args.report_type}]

    elif args.manifest:
        with open(args.manifest) as f:
            manifest = yaml.safe_load(f)
        specs = [
            {"path": Path(m["path"]), "company": m["company"],
             "year": str(m["year"]), "report_type": m.get("report_type", "")}
            for m in manifest["pdfs"]
        ]

    else:
        if not args.pdfs:
            parser.error("Provide PDF paths or use --pdf / --manifest")
        specs = []
        for p in args.pdfs:
            pdf = Path(p)
            import re as _re
            year_m = _re.search(r'(20\d{2})', pdf.stem)
            specs.append({
                "path": pdf,
                "company": pdf.stem.split("_")[0].capitalize(),
                "year": year_m.group(1) if year_m else "unknown",
                "report_type": "",
            })
        print(f"[ingest] auto-detected {len(specs)} PDFs — edit ingest_manifest.yaml "
              f"for precise company/year metadata")

    summary = full_ingest(
        specs, cfg,
        force_reparse=args.force_reparse,
        force_reembed=args.force_reembed,
    )
    print(f"\nSummary: {json.dumps(summary, indent=2, default=str)}")
