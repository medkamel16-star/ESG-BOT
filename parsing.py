"""
esg_rag/parsing.py
------------------
PDF → list[Chunk].  Called by ingest.py.

Primary: Docling (reads columns, tables, figures, heading hierarchy correctly).
Fallback: pdfplumber (layout-aware text extraction, basic table support).

Produces Chunk objects defined in schemas.py.
Chunking (splitting long text blocks) happens in chunking.py.
This module only extracts structural elements from the PDF.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from esg_rag.schemas import Chunk


# ── Helpers ───────────────────────────────────────────────────────────────────

def doc_id_from_path(pdf_path: Path) -> str:
    """SHA256[:12] of the PDF bytes — stable identity for caching."""
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:12]


def detect_report_type(filename: str) -> str:
    name = filename.lower()
    if any(k in name for k in ("csrd", "esrs")):
        return "csrd"
    if any(k in name for k in ("sustain", "esg", "climate", "environmental", "responsibility")):
        return "sustainability"
    if any(k in name for k in ("annual", "rapport", "yearly")):
        return "annual"
    return "unknown"


# ── Docling parser ────────────────────────────────────────────────────────────

def _parse_docling(pdf_path: Path, doc_id: str, company: str, year: str) -> list[Chunk]:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = True

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    report_type = detect_report_type(pdf_path.name)
    chunks: list[Chunk] = []
    idx = 0
    section_stack: list[str] = []

    for element, _level in doc.iterate_items():
        el_type = type(element).__name__

        # ── Track headings
        if el_type in ("SectionHeaderItem", "HeadingItem"):
            txt = element.text.strip()
            lvl = getattr(element, "level", 1)
            section_stack = section_stack[: lvl - 1] + [txt]
            continue

        section = " > ".join(section_stack)
        prov = getattr(element, "prov", None)
        page = prov[0].get("page_no", 0) if prov else 0

        # ── Tables
        if el_type == "TableItem":
            try:
                md = element.export_to_markdown()
                df = element.export_to_dataframe()
                tdata = [{"headers": list(df.columns), "rows": df.values.tolist()}] if df is not None and not df.empty else []
            except Exception:
                md = str(element)
                tdata = []
            chunks.append(Chunk(
                doc_id=doc_id, chunk_index=idx, company=company, year=year,
                report_type=report_type, source_pdf=pdf_path.name,
                text=md, page=page, section=section,
                has_table=True, table_data=tdata,
            ))
            idx += 1
            continue

        # ── Figures
        if el_type in ("FigureItem", "PictureItem"):
            caption = ""
            if hasattr(element, "captions") and element.captions:
                cap = element.captions[0]
                caption = cap.text if hasattr(cap, "text") else str(cap)
            if not caption:
                continue

            # Try to export figure image
            image_path = ""
            try:
                fig_dir = Path("data/figures") / doc_id
                fig_dir.mkdir(parents=True, exist_ok=True)
                fig_file = fig_dir / f"fig_{idx}.png"
                element.image.pil_image.save(str(fig_file))
                image_path = str(fig_file)
            except Exception:
                pass

            chunks.append(Chunk(
                doc_id=doc_id, chunk_index=idx, company=company, year=year,
                report_type=report_type, source_pdf=pdf_path.name,
                text=f"[Figure] {caption}", page=page, section=section,
                is_figure=True, figure_caption=caption, image_path=image_path,
            ))
            idx += 1
            continue

        # ── Text
        text = getattr(element, "text", "").strip()
        if len(text) < 30:
            continue

        chunks.append(Chunk(
            doc_id=doc_id, chunk_index=idx, company=company, year=year,
            report_type=report_type, source_pdf=pdf_path.name,
            text=text, page=page, section=section,
        ))
        idx += 1

    return chunks


# ── pdfplumber fallback ───────────────────────────────────────────────────────

def _parse_pdfplumber(pdf_path: Path, doc_id: str, company: str, year: str) -> list[Chunk]:
    import pdfplumber

    report_type = detect_report_type(pdf_path.name)
    chunks: list[Chunk] = []
    idx = 0

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # ── Tables on this page
            for table in page.extract_tables() or []:
                if not table:
                    continue
                headers = [str(c) for c in (table[0] or [])]
                rows = [[str(cell) for cell in row] for row in table[1:] if row]
                # Simple markdown
                header_row = " | ".join(headers)
                separator = " | ".join(["---"] * len(headers))
                body = "\n".join(" | ".join(row) for row in rows[:20])
                md = f"| {header_row} |\n| {separator} |\n" + "\n".join(f"| {r} |" for r in body.splitlines())
                chunks.append(Chunk(
                    doc_id=doc_id, chunk_index=idx, company=company, year=year,
                    report_type=report_type, source_pdf=pdf_path.name,
                    text=md, page=page_num, section="",
                    has_table=True, table_data=[{"headers": headers, "rows": rows}],
                ))
                idx += 1

            # ── Text
            text = page.extract_text() or ""
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 50:
                chunks.append(Chunk(
                    doc_id=doc_id, chunk_index=idx, company=company, year=year,
                    report_type=report_type, source_pdf=pdf_path.name,
                    text=text, page=page_num, section="",
                ))
                idx += 1

    return chunks


# ── Public entry point ────────────────────────────────────────────────────────

def parse_pdf(pdf_path: Path | str, company: str, year: str | int) -> list[Chunk]:
    """
    Parse a PDF → list[Chunk]. Tries Docling first, falls back to pdfplumber.
    Does NOT chunk (split long text) — that is done by chunking.py.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc_id = doc_id_from_path(pdf_path)
    year   = str(year)

    try:
        chunks = _parse_docling(pdf_path, doc_id, company, year)
        if chunks:
            print(f"  [parsing] docling: {len(chunks)} elements from {pdf_path.name}")
            return chunks
        print(f"  [parsing] docling returned 0 chunks — using pdfplumber fallback")
    except Exception as e:
        print(f"  [parsing] docling error ({type(e).__name__}: {e}) — using pdfplumber fallback")

    chunks = _parse_pdfplumber(pdf_path, doc_id, company, year)
    print(f"  [parsing] pdfplumber: {len(chunks)} elements from {pdf_path.name}")
    return chunks
