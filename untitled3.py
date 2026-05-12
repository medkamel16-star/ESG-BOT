import os
import hashlib
import io
import re
import json
import math
import sqlite3
from typing import Any

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image, ImageDraw


ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
ALBERT_EMBED_MODEL = "BAAI/bge-m3"

SQLITE_PATH = ".rag_store.sqlite"
SQLITE_TABLE = "chunks"
SQLITE_PAGES_TABLE = "pages"
UPLOADS_DIR = ".rag_uploads"

DEFAULT_CHUNK_SIZE = 1200  # chars
DEFAULT_CHUNK_OVERLAP = 200  # chars


def build_client(api_key: str) -> OpenAI:
    return OpenAI(base_url=ALBERT_BASE_URL, api_key=api_key)


@st.cache_data(show_spinner=False, ttl=60)
def list_text_generation_models(api_key: str) -> list[str]:
    client = build_client(api_key)
    models = client.models.list().data
    return sorted([m.id for m in models if getattr(m, "type", None) == "text-generation"])


def _normalize_ws(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_to_pages(file_bytes: bytes) -> list[dict[str, Any]]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages: list[dict[str, Any]] = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        txt = _normalize_ws(raw)
        if not txt:
            continue
        pages.append({"page": i + 1, "text": txt})
    return pages


def chunk_pages(
    pages: list[dict[str, Any]],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    chunks: list[dict[str, Any]] = []
    for p in pages:
        page_num = int(p["page"])
        text = str(p["text"])
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            raw_slice = text[start:end]
            chunk_text = raw_slice.strip()
            if chunk_text:
                # Store location so we can highlight later.
                left_trim = len(raw_slice) - len(raw_slice.lstrip())
                right_trim = len(raw_slice) - len(raw_slice.rstrip())
                loc_start = start + left_trim
                loc_end = end - right_trim
                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            "page": page_num,
                            "start_char": int(loc_start),
                            "end_char": int(loc_end),
                        },
                    }
                )
            if end >= len(text):
                break
            start = max(0, end - chunk_overlap)
    return chunks

def _sqlite_connect() -> sqlite3.Connection:
    con = sqlite3.connect(SQLITE_PATH)
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SQLITE_TABLE} (
            id TEXT PRIMARY KEY,
            document TEXT NOT NULL,
            metadata TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SQLITE_PAGES_TABLE} (
            file_hash TEXT NOT NULL,
            source TEXT NOT NULL,
            page INTEGER NOT NULL,
            text TEXT NOT NULL,
            PRIMARY KEY (file_hash, page)
        )
        """
    )
    return con


def get_page_text(*, file_hash: str, page: int) -> str | None:
    con = _sqlite_connect()
    cur = con.execute(
        f"SELECT text FROM {SQLITE_PAGES_TABLE} WHERE file_hash = ? AND page = ?",
        (file_hash, int(page)),
    )
    row = cur.fetchone()
    con.close()
    return row[0] if row else None


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom == 0.0:
        return 1.0
    # distance = 1 - cosine_similarity
    return 1.0 - (dot / denom)


def embed_texts(api_key: str, texts: list[str]) -> list[list[float]]:
    client = build_client(api_key)
    if not texts:
        return []

    # Albert embeddings endpoint enforces a max batch size (observed: 64).
    max_batch = 64
    out: list[list[float]] = []
    for i in range(0, len(texts), max_batch):
        batch = texts[i : i + max_batch]
        resp = client.embeddings.create(model=ALBERT_EMBED_MODEL, input=batch, encoding_format="float")
        data = sorted(resp.data, key=lambda d: d.index)
        out.extend([d.embedding for d in data])
    return out


def ingest_pdf(
    *,
    api_key: str,
    filename: str,
    file_bytes: bytes,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, Any]:
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    pdf_path = os.path.join(UPLOADS_DIR, f"{file_hash}.pdf")
    if not os.path.exists(pdf_path):
        with open(pdf_path, "wb") as f:
            f.write(file_bytes)

    pages = pdf_to_pages(file_bytes)
    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not chunks:
        return {
            "file_hash": file_hash,
            "pages": len(pages),
            "chunks": 0,
            "upserted": 0,
            "note": "No extractable text found in PDF (might be scanned).",
        }

    # Persist extracted page text for later highlighting (works for both backends).
    con_pages = _sqlite_connect()
    with con_pages:
        con_pages.executemany(
            f"INSERT OR REPLACE INTO {SQLITE_PAGES_TABLE}(file_hash, source, page, text) VALUES (?, ?, ?, ?)",
            [(file_hash, filename, int(p["page"]), str(p["text"])) for p in pages],
        )
    con_pages.close()

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(api_key, texts)

    ids: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for idx, c in enumerate(chunks):
        ids.append(f"{file_hash}:{idx}")
        metadatas.append(
            {
                "source": filename,
                "file_hash": file_hash,
                "chunk": idx,
                "page": int(c["metadata"]["page"]),
                "start_char": int(c["metadata"]["start_char"]),
                "end_char": int(c["metadata"]["end_char"]),
            }
        )


    con = _sqlite_connect()
    with con:
        con.executemany(
            f"INSERT OR REPLACE INTO {SQLITE_TABLE}(id, document, metadata, embedding) VALUES (?, ?, ?, ?)",
            [
                (i, doc, json.dumps(meta, ensure_ascii=False), json.dumps(emb, ensure_ascii=False))
                for i, doc, meta, emb in zip(ids, texts, metadatas, embeddings)
            ],
        )
    con.close()

    return {
        "file_hash": file_hash,
        "pdf_path": pdf_path,
        "pages": len(pages),
        "chunks": len(chunks),
        "upserted": len(ids),
        "note": None,
    }


def rag_retrieve(api_key: str, query: str, *, k: int = 5) -> list[dict[str, Any]]:
    q_emb = embed_texts(api_key, [query])[0]
    con = _sqlite_connect()
    cur = con.execute(f"SELECT id, document, metadata, embedding FROM {SQLITE_TABLE}")
    rows = cur.fetchall()
    con.close()

    scored: list[dict[str, Any]] = []
    for _id, doc, meta_s, emb_s in rows:
        try:
            emb = json.loads(emb_s)
            meta = json.loads(meta_s)
        except Exception:
            continue
        dist = _cosine_distance(q_emb, emb)
        scored.append({"text": doc, "metadata": meta, "distance": dist})

    scored.sort(key=lambda x: x["distance"])
    return scored[:k]


def format_context(hits: list[dict[str, Any]], *, char_limit: int = 8000) -> str:
    parts: list[str] = []
    total = 0
    for h in hits:
        meta = h.get("metadata") or {}
        header = f"[source={meta.get('source')} page={meta.get('page')} chunk={meta.get('chunk')}]"
        body = (h.get("text") or "").strip()
        if not body:
            continue
        snippet = f"{header}\n{body}"
        if total + len(snippet) + 5 > char_limit:
            break
        parts.append(snippet)
        total += len(snippet) + 5
    return "\n\n---\n\n".join(parts)


def render_highlighted_page(*, page_text: str, start_char: int, end_char: int) -> None:
    start_char = max(0, min(int(start_char), len(page_text)))
    end_char = max(start_char, min(int(end_char), len(page_text)))
    before = page_text[:start_char]
    mid = page_text[start_char:end_char]
    after = page_text[end_char:]

    html = (
        "<div style='white-space: pre-wrap; font-family: ui-sans-serif, system-ui;'>"
        f"{before}"
        "<mark style='background: #fff3a3; padding: 0.1em 0.15em; border-radius: 0.15em;'>"
        f"{mid}"
        "</mark>"
        f"{after}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _norm_token(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", (s or "").lower())).strip()


def _words_for_page(pdf_path: str, page_number: int) -> list[dict[str, Any]]:
    # pdfplumber page_number is 1-based for our UI; internally it's 0-based.
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
        # keys: text, x0, x1, top, bottom, ... (in PDF points)
        return words


def _find_word_span(words: list[dict[str, Any]], chunk_text: str) -> tuple[int, int] | None:
    # Heuristic: find a contiguous span whose normalized tokens match chunk tokens.
    wtoks = [_norm_token(w.get("text", "")) for w in words]
    wtoks = [t for t in wtoks if t]
    ctoks = [_norm_token(t) for t in chunk_text.split()]
    ctoks = [t for t in ctoks if t]
    if not ctoks or not wtoks:
        return None

    # Use an anchor of first few tokens to reduce work
    anchor = ctoks[: min(8, len(ctoks))]
    for i in range(0, max(1, len(wtoks) - len(anchor) + 1)):
        if wtoks[i : i + len(anchor)] != anchor:
            continue
        # Extend match
        j = i + len(anchor)
        k = len(anchor)
        while j < len(wtoks) and k < len(ctoks) and wtoks[j] == ctoks[k]:
            j += 1
            k += 1
        if k >= min(len(ctoks), len(anchor) + 10):  # matched enough to be confident
            # Prefer full match if possible
            if k == len(ctoks):
                return (i, j)
            # Otherwise accept partial highlight
            return (i, j)
    return None


def render_pdf_page_with_highlight(*, pdf_path: str, page: int, chunk_text: str) -> tuple[bool, bool]:
    """
    Returns (rendered, highlighted).
    Requires Poppler for pdf2image on Windows.
    """
    if not (pdf_path and os.path.exists(pdf_path) and page and page > 0):
        return (False, False)

    try:
        images = convert_from_path(pdf_path, first_page=page, last_page=page, fmt="png")
        if not images:
            return (False, False)
        img: Image.Image = images[0]
        words = _words_for_page(pdf_path, page)
        if not words:
            st.image(img, caption=f"PDF page {page}", use_container_width=True)
            return (True, False)

        span = _find_word_span(words, chunk_text)
        if not span:
            st.image(img, caption=f"PDF page {page} (no exact match found)", use_container_width=True)
            return (True, False)

        # Map word boxes to image pixels. pdfplumber uses PDF points with origin at top-left for "top/bottom".
        # We need page width/height in points to scale to pixels.
        with pdfplumber.open(pdf_path) as pdf:
            p = pdf.pages[page - 1]
            page_w = float(p.width)
            page_h = float(p.height)

        sx = img.width / page_w
        sy = img.height / page_h

        draw = ImageDraw.Draw(img, "RGBA")
        start_i, end_i = span

        # Rebuild wtoks with original indices (words may include punctuation tokens; we will highlight by best-effort matching)
        # Use a simpler mapping: highlight words whose normalized token is non-empty and within the span in the filtered list.
        filtered_indices: list[int] = []
        for idx, w in enumerate(words):
            if _norm_token(w.get("text", "")):
                filtered_indices.append(idx)

        if end_i > len(filtered_indices):
            end_i = len(filtered_indices)

        for fi in range(start_i, end_i):
            wi = filtered_indices[fi]
            w = words[wi]
            x0 = float(w["x0"]) * sx
            x1 = float(w["x1"]) * sx
            top = float(w["top"]) * sy
            bottom = float(w["bottom"]) * sy
            draw.rectangle([x0, top, x1, bottom], fill=(255, 241, 118, 120), outline=(255, 214, 10, 180))

        st.image(img, caption=f"PDF page {page} (highlighted)", use_container_width=True)
        return (True, True)
    except Exception:
        return (False, False)


def main() -> None:
    st.set_page_config(page_title="Albert API Chat", page_icon="💬", layout="centered")
    st.title("Albert API Chat")

    with st.sidebar:
        st.header("Configuration")
        api_key = "sk-eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoxMTkxNCwidG9rZW5faWQiOjI2ODgzLCJleHBpcmVzIjoxNzgzMzc1MjAwfQ.Fba-CrQfSykbKe5SSiMhLlfb4Mw1CIAsGbThKW8-vxQ"
            

        if not api_key:
            st.info("Enter your API key to load models and start chatting.")
            st.stop()

        try:
            model_ids = list_text_generation_models(api_key)
        except Exception as e:
            st.error(f"Could not list models: {e}")
            st.stop()

        if not model_ids:
            st.error("No `text-generation` model found for this API key.")
            st.stop()

        selected_model = st.selectbox("Model", options=model_ids, index=0)

        system_prompt = st.text_area(
            "System prompt",
            value="Tu réponds en français, de façon concise.",
            height=100,
        )

        use_rag = st.toggle("Use RAG", value=True)
        with st.expander("RAG settings", expanded=False):
            chunk_size = st.number_input(
                "Chunk size (chars)", min_value=300, max_value=4000, value=DEFAULT_CHUNK_SIZE, step=100
            )
            chunk_overlap = st.number_input(
                "Chunk overlap (chars)",
                min_value=0,
                max_value=int(chunk_size) - 1,
                value=min(DEFAULT_CHUNK_OVERLAP, int(chunk_size) - 1),
                step=50,
            )
            top_k = st.number_input("Top-K chunks", min_value=1, max_value=20, value=5, step=1)
            backend = "SQLite (fallback)"
            st.caption(f"Vector store backend: **{backend}**")

        if st.button("Clear chat", type="secondary", use_container_width=True):
            st.session_state.pop("messages", None)
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history (excluding system message which is controlled in sidebar)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Attachment uploader (paperclip) placed just above the chat bar.
    attach_col, upload_col, status_col = st.columns([0.6, 5, 6])
    with attach_col:
        st.markdown("📎")
    with upload_col:
        uploaded = st.file_uploader(
            "Attach a PDF",
            type=["pdf"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key="rag_pdf_uploader",
        )
    with status_col:
        last_ingest = st.session_state.get("last_ingest")
        if last_ingest:
            note = f" — {last_ingest['note']}" if last_ingest.get("note") else ""
            st.caption(
                f"Indexed `{last_ingest['upserted']}` chunks from `{last_ingest.get('source')}` (pages={last_ingest['pages']}).{note}"
            )
        else:
            st.caption("Upload a PDF to index it locally.")

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
        already = st.session_state.get("ingested_hashes", set())
        if file_hash not in already:
            with st.spinner("Indexing PDF (extract → chunk → embed → store)…"):
                info = ingest_pdf(
                    api_key=api_key,
                    filename=uploaded.name,
                    file_bytes=file_bytes,
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                )
            info["source"] = uploaded.name
            st.session_state["last_ingest"] = info
            st.session_state["ingested_hashes"] = set(already) | {file_hash}
            if info.get("upserted", 0) > 0:
                st.success(f"Indexed {info['upserted']} chunks from {uploaded.name}")
            else:
                st.warning(info.get("note") or "Nothing indexed.")

    user_text = st.chat_input("Write a message…")
    if not user_text:
        return

    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    client = build_client(api_key)

    rag_context = ""
    if use_rag:
        try:
            hits = rag_retrieve(api_key, user_text, k=int(top_k))
            rag_context = format_context(hits)
            st.session_state["last_rag_hits"] = hits
        except Exception:
            rag_context = ""
            st.session_state["last_rag_hits"] = []

    sys = system_prompt
    if rag_context:
        sys = (
            f"{system_prompt}\n\n"
            "Tu peux utiliser le contexte ci-dessous si pertinent. Si le contexte ne contient pas l'information, "
            "dis-le clairement.\n\n"
            f"Contexte:\n{rag_context}"
        )

    messages_for_api = [{"role": "system", "content": sys}] + list(st.session_state.messages)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        acc = ""
        try:
            stream = client.chat.completions.create(
                model=selected_model,
                messages=messages_for_api,
                stream=True,
            )
            for event in stream:
                choices = getattr(event, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                token = getattr(delta, "content", None) or ""
                if token:
                    acc += token
                    placeholder.markdown(acc)
        except Exception as e:
            st.error(f"Request failed: {e}")
            return

    st.session_state.messages.append({"role": "assistant", "content": acc})

    # Clickable citations: after each answer, show retrieved sources and allow highlighting.
    hits = st.session_state.get("last_rag_hits") or []
    if hits:
        # Default highlight: always show the top retrieved chunk (rank #1).
        top_hit = hits[0]
        meta = (top_hit.get("metadata") or {}).copy()
        with st.expander("Top source (auto-highlighted)", expanded=True):
            st.markdown(
                f"**Top chunk** — `{meta.get('source')}`, page **{meta.get('page')}** (chunk {meta.get('chunk')})"
            )

            pdf_path = os.path.join(UPLOADS_DIR, f"{str(meta.get('file_hash'))}.pdf")
            rendered, highlighted = render_pdf_page_with_highlight(
                pdf_path=pdf_path,
                page=int(meta.get("page") or 0),
                chunk_text=str(top_hit.get("text") or ""),
            )
            if rendered and not highlighted:
                st.caption("Rendered the cited page, but couldn't match the chunk text precisely to highlight it.")
            if not rendered:
                st.caption(
                    "PDF highlight view unavailable (likely missing Poppler). Falling back to extracted text highlight."
                )
                page_text = get_page_text(file_hash=str(meta.get("file_hash")), page=int(meta.get("page") or 0))
                if not page_text:
                    st.warning("Could not load stored page text for highlighting.")
                else:
                    render_highlighted_page(
                        page_text=page_text,
                        start_char=int(meta.get("start_char") or 0),
                        end_char=int(meta.get("end_char") or 0),
                    )

        # Still show other sources as a non-interactive list (optional context).
        if len(hits) > 1:
            with st.expander("Other retrieved sources", expanded=False):
                for i, h in enumerate(hits[1:], start=2):
                    m = h.get("metadata") or {}
                    st.markdown(f"- **[{i}]** `{m.get('source')}` — page {m.get('page')} (chunk {m.get('chunk')})")


if __name__ == "__main__":
    main()