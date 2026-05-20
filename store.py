"""
esg_rag/store.py
----------------
class QdrantStore — Qdrant in embedded (local) mode. One collection per document.

Methods (per build guide spec 2.6):
  ensure_collection(doc_id)           — create collection if needed
  upsert_chunks(chunks)               — embed-and-store or store pre-embedded
  delete_doc(doc_id)                  — drop a single document's collection
  list_docs() -> [{doc_id, company, year, n_chunks}]
  payload_filters(dict) -> Filter     — build a Qdrant Filter from a plain dict

Named vectors: {"dense": cosine, "bm25": sparse(FastEmbed BM25)}
Storage: data/.qdrant/  (embedded, no Docker needed)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue
)

from esg_rag.schemas import Chunk, SearchHit

QDRANT_PATH = Path("data/.qdrant")
QDRANT_PATH.mkdir(parents=True, exist_ok=True)

DENSE_VEC  = "dense"
SPARSE_VEC = "bm25"

UPSERT_BATCH = 64


# ── BM25 encoder (local, no API call) ────────────────────────────────────────

_BM25_ENC = None

def _bm25_encoder():
    global _BM25_ENC
    if _BM25_ENC is None:
        try:
            from fastembed import SparseTextEmbedding
            _BM25_ENC = SparseTextEmbedding(model_name="Qdrant/bm25")
        except Exception as e:
            print(f"  [store] BM25 encoder unavailable ({e}) — sparse search disabled")
            _BM25_ENC = False
    return _BM25_ENC if _BM25_ENC is not False else None


def _encode_bm25(texts: list[str]) -> list[SparseVector | None]:
    enc = _bm25_encoder()
    if enc is None:
        return [None] * len(texts)
    results = list(enc.embed(texts))
    out = []
    for r in results:
        idx = r.indices.tolist() if hasattr(r.indices, "tolist") else list(r.indices)
        val = r.values.tolist()  if hasattr(r.values,  "tolist") else list(r.values)
        out.append(SparseVector(indices=idx, values=val))
    return out


# ── QdrantStore ───────────────────────────────────────────────────────────────

class QdrantStore:

    def __init__(self, path: str | Path = QDRANT_PATH):
        self._client = QdrantClient(path=str(path))

    def _col(self, doc_id: str) -> str:
        return f"doc_{doc_id}"

    # ── ensure_collection ─────────────────────────────────────────────────────

    def ensure_collection(self, doc_id: str, dim: int) -> None:
        """Create the collection for doc_id if it doesn't already exist."""
        name = self._col(doc_id)
        existing = {c.name for c in self._client.get_collections().collections}
        if name in existing:
            return

        has_bm25 = _bm25_encoder() is not None
        sparse_cfg = (
            {SPARSE_VEC: SparseVectorParams(index=SparseIndexParams(on_disk=False))}
            if has_bm25 else {}
        )

        self._client.create_collection(
            collection_name=name,
            vectors_config={DENSE_VEC: VectorParams(size=dim, distance=Distance.COSINE)},
            sparse_vectors_config=sparse_cfg,
        )
        print(f"  [store] created '{name}' (dim={dim}, bm25={'yes' if has_bm25 else 'no'})")

    # ── upsert_chunks ─────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """
        Store chunks with pre-computed embeddings (chunk.embedding must be set).
        BM25 sparse vectors are computed here locally.
        """
        if not chunks:
            return

        dim = len(chunks[0].embedding)
        by_doc: dict[str, list[Chunk]] = {}
        for c in chunks:
            by_doc.setdefault(c.doc_id, []).append(c)

        for doc_id, doc_chunks in by_doc.items():
            self.ensure_collection(doc_id, dim)
            name = self._col(doc_id)

            texts = [c.full_text for c in doc_chunks]
            sparse_vecs = _encode_bm25(texts)

            points: list[PointStruct] = []
            for chunk, sparse in zip(doc_chunks, sparse_vecs):
                vectors: dict[str, Any] = {DENSE_VEC: chunk.embedding}
                if sparse is not None:
                    vectors[SPARSE_VEC] = sparse

                points.append(PointStruct(
                    id=chunk.chunk_index,
                    vector=vectors,
                    payload={
                        "doc_id":         chunk.doc_id,
                        "chunk_index":    chunk.chunk_index,
                        "company":        chunk.company,
                        "year":           chunk.year,
                        "report_type":    chunk.report_type,
                        "source_pdf":     chunk.source_pdf,
                        "page":           chunk.page,
                        "section":        chunk.section,
                        "text":           chunk.text,
                        "context_prefix": chunk.context_prefix,
                        "has_table":      chunk.has_table,
                        "is_figure":      chunk.is_figure,
                        "figure_caption": chunk.figure_caption,
                        "quality_score":  chunk.quality_score,
                        "table_data":     json.dumps(chunk.table_data),
                    },
                ))

            for i in range(0, len(points), UPSERT_BATCH):
                self._client.upsert(collection_name=name, points=points[i : i + UPSERT_BATCH])

            info = self._client.get_collection(name)
            print(f"  [store] {doc_id}: {len(doc_chunks)} chunks upserted "
                  f"(total in collection: {info.points_count})")

    # ── delete_doc ────────────────────────────────────────────────────────────

    def delete_doc(self, doc_id: str) -> None:
        """Remove a document's entire collection from Qdrant."""
        try:
            self._client.delete_collection(self._col(doc_id))
            print(f"  [store] deleted '{self._col(doc_id)}'")
        except Exception as e:
            print(f"  [store] delete_doc error: {e}")

    # ── list_docs ─────────────────────────────────────────────────────────────

    def list_docs(self) -> list[dict[str, str | int]]:
        """
        Return metadata for every indexed document.
        Format: [{doc_id, company, year, report_type, source_pdf, n_chunks}]
        """
        docs = []
        for col in self._client.get_collections().collections:
            if not col.name.startswith("doc_"):
                continue
            try:
                info = self._client.get_collection(col.name)
                n = info.points_count or 0
                points, _ = self._client.scroll(col.name, limit=1, with_payload=True)
                if points:
                    p = points[0].payload or {}
                    docs.append({
                        "doc_id":      p.get("doc_id", col.name[4:]),
                        "company":     p.get("company", ""),
                        "year":        p.get("year", ""),
                        "report_type": p.get("report_type", ""),
                        "source_pdf":  p.get("source_pdf", ""),
                        "n_chunks":    n,
                    })
            except Exception:
                pass
        return docs

    # ── payload_filters ───────────────────────────────────────────────────────

    def payload_filters(self, filters: dict[str, str]) -> Filter | None:
        """
        Build a Qdrant Filter from a plain dict.
        Example: {"company": "TotalEnergies", "year": "2024"}
        Returns None if filters is empty (match all).
        """
        if not filters:
            return None
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        return Filter(must=conditions)

    # ── Search helpers (used by Phase 3 search modules) ───────────────────────

    def _target_collections(
        self,
        doc_ids: list[str] | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[str]:
        """Resolve which Qdrant collections to search."""
        all_cols = [c.name for c in self._client.get_collections().collections
                    if c.name.startswith("doc_")]
        if doc_ids:
            wanted = {self._col(d) for d in doc_ids}
            return [c for c in all_cols if c in wanted]
        if filters:
            docs = self.list_docs()
            matched = [
                d for d in docs
                if all(str(d.get(k, "")).lower() == str(v).lower() for k, v in filters.items())
            ]
            return [self._col(d["doc_id"]) for d in matched]
        return all_cols

    def search_dense(
            self,
            query_vector: list[float],
            top_k: int = 30,
            doc_ids: list[str] | None = None,
            filters: dict[str, str] | None = None,
    ) -> list[SearchHit]:
        cols = self._target_collections(doc_ids, filters)
        hits: list[SearchHit] = []
        for col in cols:
            try:
                response = self._client.query_points(
                    collection_name=col,
                    query=query_vector,
                    using=DENSE_VEC,
                    limit=top_k,
                    with_payload=True,
                )
                for r in response.points:
                    hits.append(SearchHit.from_payload(r.payload or {}, r.score, "dense"))
            except Exception as e:
                print(f"  [store] dense search error in {col}: {e}")
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def search_sparse(
            self,
            query_text: str,
            top_k: int = 30,
            doc_ids: list[str] | None = None,
            filters: dict[str, str] | None = None,
    ) -> list[SearchHit]:
        sparse_vecs = _encode_bm25([query_text])
        if not sparse_vecs or sparse_vecs[0] is None:
            return []
        query_sparse = sparse_vecs[0]

        cols = self._target_collections(doc_ids, filters)
        hits: list[SearchHit] = []
        for col in cols:
            try:
                response = self._client.query_points(
                    collection_name=col,
                    query=query_sparse,
                    using=SPARSE_VEC,
                    limit=top_k,
                    with_payload=True,
                )
                for r in response.points:
                    hits.append(SearchHit.from_payload(r.payload or {}, r.score, "bm25"))
            except Exception as e:
                print(f"  [store] sparse search error in {col}: {e}")
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
