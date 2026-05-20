"""
esg_rag/albert.py
-----------------
Singleton Albert client. Every Albert API call in the project goes through here.
Do NOT call the Albert API directly anywhere else.

Functions:
  client()                              → httpx.Client singleton
  embed_texts(texts, model)            → list[list[float]], batched at 64
  chat(messages, tools=None)           → full API response dict
  chat_stream(messages)                → generator of text tokens
  rerank(query, docs, top_n=8)         → list[{index, score, document}]
  list_models()                        → list[str]

All calls:
  - Log to data/albert_costs.jsonl (ts, caller, model, tokens_in, tokens_out, latency_ms)
  - Retry with exponential backoff on 429 / 5xx (max 5 attempts)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Generator

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

ALBERT_BASE_URL: str = os.environ["ALBERT_BASE_URL"].rstrip("/")
ALBERT_API_KEY: str  = os.environ["ALBERT_API_KEY"]

CHAT_MODEL   = os.getenv("ALBERT_CHAT_MODEL",   "mistralai/Ministral-3-8B-Instruct-2512")
STRONG_MODEL = os.getenv("ALBERT_STRONG_MODEL", "openai/gpt-4o-mini")
EMBED_MODEL  = os.getenv("ALBERT_EMBED_MODEL",  "BAAI/bge-m3")
RERANK_MODEL = os.getenv("ALBERT_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

COST_LOG = Path("data/albert_costs.jsonl")
COST_LOG.parent.mkdir(parents=True, exist_ok=True)

EMBED_BATCH = 4
MAX_RETRIES = 5

# ── Singleton client ──────────────────────────────────────────────────────────

_client: httpx.Client | None = None


def client() -> httpx.Client:
    """Return the singleton httpx.Client (created on first call)."""
    global _client
    if _client is None:
        _client = httpx.Client(
            base_url=ALBERT_BASE_URL,
            headers={
                "Authorization": f"Bearer {ALBERT_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )
    return _client


# ── Internal helpers ──────────────────────────────────────────────────────────

def _log(caller: str, model: str, tokens_in: int, tokens_out: int, latency_ms: float) -> None:
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "caller": caller,
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": round(latency_ms, 1),
    }
    with COST_LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST with exponential backoff on 429 / 5xx."""
    c = client()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = c.post(path, content=json.dumps(payload))
            if r.status_code == 429 or r.status_code >= 500:
                wait = 2 ** attempt
                print(f"  [albert] {r.status_code} on {path} — retrying in {wait}s "
                      f"(attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except httpx.TimeoutException:
            wait = 2 ** attempt
            print(f"  [albert] timeout on {path} — retrying in {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"Albert API failed after {MAX_RETRIES} retries: {path}")


def _get(path: str) -> dict[str, Any]:
    r = client().get(path)
    r.raise_for_status()
    return r.json()


# ── Public API ────────────────────────────────────────────────────────────────

def list_models() -> list[str]:
    """Return model IDs available on this Albert instance."""
    data = _get("/v1/models")
    return [m["id"] for m in data.get("data", [])]


def embed_texts(
    texts: list[str],
    model: str = EMBED_MODEL,
    caller: str = "embed",
) -> list[list[float]]:
    """
    Embed a list of strings. Batches at EMBED_BATCH (64) to stay within limits.
    Returns vectors in the same order as input.
    """
    all_vecs: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = [t[:800] for t in texts[i : i + EMBED_BATCH]]
        t0 = time.perf_counter()
        data = _post("/v1/embeddings", {"model": model, "input": batch})
        latency = (time.perf_counter() - t0) * 1000

        # Albert sorts by index; preserve order explicitly
        vecs = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        all_vecs.extend(vecs)

        usage = data.get("usage", {})
        _log(caller, model, usage.get("prompt_tokens", 0), 0, latency)

    return all_vecs


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict] | None = None,
    model: str = CHAT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    caller: str = "chat",
) -> dict[str, Any]:
    """
    Call Albert chat completions. Returns the full API response dict.
    Access the reply with: resp["choices"][0]["message"]["content"]

    If tools is not None, adds tool_choice="auto". If the model doesn't
    support native tools, this will be ignored gracefully (agent.py handles
    the ReAct fallback based on docs/albert_capabilities.md).
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    t0 = time.perf_counter()
    data = _post("/v1/chat/completions", payload)
    latency = (time.perf_counter() - t0) * 1000

    usage = data.get("usage", {})
    _log(caller, model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), latency)
    return data


def chat_text(
    messages: list[dict[str, Any]],
    model: str = CHAT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    caller: str = "chat_text",
) -> str:
    """Convenience wrapper — returns just the assistant text."""
    resp = chat(messages, model=model, temperature=temperature,
                max_tokens=max_tokens, caller=caller)
    return resp["choices"][0]["message"]["content"] or ""


def chat_stream(
    messages: list[dict[str, Any]],
    model: str = STRONG_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    caller: str = "chat_stream",
) -> Generator[str, None, None]:
    """
    Stream the assistant response. Yields text tokens as they arrive.
    Usage:
        for token in chat_stream(messages):
            print(token, end="", flush=True)
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    t0 = time.perf_counter()
    tokens_out = 0

    with client().stream("POST", "/v1/chat/completions", content=json.dumps(payload)) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(raw)
                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    tokens_out += 1
                    yield token
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    latency = (time.perf_counter() - t0) * 1000
    _log(caller, model, 0, tokens_out, latency)


def rerank(
    query: str,
    docs: list[str],
    top_n: int = 8,
    model: str = RERANK_MODEL,
    caller: str = "rerank",
) -> list[dict[str, Any]]:
    """
    Re-rank docs against query using Albert's cross-encoder.
    Returns list of {index, score, document} sorted by score descending,
    with scores min-max normalised to [0, 1].

    Falls back gracefully (passthrough order, score=0) if /rerank is unavailable.
    """
    payload: dict[str, Any] = {
        "model": model,
        "query": query,
        "documents": docs,
        "top_n": top_n,
    }
    t0 = time.perf_counter()
    try:
        data = _post("/v1/rerank", payload)
        latency = (time.perf_counter() - t0) * 1000
        _log(caller, model, len(docs), 0, latency)

        results = data.get("results", data.get("data", []))
        ranked = sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)

        # Min-max normalise
        scores = [r.get("relevance_score", 0) for r in ranked]
        lo, hi = min(scores, default=0), max(scores, default=1)
        rng = hi - lo if hi > lo else 1.0

        return [
            {
                "index": r["index"],
                "score": round((r.get("relevance_score", 0) - lo) / rng, 4),
                "document": docs[r["index"]],
            }
            for r in ranked
        ]

    except (httpx.HTTPStatusError, RuntimeError) as e:
        print(f"  [albert] rerank unavailable ({e}) — passthrough order")
        return [{"index": i, "score": 0.0, "document": d} for i, d in enumerate(docs[:top_n])]


# ── Cost summary ──────────────────────────────────────────────────────────────

def cost_summary(since_date: str | None = None) -> dict[str, Any]:
    """
    Summarise the cost log. Pass since_date="2025-05-19" to filter by day.
    Returns: {total_calls, total_tokens_in, total_tokens_out, by_model, by_caller}
    """
    if not COST_LOG.exists():
        return {"total_calls": 0, "total_tokens_in": 0, "total_tokens_out": 0}
    rows = [json.loads(l) for l in COST_LOG.read_text().splitlines() if l.strip()]
    if since_date:
        rows = [r for r in rows if r["ts"].startswith(since_date)]

    summary: dict[str, Any] = {
        "total_calls": len(rows),
        "total_tokens_in": sum(r["tokens_in"] for r in rows),
        "total_tokens_out": sum(r["tokens_out"] for r in rows),
        "by_model": {},
        "by_caller": {},
    }
    for r in rows:
        for dim in ("model", "caller"):
            key = r[dim]
            b = summary[f"by_{dim}"].setdefault(key, {"calls": 0, "tokens_in": 0, "tokens_out": 0})
            b["calls"] += 1
            b["tokens_in"] += r["tokens_in"]
            b["tokens_out"] += r["tokens_out"]
    return summary
