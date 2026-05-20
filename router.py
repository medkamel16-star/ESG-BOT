"""
esg_rag/router.py
-----------------
classify_query(q) -> QueryClass

Classifies a question into one of 5 labels so retrieve.py can apply
the right fusion weights and top-k settings.

Labels:
  factual_lookup      "What were Apple's Scope 1 emissions in 2024?"
  cross_doc_compare   "Compare Microsoft and Google's water usage"
  numeric_computation "How much did Tesla's emissions change YoY?"
  regulatory_check    "What does ESRS E1 require on Scope 3?"
  out_of_corpus       "What is the current EU ETS carbon price?"

Two-tier approach (per build guide 3.6):
  Tier 1: fast regex rules — handles the obvious cases with zero Albert calls
  Tier 2: cheap Albert call for anything the rules can't confidently classify

This keeps latency low: ~95% of ESG analyst questions are caught by Tier 1.
"""

from __future__ import annotations

import re

import esg_rag.albert as albert
from esg_rag.schemas import QueryClass

# ── Tier 1 — regex rules ──────────────────────────────────────────────────────

_COMPARE_RE = re.compile(
    r'\b(compar|vs\.?|versus|both|across|between|differ|benchmark)\b',
    re.IGNORECASE,
)
_NUMERIC_RE = re.compile(
    r'\b(how much|how many|calculat|change|growth|declin|increase|decreas|'
    r'delta|year.over.year|yoy|percent|ratio|total|sum|trend)\b',
    re.IGNORECASE,
)
_REGULATORY_RE = re.compile(
    r'\b(ESRS|CSRD|GRI|SASB|TCFD|SBTi|EU Taxonomy|regulation|directive|'
    r'standard|requirement|comply|compliance|mandate|disclosure|framework)\b',
    re.IGNORECASE,
)
_OUT_OF_CORPUS_RE = re.compile(
    r'\b(current|today|latest|recent|now|live|price|market|news|2025|2026)\b',
    re.IGNORECASE,
)
_FACTUAL_RE = re.compile(
    r'\b(what (is|are|was|were)|how (does|did|is)|tell me|show me|find|'
    r'list|give me|state|report|disclose)\b',
    re.IGNORECASE,
)

# Company filter hints
_COMPANY_HINTS = {
    "apple": "Apple", "microsoft": "Microsoft", "amazon": "Amazon",
    "google": "Google", "alphabet": "Google", "nvidia": "NVIDIA",
    "meta": "Meta", "tesla": "Tesla", "exxon": "ExxonMobil",
    "home depot": "Home Depot", "netflix": "Netflix", "oracle": "Oracle",
    "procter": "Procter & Gamble", "abbvie": "AbbVie", "abbott": "Abbott Laboratories",
    "adobe": "Adobe", "bristol": "Bristol-Myers Squibb", "costco": "Costco",
    "chevron": "Chevron", "johnson": "Johnson & Johnson", "jpmorgan": "JPMorgan Chase",
    "coca-cola": "Coca-Cola", "coke": "Coca-Cola", "linde": "Linde",
    "eli lilly": "Eli Lilly", "lilly": "Eli Lilly", "mastercard": "Mastercard",
    "mcdonald": "McDonald's", "merck": "Merck", "nike": "Nike",
    "pepsi": "PepsiCo", "pfizer": "Pfizer", "philip morris": "Philip Morris",
    "raytheon": "Raytheon Technologies", "thermo fisher": "Thermo Fisher Scientific",
    "texas instruments": "Texas Instruments", "unitedhealth": "UnitedHealth",
    "visa": "Visa", "verizon": "Verizon", "wells fargo": "Wells Fargo",
    "walmart": "Walmart", "amd": "AMD", "broadcom": "Broadcom",
    "bank of america": "Bank of America", "comcast": "Comcast",
    "salesforce": "Salesforce", "cisco": "Cisco", "danaher": "Danaher",
    "disney": "Walt Disney", "nextera": "NextEra Energy",
}

def _extract_company_filter(q: str) -> dict[str, str]:
    q_lower = q.lower()
    matches = [company for key, company in _COMPANY_HINTS.items() if key in q_lower]
    # Only apply a company filter if exactly ONE company is mentioned
    # If multiple companies → cross_doc_compare, no filter
    if len(matches) == 1:
        return {"company": matches[0]}
    return {}


def _count_companies(q: str) -> int:
    q_lower = q.lower()
    return sum(1 for key in _COMPANY_HINTS if key in q_lower)


def _tier1(q: str) -> QueryClass | None:
    """Fast regex classification. Returns None if not confident."""
    filters = _extract_company_filter(q)

    if _OUT_OF_CORPUS_RE.search(q) and not any(
        w in q.lower() for w in ("scope", "emission", "report", "disclose", "target")
    ):
        return QueryClass("out_of_corpus", 0.85, {}, 0)

    if _COMPARE_RE.search(q) or _count_companies(q) >= 2:
        return QueryClass("cross_doc_compare", 0.92, {}, 12)

    if _NUMERIC_RE.search(q):
        return QueryClass("numeric_computation", 0.82, filters, 8)

    if _REGULATORY_RE.search(q):
        return QueryClass("regulatory_check", 0.90, {}, 6)

    if _FACTUAL_RE.search(q):
        return QueryClass("factual_lookup", 0.80, filters, 8)

    return None  # Tier 2 needed


# ── Tier 2 — cheap Albert call ────────────────────────────────────────────────

_SYSTEM = """You classify ESG analyst questions into exactly one of these categories:
factual_lookup, cross_doc_compare, numeric_computation, regulatory_check, out_of_corpus

Rules:
- factual_lookup: asking for a specific stated fact from a report
- cross_doc_compare: comparing metrics across companies or years
- numeric_computation: requires arithmetic or YoY calculation
- regulatory_check: about ESRS/CSRD/GRI/TCFD/regulation requirements
- out_of_corpus: requires live data or information not in ESG reports

Respond with JSON only: {"label": "...", "confidence": 0.0-1.0}"""


def _tier2(q: str) -> QueryClass:
    """Albert classification for ambiguous questions."""
    try:
        resp = albert.chat_text(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"Question: {q}"},
            ],
            model=albert.CHAT_MODEL,
            max_tokens=60,
            temperature=0.0,
            caller="router",
        )
        import json
        data = json.loads(resp.strip())
        label = data.get("label", "factual_lookup")
        conf  = float(data.get("confidence", 0.7))
        filters = _extract_company_filter(q)
        top_k = {"cross_doc_compare": 12, "out_of_corpus": 0}.get(label, 8)
        return QueryClass(label, conf, filters, top_k)
    except Exception:
        # Safest fallback
        return QueryClass("factual_lookup", 0.5, _extract_company_filter(q), 8)


# ── Public API ────────────────────────────────────────────────────────────────

def classify_query(q: str) -> QueryClass:
    """
    Classify a query into a QueryClass.
    Tries Tier 1 (regex) first, falls back to Tier 2 (Albert) if needed.
    """
    result = _tier1(q)
    if result is not None:
        return result
    return _tier2(q)
