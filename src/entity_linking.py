"""Entity linking: local curated aliases OR *candidate* Wikidata searches.

Search result rank is NOT a ground-truth identity; user approval is required.
"""
from __future__ import annotations
import os
import re
from typing import Any

from src.core import resolve_entity

QID = re.compile(r"^Q[1-9][0-9]*$")
WIKIDATA_ACTION_API = "https://www.wikidata.org/w/api.php"


def local_link(name: str, language: str, graph: Any) -> dict | None:
    qid = resolve_entity(name, language, graph)
    if qid is None:
        return None
    return {
        "id": qid,
        "label": graph.nodes[qid].get(f"label_{language}", name),
        "url": f"https://www.wikidata.org/wiki/{qid}",
        "method": "curated_alias_match",
    }


def search_wikidata_candidates(name: str, language: str, limit: int = 5) -> list[dict]:
    """User-triggered network request. Candidates must be manually reviewed."""
    if language not in {"id", "en", "zh"}:
        raise ValueError("Use id, en or zh")
    if not name.strip():
        return []
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Install requests via requirements-ner.txt") from exc
    ua = os.getenv(
        "WIKIDATA_USER_AGENT",
        "VeriGraphAI-Portfolio/0.2 (local educational project; configure WIKIDATA_USER_AGENT)",
    )
    response = requests.get(
        WIKIDATA_ACTION_API,
        params={
            "action": "wbsearchentities",
            "search": name.strip(),
            "language": language,
            "format": "json",
            "type": "item",
            "limit": min(max(1, limit), 10),
        },
        headers={"User-Agent": ua, "Accept": "application/json"},
        timeout=12,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    candidates = []
    for hit in data.get("search", []):
        qid = str(hit.get("id", ""))
        if not QID.fullmatch(qid):
            continue
        candidates.append({
            "id": qid,
            "label": hit.get("label", ""),
            "description": hit.get("description", ""),
            "url": f"https://www.wikidata.org/wiki/{qid}",
            "method": "candidate_requires_review",
        })
    return candidates
