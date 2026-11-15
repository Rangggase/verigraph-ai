"""Lightweight multilingual KG demo (rule-based, NOT a fake-news ML model).

All annotations in data/ are hand-curated examples. The graph's facts are not
an exhaustive oracle and cannot establish truth of arbitrary news claims.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def read_csv(filename: str) -> list[dict[str, str]]:
    with (DATA_DIR / filename).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def make_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    for entity in read_csv("entities.csv"):
        entity_id = entity.pop("entity_id")
        graph.add_node(entity_id, **entity)
    for relation in read_csv("relations.csv"):
        source_id, target_id = relation["subject_id"], relation["object_id"]
        if source_id not in graph or target_id not in graph:
            raise ValueError(f"Unknown entity in edge: {source_id} -> {target_id}")
        graph.add_edge(source_id, target_id, **relation)
    return graph


def resolve_entity(name: str, lang: str, graph: nx.DiGraph) -> str | None:
    normalized = name.strip().casefold()
    for entity_id, attrs in graph.nodes(data=True):
        choices = [attrs.get(f"label_{lang}", "")]
        choices += attrs.get(f"aliases_{lang}", "").split("|")
        if any(normalized == alias.strip().casefold() for alias in choices if alias.strip()):
            return entity_id
    return None


PATTERNS = {
    "en": re.compile(r"^(.+?)\s+is\s+the\s+capital\s+of\s+(.+?)[.!?]?\s*$", re.I),
    "id": re.compile(r"^(.+?)\s+(?:adalah|merupakan)\s+ibu\s+kota\s+(.+?)[.!?]?\s*$", re.I),
    "zh": re.compile(r"^(.+?)是(.+?)的首都[。.!?]?\s*$"),
}


def parse_demo_claim(text: str, language: str, graph: nx.DiGraph) -> dict:
    """Rule-based parsing for a single relation: CITY is capital of COUNTRY."""
    pattern = PATTERNS.get(language)
    if not pattern:
        return {"status": "unsupported_language", "message": "Use en, id, or zh"}
    match = pattern.match(text.strip())
    if not match:
        return {
            "status": "unparsed",
            "message": "Demo parser only understands simple 'city is capital of country' claims.",
        }

    city_name, country_name = match.groups()
    city_id = resolve_entity(city_name, language, graph)
    country_id = resolve_entity(country_name, language, graph)
    if not city_id or not country_id:
        return {
            "status": "insufficient_evidence",
            "message": "At least one entity is not linked in the demo KG.",
            "city_id": city_id,
            "country_id": country_id,
        }

    data = graph.get_edge_data(country_id, city_id)
    capital_candidates = [
        (neighbor, edge)
        for neighbor, edge in graph[country_id].items()
        if edge.get("predicate") == "capital"
    ]
    result = {
        "city_id": city_id,
        "country_id": country_id,
        "city": graph.nodes[city_id][f"label_{language}"],
        "country": graph.nodes[country_id][f"label_{language}"],
    }
    if data and data.get("predicate") == "capital":
        result.update({
            "status": "supported",
            "message": "The curated demonstration KG supports this capital relation.",
            "source_url": data.get("reference_url"),
        })
    elif capital_candidates:
        candidate_id, edge = capital_candidates[0]
        result.update({
            "status": "conflict",
            "message": "This conflicts with the capital relation stored in the demonstration KG. This is not a general fake-news verdict.",
            "stored_capital": graph.nodes[candidate_id][f"label_{language}"],
            "source_url": edge.get("reference_url"),
        })
    else:
        result.update({
            "status": "insufficient_evidence",
            "message": "No capital relation exists for this country in the demonstration KG.",
        })
    return result
