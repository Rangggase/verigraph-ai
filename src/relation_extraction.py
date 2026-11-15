"""Auditable v0.3 multilingual *candidate* relation extraction.

Rule-based bootstrap only: these triples are the claims' assertions, NEVER
verified facts. Intended to be replaced/compared with an LLM extractor later.
"""
from __future__ import annotations

import re
from typing import Mapping

import networkx as nx

from src.assertion_graph import build_mention_graph

LANGUAGES = {"id", "en", "zh"}
# Each predicate includes language-specific triggers and admissible object types.
RULES = {
    "id": [
        ("VISITS", r"\b(?:melakukan\s+kunjungan\s+kerja|berkunjung|mengunjungi|kunjungan\s+kerja|melakukan\s+kunjungan)\b", {"location"}),
        ("MEETS", r"\b(?:bertemu|menemui)\b", {"person", "organization"}),
        ("ATTENDS_IN", r"\b(?:menghadiri|hadir\s+di|mengikuti\s+pertemuan)\b", {"location"}),
    ],
    "en": [
        ("VISITS", r"\b(?:visits?|visited|travel(?:s|led)?\s+to)\b", {"location"}),
        ("MEETS", r"\b(?:meets?|met)\b", {"person", "organization"}),
        ("ATTENDS_IN", r"\b(?:attends?|attended)\b", {"location"}),
    ],
    "zh": [
        ("VISITS", r"(?:访问|到访|拜访|前往)", {"location"}),
        ("MEETS", r"(?:会见|会晤|见面)", {"person", "organization"}),
        ("ATTENDS_IN", r"(?:出席|参加)", {"location"}),
    ],
}


def _sentence_segments(text: str) -> list[tuple[int, int]]:
    """Keep original character offsets; do not split on intra-name spaces."""
    boundaries = [0]
    for match in re.finditer(r"[.!?。！？;；\n]+", text):
        boundaries.append(match.end())
    boundaries.append(len(text))
    return [(a, b) for a, b in zip(boundaries, boundaries[1:]) if text[a:b].strip()]


def extract_relation_candidates(text: str, language: str, entities: list[dict]) -> list[dict]:
    """Return only subject-trigger-object patterns, with explicit text spans.

    No relation is inferred without a verb/phrase trigger in the same sentence.
    At most one best admissible subject and object per rule occurrence.
    """
    if language not in LANGUAGES:
        raise ValueError("Unsupported language")
    if not text.strip() or not entities:
        return []
    results = []
    dedup = set()
    for sentence_start, sentence_end in _sentence_segments(text):
        sent = text[sentence_start:sentence_end]
        indexed = [(i, e) for i, e in enumerate(entities)
                   if sentence_start <= int(e["start"]) < int(e["end"]) <= sentence_end]
        for predicate, pattern, allowed_objects in RULES[language]:
            for hit in re.finditer(pattern, sent, flags=re.IGNORECASE):
                trigger_start = sentence_start + hit.start()
                trigger_end = sentence_start + hit.end()
                # Subjects must occur before the trigger; objects after it.
                subjects = [(i, e) for i, e in indexed
                            if e.get("label") == "person" and int(e["end"]) <= trigger_start]
                objects = [(i, e) for i, e in indexed
                           if e.get("label") in allowed_objects and int(e["start"]) >= trigger_end]
                # Chinese often places locations before the verb: "在北京出席会议".
                if language == "zh" and predicate == "ATTENDS_IN" and not objects:
                    objects = [(i, e) for i, e in indexed
                               if e.get("label") == "location" and int(e["end"]) <= trigger_start]
                if not subjects or not objects:
                    continue
                subject_i, subject = min(subjects, key=lambda p: trigger_start - int(p[1]["end"]))
                object_i, obj = min(objects, key=lambda p: abs(int(p[1]["start"]) - trigger_end))
                if subject_i == object_i:
                    continue
                key = (subject_i, predicate, object_i, trigger_start)
                if key in dedup:
                    continue
                dedup.add(key)
                results.append({
                    "subject_index": subject_i,
                    "subject": subject["text"],
                    "predicate": predicate,
                    "object_index": object_i,
                    "object": obj["text"],
                    "trigger": text[trigger_start:trigger_end],
                    "trigger_start": trigger_start,
                    "trigger_end": trigger_end,
                    "evidence_text": sent.strip(),
                    "language": language,
                    "method": "rule_based_v0.3",
                    "extraction_review": "pending",
                    "truth_status": "unverified",
                })
    return results


def build_assertion_graph(
    text: str,
    language: str,
    entities: list[dict],
    links: Mapping[int, dict | None],
    candidates: list[dict],
    review: Mapping[int, str] | None = None,
) -> nx.MultiDiGraph:
    """Build provenance-bearing graph, without promoting any edge to fact."""
    graph = build_mention_graph(text, language, entities, links)
    claim_id = next(n for n, a in graph.nodes(data=True) if a.get("node_type") == "unverified_claim")
    # Entity -> node ID mirrors the mention graph, including its QID merging.
    mention_targets: dict[int, str] = {}
    for _, target, _, data in graph.out_edges(claim_id, keys=True, data=True):
        index = next((i for i, e in enumerate(entities)
                      if e["text"] == data["surface"]
                      and int(e["start"]) == int(data["start"])), None)
        if index is not None:
            mention_targets[index] = target
    for i, candidate in enumerate(candidates):
        extraction_review = (review or {}).get(i, "pending")
        if extraction_review not in {"pending", "approved_extraction", "rejected"}:
            raise ValueError("Invalid extraction review status")
        if extraction_review == "rejected":
            continue
        s = mention_targets.get(candidate["subject_index"])
        o = mention_targets.get(candidate["object_index"])
        if s is None or o is None or s == o:
            continue
        graph.add_edge(
            s, o, relation=candidate["predicate"],
            provenance_claim_id=claim_id,
            evidence_text=candidate["evidence_text"],
            evidence_trigger=candidate["trigger"],
            trigger_start=int(candidate["trigger_start"]),
            trigger_end=int(candidate["trigger_end"]),
            language=language,
            extraction_method=candidate["method"],
            extraction_review=extraction_review,
            truth_status="unverified",
            edge_kind="claim_assertion",
        )
    return graph
