"""Build a claim->entity mention graph, separate from the trusted facts KG.

MENTIONS edges do not establish factual truth or relationships among entities.
"""
from __future__ import annotations
import hashlib
import re
from typing import Mapping

import networkx as nx

VALID_QID = re.compile(r"^Q[1-9][0-9]*$")


def build_mention_graph(
    text: str, language: str, entities: list[dict], links: Mapping[int, dict | None]
) -> nx.MultiDiGraph:
    if language not in {"id", "en", "zh"}:
        raise ValueError("Unsupported language")
    g = nx.MultiDiGraph()
    digest = hashlib.sha256(f"{language}:{text}".encode("utf-8")).hexdigest()[:16]
    claim_id = f"claim:{digest}"
    g.add_node(claim_id, node_type="unverified_claim", label=text, language=language)
    for index, entity in enumerate(entities):
        qid_data = links.get(index) or {}
        qid = str(qid_data.get("id", ""))
        approved_link = bool(VALID_QID.fullmatch(qid))
        node_id = qid if approved_link else f"mention:{digest}:{index}"
        method = qid_data.get("method", "unresolved") if approved_link else "unresolved"
        g.add_node(
            node_id,
            node_type="linked_entity" if approved_link else "unresolved_mention",
            label=qid_data.get("label") or entity["text"],
            wikidata_id=qid if approved_link else "",
            linkage_method=method,
        )
        g.add_edge(
            claim_id, node_id,
            relation="MENTIONS",
            surface=entity["text"],
            ner_label=entity["label"],
            ner_score=float(entity["score"]),
            start=int(entity["start"]),
            end=int(entity["end"]),
            claim_status="unverified",
        )
    return g
