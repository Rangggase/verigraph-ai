"""v0.4 auditable batch claim -> candidate graph builder.

A claim is NOT proof of its content. No fact-checking labels or gold evidence
are used for entity extraction, linking, relations, or graph construction.
"""
from __future__ import annotations

import csv
import hashlib
import io
from itertools import islice
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import networkx as nx
from networkx.readwrite import json_graph

from src.ner_pipeline import extract_entities
from src.relation_extraction import extract_relation_candidates

LANGUAGES = {"id", "en", "zh"}
MAX_TEXT = 1500
MAX_UPLOAD_BYTES = 75 * 1024 * 1024
MAX_ROWS = 1000
QID_PATTERN = re.compile(r"^Q[1-9][0-9]*$")
COLUMN_CANDIDATES = {
    "claim": ("claim", "text", "statement", "claim_text", "content"),
    "id": ("claim_id", "id", "uid", "index"),
    "language": ("language", "lang"),
}


def load_records(content: bytes, filename: str) -> list[dict[str, Any]]:
    """Load JSON/JSONL/CSV, reject overly large or ambiguous data."""
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Dataset exceeds 75 MB v0.4 pilot safety limit; extract a smaller sample first")
    if re.search(r"(?:^|[_.-])(?:test|heldout|evaluation)(?:[_.-]|$)", Path(filename).name.lower()):
        raise ValueError("Held-out/test files are blocked in v0.4 to protect future evaluation")
    suffix = Path(filename).suffix.lower()
    text = content.decode("utf-8-sig")
    if suffix == ".csv":
        rows = list(islice(csv.DictReader(io.StringIO(text)), MAX_ROWS))
    elif suffix == ".jsonl":
        rows = [json.loads(line) for line in islice((ln for ln in text.splitlines() if ln.strip()), MAX_ROWS)]
    elif suffix == ".json":
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            # Explicitly known wrapper keys; never silently flatten a different schema.
            parsed = next((parsed[k] for k in ("records", "data", "claims")
                           if isinstance(parsed.get(k), list)), None)
        if not isinstance(parsed, list):
            raise ValueError("JSON must be a list or have a 'records'/'data'/'claims' list")
        rows = parsed[:MAX_ROWS]
    else:
        raise ValueError("Supported dataset formats: .csv, .jsonl, .json")
    if not all(isinstance(r, dict) for r in rows):
        raise ValueError("Dataset must contain an array of JSON objects / CSV rows")
    return rows


def guess_column(columns: Iterable[str], purpose: str) -> str | None:
    options = list(columns)
    lowered = {col.lower(): col for col in options}
    return next((lowered[k] for k in COLUMN_CANDIDATES[purpose] if k in lowered), None)


def normalize_records(
    rows: list[dict[str, Any]], *, claim_field: str, dataset: str,
    default_language: str, split: str = "train", id_field: str | None = None,
    language_field: str | None = None, limit: int = 10,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Create unique internal claim IDs; preserve external IDs for provenance.

    Gold labels and fact-check explanations are deliberately discarded.
    The input order is preserved for reproducibility.
    """
    if not claim_field:
        raise ValueError("Choose the claim/text column")
    if default_language not in LANGUAGES:
        raise ValueError("Default language must be id, en, or zh")
    if not dataset.strip():
        raise ValueError("Dataset name is required")
    if split.lower() in {"test", "heldout", "evaluation"}:
        raise ValueError("v0.4 blocks held-out/test sets to reduce evaluation leakage")
    if not 1 <= limit <= MAX_ROWS:
        raise ValueError(f"Limit must be 1..{MAX_ROWS}")
    records: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    for index, row in enumerate(rows[:limit]):
        raw = row.get(claim_field, "")
        if not isinstance(raw, str):
            rejected.append({"row": str(index), "reason": "claim is not a string"})
            continue
        claim = raw.strip()
        if not claim:
            rejected.append({"row": str(index), "reason": "empty claim"})
            continue
        if len(claim) > MAX_TEXT:
            rejected.append({"row": str(index), "reason": "text over 1500 characters"})
            continue
        lang = str(row.get(language_field, default_language) or default_language).lower().strip() if language_field else default_language
        if lang not in LANGUAGES:
            rejected.append({"row": str(index), "reason": f"unsupported language {lang}"})
            continue
        external_id = str(row.get(id_field, index) if id_field else index)
        # Dataset and split are part of the ID; text disambiguates duplicate source IDs.
        raw_id = "\u241f".join((dataset, split, external_id, lang, claim))
        internal_id = "claim:" + hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:20]
        records.append({
            "claim_id": internal_id, "external_id": external_id, "claim": claim,
            "language": lang, "source_dataset": dataset, "split": split,
        })
    return records, rejected


def approved_links_from_csv(content: bytes) -> dict[tuple[str, str], dict[str, str]]:
    """Explicit user-reviewed mapping only. No automatic speculative QID assignment."""
    entries = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    if len(entries) > 3000:
        raise ValueError("Too many approved links for pilot")
    mapping: dict[tuple[str, str], dict[str, str]] = {}
    for row in entries:
        language = str(row.get("language", "")).strip().lower()
        surface = str(row.get("surface", "")).strip()
        qid = str(row.get("qid", "")).strip()
        label = str(row.get("label") or surface).strip()
        if language not in LANGUAGES or not surface or not QID_PATTERN.fullmatch(qid):
            raise ValueError("Approved links CSV requires language,surface,qid,label (valid QID)")
        key = (language, surface.casefold())
        if key in mapping and mapping[key]["id"] != qid:
            raise ValueError(f"Conflicting human-reviewed mappings for {surface}")
        mapping[key] = {"id": qid, "label": label, "method": "human_reviewed_registry"}
    return mapping


def build_batch_graph(
    records: list[dict[str, str]], *, model: Any | None,
    threshold: float = .35,
    approved_links: Mapping[tuple[str, str], dict[str, str]] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[nx.MultiDiGraph, list[dict[str, Any]]]:
    """Graph with claim->mention and candidate assertion edges.

    No use of ground-truth label/evidence. Without model, makes claim-only graph.
    """
    if not .0 < threshold <= 1.0:
        raise ValueError("Invalid NER confidence threshold")
    graph = nx.MultiDiGraph()
    summaries: list[dict[str, Any]] = []
    known = approved_links or {}
    for n, record in enumerate(records, start=1):
        claim_id = record["claim_id"]
        graph.add_node(
            claim_id, node_type="unverified_claim", label=record["claim"],
            language=record["language"], source_dataset=record["source_dataset"],
            split=record["split"], external_id=record["external_id"],
            truth_status="unverified",
        )
        entities: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        error = ""
        try:
            if model is not None:
                entities = extract_entities(record["claim"], record["language"], model, threshold)
                candidates = extract_relation_candidates(record["claim"], record["language"], entities)
        except Exception as exc:
            # Keep record in graph; expose the error rather than inventing entities.
            error = f"{type(exc).__name__}: {str(exc)[:180]}"
            entities, candidates = [], []
        mentions: dict[int, str] = {}
        for i, ent in enumerate(entities):
            surface = str(ent["text"])
            link = known.get((record["language"], surface.casefold()))
            qid = str(link.get("id", "")) if link else ""
            is_linked = bool(QID_PATTERN.fullmatch(qid))
            entity_id = qid if is_linked else f"mention:{claim_id[6:]}:{i}"
            graph.add_node(
                entity_id, node_type="linked_entity" if is_linked else "unresolved_mention",
                label=(link.get("label") if link else surface),
                wikidata_id=qid, linkage_method="human_reviewed_registry" if is_linked else "unresolved",
            )
            graph.add_edge(
                claim_id, entity_id, relation="MENTIONS", edge_kind="claim_mention",
                surface=surface, ner_label=str(ent["label"]),
                ner_score=float(ent["score"]), start=int(ent["start"]), end=int(ent["end"]),
                provenance_claim_id=claim_id, extraction_method="GLiNER", truth_status="unverified",
            )
            mentions[i] = entity_id
        for cand in candidates:
            a = mentions.get(cand["subject_index"])
            b = mentions.get(cand["object_index"])
            if not a or not b or a == b:
                continue
            graph.add_edge(
                a, b, relation=str(cand["predicate"]), edge_kind="claim_assertion",
                provenance_claim_id=claim_id,
                evidence_text=str(cand["evidence_text"]), evidence_trigger=str(cand["trigger"]),
                language=record["language"], extraction_method=str(cand["method"]),
                extraction_review="pending", truth_status="unverified",
            )
        summaries.append({
            "claim_id": claim_id, "external_id": record["external_id"],
            "language": record["language"], "entities": len(entities),
            "candidate_relations": len(candidates), "error": error,
        })
        if progress_callback:
            progress_callback(n, len(records))
    return graph, summaries


def export_graph_bundle(graph: nx.MultiDiGraph, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    edge_types: dict[str, int] = {}
    for _, _, attrs in graph.edges(data=True):
        name = str(attrs.get("edge_kind", "unknown"))
        edge_types[name] = edge_types.get(name, 0) + 1
    return {
        "schema_version": "0.4",
        "warning": "EXTRACTIONS ARE NOT VERIFIED FACTS. Gold labels and justifications not ingested.",
        "stats": {"claims": len(summaries), "nodes": graph.number_of_nodes(),
                  "edges": graph.number_of_edges(), "edge_types": edge_types,
                  "ner_errors": sum(bool(x["error"]) for x in summaries)},
        "runs": summaries,
        "graph": json_graph.node_link_data(graph, edges="links"),
    }
