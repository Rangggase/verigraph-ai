"""VeriGraph AI v0.4.2 — reviewed, canonical, leakage-aware graph exports.

All graph edges derived from a claim are unverified *assertions*, even when the
extractor and entity linking have been human-reviewed. Do not convert an
accepted extraction into a verified factual relation.

Input: an exported v0.4.1 bundle. Output: separate audit and model-ready clean
graphs, review tables, and reproducible, fail-closed validation diagnostics.
"""
from __future__ import annotations

from collections import Counter
import copy
import json
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph

from src.kg_enrichment import QID, VALID_DECISIONS, mention_key

RELATION_DECISIONS = {"pending", "accepted_extraction", "rejected_extraction"}
PROHIBITED_FIELDS = {"gold_label", "label_original", "justification", "evidence_json", "veracity", "is_fake", "ground_truth"}
LIMIT_BYTES = 12 * 1024 * 1024


def edge_id(src: str, dst: str, key: Any, claim: str) -> str:
    return json.dumps([src, dst, key, claim], ensure_ascii=False, separators=(",", ":"))


def load_v041(payload: bytes | str | dict[str, Any]):
    """Import v0.4.1 strictly, before passing the graph to NetworkX.

    Validations prevent orphan endpoints being silently auto-created by
    NetworkX, spurious factual truth, and injected gold labels.
    """
    if isinstance(payload, bytes):
        if len(payload) > LIMIT_BYTES:
            raise ValueError("Graph file too large for pilot")
        raw = json.loads(payload.decode("utf-8-sig"))
    elif isinstance(payload, str):
        if len(payload.encode("utf-8")) > LIMIT_BYTES:
            raise ValueError("Graph file too large for pilot")
        raw = json.loads(payload)
    else:
        raw = copy.deepcopy(payload)
    if not isinstance(raw, dict) or raw.get("schema_version") != "0.4.1":
        raise ValueError("Expected a v0.4.1 reviewed KG JSON")
    serialized = raw.get("graph")
    if not isinstance(serialized, dict) or not isinstance(serialized.get("nodes"), list) or not isinstance(serialized.get("links"), list):
        raise ValueError("Invalid graph layout")
    if len(serialized["nodes"]) > 5000 or len(serialized["links"]) > 10000:
        raise ValueError("Pilot graph too large")
    if serialized.get("directed") is not True or serialized.get("multigraph") is not True:
        raise ValueError("Expected a directed MultiDiGraph")
    nodes = serialized["nodes"]
    links = serialized["links"]
    node_ids = [n.get("id") if isinstance(n, dict) else None for n in nodes]
    if any(not isinstance(i, str) or not i for i in node_ids) or len(set(node_ids)) != len(node_ids):
        raise ValueError("Invalid or duplicate graph node IDs")
    node_set = set(node_ids)
    for n in nodes:
        if PROHIBITED_FIELDS & n.keys():
            raise ValueError("Prohibited gold label or evidence field in node")
        if n.get("node_type") == "unverified_claim" and n.get("truth_status") != "unverified":
            raise ValueError("Claim must be explicitly unverified")
    for e in links:
        if not isinstance(e, dict) or PROHIBITED_FIELDS & e.keys():
            raise ValueError("Prohibited or malformed graph edge")
        if e.get("source") not in node_set or e.get("target") not in node_set:
            raise ValueError("Edge references missing node")
        if e.get("truth_status") != "unverified":
            raise ValueError("Graph edge must be unverified")
    graph = json_graph.node_link_graph(serialized, edges="links", directed=True, multigraph=True)
    if not isinstance(graph, nx.MultiDiGraph) or graph.number_of_nodes() != len(nodes) or graph.number_of_edges() != len(links):
        raise ValueError("Graph import unexpectedly changed size")
    return raw, graph


def resume_v042_as_v041(payload: bytes | str | dict[str, Any]):
    """Resume existing v0.4.2 decisions without rerunning NER or reverting review edits.

    Returns a validated v0.4.1-shaped source containing the v0.4.2 audit graph,
    preserving all review metadata. Accepting extraction is NOT truth verification.
    """
    if isinstance(payload, bytes):
        if len(payload) > LIMIT_BYTES:
            raise ValueError("Reviewed graph file too large")
        saved = json.loads(payload.decode("utf-8-sig"))
    elif isinstance(payload, str):
        if len(payload.encode("utf-8")) > LIMIT_BYTES:
            raise ValueError("Reviewed graph file too large")
        saved = json.loads(payload)
    else:
        saved = copy.deepcopy(payload)
    if not isinstance(saved, dict) or saved.get("schema_version") != "0.4.2":
        raise ValueError("Expected a v0.4.2 reviewed graph")
    source = {
        "schema_version": "0.4.1",
        "graph": saved.get("audit_graph"),
        "mention_reviews": saved.get("mention_reviews"),
    }
    raw, graph = load_v041(source)
    get_mention_rows(raw, graph)
    get_relation_rows(graph)
    return raw, graph


def get_mention_rows(bundle: dict[str, Any], graph: nx.MultiDiGraph) -> list[dict[str, Any]]:
    """Recover all 0.4.1 review entries, ensuring each maps to an actual mention edge."""
    rows = bundle.get("mention_reviews")
    if not isinstance(rows, list):
        raise ValueError("Missing mention_reviews")
    actual = {}
    for src, dst, key, e in graph.edges(keys=True, data=True):
        if e.get("edge_kind") != "claim_mention":
            continue
        if graph.nodes[src].get("node_type") != "unverified_claim":
            raise ValueError("MENTIONS edge must originate from a claim")
        identity = mention_key(str(src), str(dst), key)
        actual[identity] = (str(src), str(dst), key, e)
    seen = set()
    for r in rows:
        if not isinstance(r, dict) or PROHIBITED_FIELDS & r.keys():
            raise ValueError("Malformed or unsafe mention review")
        if r.get("mention_key") not in actual or r["mention_key"] in seen:
            raise ValueError("Review row missing, duplicated, or unmatched")
        seen.add(r["mention_key"])
        src, dst, key, edge = actual[r["mention_key"]]
        if str(r.get("target_id")) != dst or str(r.get("claim_id")) != src or str(r.get("surface")) != str(edge.get("surface")):
            raise ValueError("Review data differs from original mention edge")
        if r.get("decision") not in VALID_DECISIONS:
            raise ValueError("Unsupported mention decision")
        qid = str(r.get("qid", "") or "").strip().upper()
        if qid and (r["decision"] != "accepted_mention" or not QID.fullmatch(qid)):
            raise ValueError("QID must be valid and manually accepted")
        e_qid = str(edge.get("reviewed_qid", "") or "").strip().upper()
        # Legacy exports can be reused. Do not silently treat mismatched QIDs as verified.
        if e_qid and e_qid != qid:
            raise ValueError("Mention QID disagrees with graph edge")
    if seen != set(actual):
        raise ValueError("Review table does not cover every mention edge")
    return copy.deepcopy(rows)


def get_relation_rows(graph: nx.MultiDiGraph) -> list[dict[str, Any]]:
    """Separate human review of relation extraction from truth verification."""
    rows = []
    for src, dst, key, e in graph.edges(keys=True, data=True):
        if e.get("edge_kind") != "claim_assertion":
            continue
        claim = str(e.get("provenance_claim_id", ""))
        if claim not in graph or graph.nodes[claim].get("node_type") != "unverified_claim":
            raise ValueError("Relation provenance must point to a claim node")
        if src == dst or not e.get("relation"):
            raise ValueError("Invalid self-relation or predicate")
        if not isinstance(e.get("evidence_text"), str) or e["evidence_text"] not in str(graph.nodes[claim].get("label", "")):
            raise ValueError("Relation evidence must be an exact quote from its claim")
        review = e.get("extraction_review", "pending")
        if review not in RELATION_DECISIONS:
            raise ValueError("Invalid original relation review")
        rows.append({
            "relation_key": edge_id(str(src), str(dst), key, claim),
            "claim_id": claim,
            "external_id": str(graph.nodes[claim].get("external_id", "")),
            "subject": str(graph.nodes[src].get("label", src)),
            "predicate": str(e["relation"]),
            "object": str(graph.nodes[dst].get("label", dst)),
            "evidence_text": str(e["evidence_text"]),
            "decision": review,
        })
    return sorted(rows, key=lambda r: (r["external_id"], r["relation_key"]))


def build_reviewed_graph(bundle: dict[str, Any], mention_rows: list[dict[str, Any]],
                         relation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build audit and clean graphs, preserving direction, provenance and no gold labels.

    Clean graph: all claim nodes; ONLY accepted mentions; shared QID nodes if
    explicitly manually reviewed; ONLY accepted extraction relations whose BOTH
    endpoints have accepted mentions. No statement is marked verified.
    """
    raw, graph = load_v041(bundle)
    originals = get_mention_rows(raw, graph)
    expected = {r["mention_key"]: r for r in originals}
    supplied = {str(r.get("mention_key")): r for r in mention_rows}
    if len(supplied) != len(mention_rows) or set(supplied) != set(expected):
        raise ValueError("Mention review rows must match original keys")

    if any(PROHIBITED_FIELDS & r.keys() for r in mention_rows + relation_rows):
        raise ValueError("Review input contains prohibited gold fields")
    accepted_targets: dict[str, dict[str, Any]] = {}
    decisions = Counter()
    resolved_qids: dict[str, str] = {}
    for src, dst, key, e in graph.edges(keys=True, data=True):
        if e.get("edge_kind") != "claim_mention":
            continue
        review_key = mention_key(str(src), str(dst), key)
        r = supplied[review_key]
        base = expected[review_key]
        if any(str(r.get(k, "")) != str(base.get(k, "")) for k in ("claim_id", "target_id", "surface")):
            raise ValueError("Immutable mention data was changed")
        decision = str(r.get("decision", ""))
        qid = str(r.get("qid", "") or "").strip().upper()
        if decision not in VALID_DECISIONS:
            raise ValueError("Invalid mention review decision")
        if qid and (decision != "accepted_mention" or not QID.fullmatch(qid)):
            raise ValueError("QID requires accepted mention and valid Q-number")
        e["mention_review"] = decision
        e["reviewed_qid"] = qid
        e["truth_status"] = "unverified"
        decisions[decision] += 1
        if decision == "accepted_mention":
            if dst in accepted_targets and (accepted_targets[dst]["qid"] != qid):
                raise ValueError("Conflicting QIDs for a mention node")
            accepted_targets[dst] = {"qid": qid, "label": str(r.get("canonical_label", "")).strip(), "surface": str(base["surface"])}
            if qid:
                resolved_qids[dst] = qid

    original_relations = get_relation_rows(graph)
    by_key = {r["relation_key"]: r for r in original_relations}
    entered = {str(r.get("relation_key")): r for r in relation_rows}
    if len(entered) != len(relation_rows) or set(entered) != set(by_key):
        raise ValueError("Relation review rows must match all candidate relations")

    relation_decisions = Counter()
    allowed_relation_edges = []
    skipped_relations = 0
    for src, dst, key, e in graph.edges(keys=True, data=True):
        if e.get("edge_kind") != "claim_assertion":
            continue
        lookup = edge_id(str(src), str(dst), key, str(e["provenance_claim_id"]))
        incoming = entered[lookup]
        expected_row = by_key[lookup]
        if any(str(incoming.get(k, "")) != str(expected_row.get(k, "")) for k in ("claim_id", "subject", "predicate", "object", "evidence_text")):
            raise ValueError("Immutable relation data was changed")
        decision = str(incoming.get("decision", ""))
        if decision not in RELATION_DECISIONS:
            raise ValueError("Unsupported relation decision")
        e["extraction_review"] = decision
        e["truth_status"] = "unverified"
        relation_decisions[decision] += 1
        if decision == "accepted_extraction":
            claim_id = str(e["provenance_claim_id"])
            # Accepting an extracted relation does NOT approve either entity.
            if src not in accepted_targets or dst not in accepted_targets:
                skipped_relations += 1
                continue
            linked_to_claim = {
                str(t) for _, t, e2 in graph.out_edges(claim_id, data=True)
                if e2.get("edge_kind") == "claim_mention" and e2.get("mention_review") == "accepted_mention"
            }
            if src not in linked_to_claim or dst not in linked_to_claim:
                skipped_relations += 1
                continue
            allowed_relation_edges.append((str(src), str(dst), dict(e)))

    audit = graph
    clean = nx.MultiDiGraph(graph_role="reviewed_clean", truth_status="unverified")
    for node, attrs in audit.nodes(data=True):
        if attrs.get("node_type") == "unverified_claim":
            clean.add_node(node, **copy.deepcopy(attrs))

    # Use QID as *direct* canonical node in the model input graph.
    # Unlinked, reviewed mentions keep their unique local IDs (no guessed merges).
    def canonical_for(node: str) -> str:
        return resolved_qids.get(node, node)

    for src, dst, key, e in audit.edges(keys=True, data=True):
        if e.get("edge_kind") != "claim_mention" or e.get("mention_review") != "accepted_mention":
            continue
        canonical = canonical_for(str(dst))
        if canonical not in clean:
            attrs = copy.deepcopy(audit.nodes[dst])
            if canonical != dst:
                attrs.update(node_type="canonical_entity", label=accepted_targets[dst]["label"] or attrs.get("label", ""), wikidata_id=canonical, linkage_method="human_reviewed_qid")
            else:
                attrs.update(node_type="accepted_unresolved_mention", linkage_method="reviewed_no_qid")
            clean.add_node(canonical, **attrs)
        copied = copy.deepcopy(e)
        copied["source_mention_id"] = str(dst)
        copied["truth_status"] = "unverified"
        clean.add_edge(src, canonical, **copied)

    for src, dst, e in allowed_relation_edges:
        canonical_src, canonical_dst = canonical_for(src), canonical_for(dst)
        if canonical_src not in clean or canonical_dst not in clean:
            skipped_relations += 1
            continue
        if canonical_src == canonical_dst:
            skipped_relations += 1
            continue
        copied = copy.deepcopy(e)
        copied["source_mention_id"] = src
        copied["target_mention_id"] = dst
        copied["truth_status"] = "unverified"
        clean.add_edge(canonical_src, canonical_dst, **copied)

    # Keep the semantic graph direction intact. Separately derive symmetric
    # message-passing adjacency so entity -> claim information can actually
    # reach claim nodes during future GNN training.
    gnn = nx.MultiDiGraph(graph_role="message_passing", truth_status="unverified")
    gnn.add_nodes_from((n, copy.deepcopy(a)) for n, a in clean.nodes(data=True))
    for src, dst, e in clean.edges(data=True):
        gnn.add_edge(src, dst, **copy.deepcopy(e))
        reverse = copy.deepcopy(e)
        reverse["relation"] = "INVERSE_" + str(e.get("relation", ""))
        reverse["edge_kind"] = "inverse_message_passing"
        reverse["reverse_of_kind"] = str(e.get("edge_kind", ""))
        reverse["truth_status"] = "unverified"
        gnn.add_edge(dst, src, **reverse)

    node_ids = sorted(str(n) for n in gnn.nodes)
    node_mapping = {node_id: i for i, node_id in enumerate(node_ids)}
    relation_names = sorted({str(e.get("relation", "")) for _, _, e in gnn.edges(data=True)})
    relation_mapping = {name: i for i, name in enumerate(relation_names)}
    sorted_edges = sorted(((str(src), str(dst), str(e.get("relation", "")))
                           for src, dst, e in gnn.edges(data=True)))
    gnn_ready = {
        "node_mapping": node_mapping,
        "edge_type_mapping": relation_mapping,
        "edge_index": [
            [node_mapping[s] for s, _, _ in sorted_edges],
            [node_mapping[t] for _, t, _ in sorted_edges],
        ],
        "edge_type": [relation_mapping[r] for _, _, r in sorted_edges],
        "num_nodes": len(node_mapping),
        "num_edges": len(sorted_edges),
        "warning": "No features or labels included; this is unverified message-passing topology only."
    }
    # Integrity checks independent of chosen review state.
    validation = validate_clean_graph(clean)
    if not validation["passed"]:
        raise ValueError("Clean graph validation failed: " + "; ".join(validation["errors"]))
    counts = Counter(str(e.get("edge_kind")) for _, _, e in clean.edges(data=True))
    report = {
        "claim_count": sum(a.get("node_type") == "unverified_claim" for _, a in clean.nodes(data=True)),
        "mention_decisions": dict(decisions),
        "relation_decisions": dict(relation_decisions),
        "canonical_qids": len({v for v in resolved_qids.values()}),
        "accepted_relation_edges_in_clean": counts["claim_assertion"],
        "accepted_relations_excluded_due_to_endpoints": skipped_relations,
        "raw_nodes": audit.number_of_nodes(), "raw_edges": audit.number_of_edges(),
        "clean_nodes": clean.number_of_nodes(), "clean_edges": clean.number_of_edges(),
        "clean_edge_types": dict(counts),
        "gnn_message_edges": gnn.number_of_edges(),
    }
    return {
        "schema_version": "0.4.2", "source_schema": "0.4.1",
        "warning": "NO FACT VERIFICATION. All claim assertions remain unverified. No gold labels, gold evidence or justifications imported.",
        "report": report, "validation": validation,
        # Persist the EDITED review decisions (not the original snapshot).
        # Immutable metadata is copied from validated source records.
        "mention_reviews": [
            {
                **copy.deepcopy(expected[r["mention_key"]]),
                "decision": str(r["decision"]),
                "qid": str(r.get("qid", "") or "").strip().upper(),
                "canonical_label": str(r.get("canonical_label", "") or "").strip(),
            }
            for r in mention_rows
        ],
        "relation_reviews": [
            {**copy.deepcopy(by_key[r["relation_key"]]), "decision": str(r["decision"])}
            for r in relation_rows
        ],
        "audit_graph": json_graph.node_link_data(audit, edges="links"),
        "clean_graph": json_graph.node_link_data(clean, edges="links"),
        "gnn_graph": json_graph.node_link_data(gnn, edges="links"),
        "gnn_ready": gnn_ready,
    }


def validate_clean_graph(graph: nx.MultiDiGraph) -> dict[str, Any]:
    errors = []
    if not isinstance(graph, nx.MultiDiGraph) or not graph.is_directed():
        errors.append("Expected a directed MultiDiGraph")
        return {"passed": False, "errors": errors}
    claim_ids = {n for n, a in graph.nodes(data=True) if a.get("node_type") == "unverified_claim"}
    for src, dst, e in graph.edges(data=True):
        kind = e.get("edge_kind")
        if e.get("truth_status") != "unverified":
            errors.append("Detected non-unverified edge")
        if kind == "claim_mention":
            if src not in claim_ids or e.get("mention_review") != "accepted_mention":
                errors.append("Claim mention edge is unapproved or reversed")
        elif kind == "claim_assertion":
            if e.get("extraction_review") != "accepted_extraction":
                errors.append("Unapproved candidate relation in clean graph")
            prov = e.get("provenance_claim_id")
            if prov not in claim_ids:
                errors.append("Assertion missing claim provenance")
        else:
            errors.append("Unrecognized edge kind in clean graph")
    for _, a in graph.nodes(data=True):
        if a.get("node_type") in {"unresolved_mention", "reviewed_entity"}:
            errors.append("Raw-only node type found in clean graph")
        if PROHIBITED_FIELDS & a.keys():
            errors.append("Gold labels or justifications in clean graph")
    return {"passed": not errors, "errors": sorted(set(errors)),
            "directed": graph.is_directed(), "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges()}
