"""Regression tests for 0.4.2 using actual uploaded AVeriTeC pilot output."""
import copy
import json
from pathlib import Path

import networkx as nx
import pytest
from networkx.readwrite import json_graph

from src.reviewed_graph import (
    build_reviewed_graph, get_mention_rows, get_relation_rows,
    load_v041, validate_clean_graph,
)

INPUT = Path(__file__).resolve().parent / "fixtures" / "batch_kg_v041_v060_reference.json"


def prep():
    bundle, graph = load_v041(INPUT.read_bytes())
    return bundle, get_mention_rows(bundle, graph), get_relation_rows(graph)


def test_user_review_record_roundtrip_and_audit_graph():
    bundle, mentions, relations = prep()
    out = build_reviewed_graph(bundle, mentions, relations)
    assert out["schema_version"] == "0.4.2"
    assert out["report"]["mention_decisions"] == {
        "accepted_mention": 5, "pending": 11, "rejected_mention": 1,
    }
    assert out["report"]["raw_nodes"] == 22
    assert out["report"]["raw_edges"] == 18
    assert out["report"]["clean_nodes"] == 10  # 5 claims + 5 approved mentions
    assert out["report"]["clean_edges"] == 5
    assert out["report"]["canonical_qids"] == 0
    assert out["validation"]["passed"]
    assert not any(n["label"] == "Biden" for n in out["clean_graph"]["nodes"])
    assert any(n["label"] == "Biden" for n in out["audit_graph"]["nodes"])
    assert all(n.get("node_type") == "unverified_claim" or n.get("node_type") == "accepted_unresolved_mention"
               for n in out["clean_graph"]["nodes"])


def test_manual_qid_merges_distinct_mentions_across_claims():
    bundle, mentions, relations = prep()
    for r in mentions:
        if r["surface"] in {"Hunter Biden", "Biden"}:
            r.update(decision="accepted_mention", qid="Q999888", canonical_label="Reviewed same person")
    out = build_reviewed_graph(bundle, mentions, relations)
    canonical = [n for n in out["clean_graph"]["nodes"] if n["id"] == "Q999888"]
    assert len(canonical) == 1
    assert out["report"]["canonical_qids"] == 1
    assert sum(e.get("target") == "Q999888" and e.get("edge_kind") == "claim_mention"
               for e in out["clean_graph"]["links"]) == 2
    assert sum(n["id"] in {"Q999888", "mention:7346067d59ca44bf36fe:0", "mention:ee035d88ed70d338d8db:0"}
               for n in out["clean_graph"]["nodes"]) == 1


def test_accepted_relation_requires_accepted_endpoints():
    bundle, mentions, relations = prep()
    for relation in relations:
        relation["decision"] = "accepted_extraction"
    out = build_reviewed_graph(bundle, mentions, relations)
    assert out["report"]["accepted_relation_edges_in_clean"] == 1  # Hunter -> Burisma
    assert out["report"]["accepted_relations_excluded_due_to_endpoints"] == 0  # current input has one relation
    semantic = [e for e in out["clean_graph"]["links"] if e.get("edge_kind") == "claim_assertion"]
    assert len(semantic) == 1
    assert semantic[0]["relation"] == "BOARD_MEMBER_OF"
    assert semantic[0]["truth_status"] == "unverified"
    assert semantic[0]["extraction_review"] == "accepted_extraction"


def test_rejecting_subject_prevents_model_relation_even_if_relation_reviewed():
    bundle, mentions, relations = prep()
    for r in mentions:
        if r["surface"] == "Hunter Biden":
            r["decision"] = "rejected_mention"
    for rel in relations:
        rel["decision"] = "accepted_extraction"
    out = build_reviewed_graph(bundle, mentions, relations)
    assert out["report"]["accepted_relation_edges_in_clean"] == 0


def test_pending_or_rejected_relations_not_enter_clean():
    bundle, mentions, relations = prep()
    for rel in relations:
        rel["decision"] = "rejected_extraction"
    out = build_reviewed_graph(bundle, mentions, relations)
    assert out["report"]["accepted_relation_edges_in_clean"] == 0
    assert out["report"]["relation_decisions"]["rejected_extraction"] == len(relations)


def test_malformed_qid_rejected_and_unreviewed_qid_rejected():
    bundle, mentions, relations = prep()
    for r in mentions:
        if r["surface"] == "Ukraine":
            r["qid"] = "Q212" if r["decision"] == "pending" else "P123"
    with pytest.raises(ValueError, match="QID"):
        build_reviewed_graph(bundle, mentions, relations)
    for r in mentions:
        if r["surface"] == "largest tax cuts":
            r["qid"] = "Q212"
    with pytest.raises(ValueError, match="QID"):
        build_reviewed_graph(bundle, mentions, relations)


def test_missing_or_duplicate_relation_rows_rejected():
    bundle, mentions, relations = prep()
    with pytest.raises(ValueError, match="Relation review rows"):
        build_reviewed_graph(bundle, mentions, relations[:-1])
    with pytest.raises(ValueError, match="Relation review rows"):
        build_reviewed_graph(bundle, mentions, [relations[0], relations[0]])


def test_tampered_relation_predicate_or_mention_label_rejected():
    bundle, mentions, relations = prep()
    relations[0]["predicate"] = "VERIFIED_BY"
    with pytest.raises(ValueError, match="Immutable relation"):
        build_reviewed_graph(bundle, mentions, relations)
    _, mentions, relations = prep()
    mentions[0]["surface"] = "Fake identity"
    with pytest.raises(ValueError, match="Immutable mention"):
        build_reviewed_graph(bundle, mentions, relations)


def test_orphan_edge_is_rejected_before_networkx_auto_adds_node():
    bundle, _, _ = prep()
    altered = copy.deepcopy(bundle)
    altered["graph"]["links"][0]["target"] = "not-a-real-node"
    with pytest.raises(ValueError, match="missing node"):
        load_v041(altered)


def test_invalid_orientation_fails_clean_graph_integrity_check():
    graph = nx.MultiDiGraph()
    graph.add_node("c", node_type="unverified_claim", truth_status="unverified")
    graph.add_node("m", node_type="accepted_unresolved_mention")
    graph.add_edge("m", "c", edge_kind="claim_mention", mention_review="accepted_mention", truth_status="unverified")
    result = validate_clean_graph(graph)
    assert not result["passed"]
    assert any("reversed" in e for e in result["errors"])


def test_gold_label_injection_rejected():
    bundle, _, _ = prep()
    tampered = copy.deepcopy(bundle)
    tampered["graph"]["nodes"][0]["gold_label"] = "SUPPORTED"
    with pytest.raises(ValueError, match="Prohibited gold"):
        load_v041(tampered)


def test_serialization_has_no_gold_and_preserves_parallel_edges():
    bundle, mentions, relations = prep()
    out = build_reviewed_graph(bundle, mentions, relations)
    reload_graph = json_graph.node_link_graph(json.loads(json.dumps(out))["clean_graph"], edges="links", directed=True, multigraph=True)
    assert isinstance(reload_graph, nx.MultiDiGraph)
    assert reload_graph.number_of_edges() == out["report"]["clean_edges"]
    assert "evidence_json" not in json.dumps(out)
    assert "label_original" not in json.dumps(out)
    assert all("justification" not in node and "evidence_json" not in node for node in out["clean_graph"]["nodes"])
    assert all("justification" not in edge and "evidence_json" not in edge for edge in out["clean_graph"]["links"])


def test_source_bundle_is_not_mutated():
    bundle, mentions, relations = prep()
    baseline = json.dumps(bundle, sort_keys=True)
    build_reviewed_graph(bundle, mentions, relations)
    assert json.dumps(bundle, sort_keys=True) == baseline


def test_gnn_reverse_flow_reaches_claims_without_leaking_gold():
    bundle, mentions, relations = prep()
    out = build_reviewed_graph(bundle, mentions, relations)
    gnn = json_graph.node_link_graph(out["gnn_graph"], edges="links", directed=True, multigraph=True)
    assert gnn.number_of_edges() == 2 * out["report"]["clean_edges"]
    assert out["gnn_ready"]["num_edges"] == gnn.number_of_edges()
    assert len(out["gnn_ready"]["edge_index"]) == 2
    for src, dst, edata in gnn.edges(data=True):
        if edata["edge_kind"] != "inverse_message_passing":
            assert any(x["edge_kind"] == "inverse_message_passing" for x in gnn[dst][src].values())
    first_claim = next(n for n, a in gnn.nodes(data=True) if a.get("node_type") == "unverified_claim" and a.get("external_id") == "averitec_00000")
    assert gnn.in_degree(first_claim) == 3  # entity features could flow into claim


def test_review_gold_field_injection_rejected():
    bundle, mentions, relations = prep()
    mentions[0]["gold_label"] = "Supported"
    with pytest.raises(ValueError, match="gold"):
        build_reviewed_graph(bundle, mentions, relations)
