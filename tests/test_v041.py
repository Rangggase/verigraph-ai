"""v0.4.1 tests using the user's exact AVeriTeC pilot export (no network/models)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.kg_enrichment import (
    load_v04_bundle, review_rows, enrich_bundle, validate_model_relations,
    ollama_relations,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "data" / "examples" / "averitec_5_v04.json"


def make_input():
    return load_v04_bundle(EXAMPLE.read_bytes())


def test_import_and_review_real_batch():
    bundle, graph = make_input()
    assert bundle["stats"]["claims"] == 5
    assert graph.number_of_nodes() == 23
    assert len(review_rows(graph)) == 18
    flags = {item["surface"]: item["warnings"] for item in review_rows(graph)}
    assert "numeric_as_event" in flags["20%"]
    assert "abstract_as_location" in flags["American history"]
    assert "leading_preposition" in flags["In Nigeria"]


def test_rule_enrichment_creates_provenance_and_does_not_claim_truth():
    bundle, graph = make_input()
    out = enrich_bundle(bundle, review_rows(graph))
    stats = out["quality_report"]
    assert stats["candidate_relations"] == 2
    assert stats["edge_types"]["claim_assertion"] == 2
    relations = [a for a in out["graph"]["links"] if a.get("edge_kind") == "claim_assertion"]
    assert {a["relation"] for a in relations} == {"BOARD_MEMBER_OF", "PLEDGED_TO"}
    assert all(a["truth_status"] == "unverified" and a["extraction_review"] == "pending" for a in relations)
    assert all(a["provenance_claim_id"].startswith("claim:") and a["evidence_text"] for a in relations)
    exported = json.dumps(out)
    assert "No former experience stated" not in exported
    assert "label_original" not in exported
    assert "justification" not in exported.lower()  # no gold justification fields/text


def test_reviewed_qid_resolves_two_instances_to_one_entity():
    bundle, graph = make_input()
    rows = review_rows(graph)
    # Real v0.4 example only contains one Ukraine mention; simulate same QID for
    # two distinct mentions to demonstrate explicit cross-claim canonical linking.
    for row in rows:
        if row["surface"] in {"Hunter Biden", "Biden"}:
            row["decision"] = "accepted_mention"
            row["qid"] = "Q999999"
            row["canonical_label"] = "Example canonical person (reviewed)"
    out = enrich_bundle(bundle, rows, extractor="none")
    assert out["quality_report"]["reviewed_qid_count"] == 1
    nodes = [n for n in out["graph"]["nodes"] if n["id"] == "Q999999"]
    assert len(nodes) == 1
    resolutions = [a for a in out["graph"]["links"] if a.get("edge_kind") == "entity_resolution"]
    assert len(resolutions) == 2
    assert all(a["target"] == "Q999999" for a in resolutions)


def test_rejected_or_flagged_mentions_not_used_without_acceptance():
    bundle, graph = make_input()
    rows = review_rows(graph)
    out = enrich_bundle(bundle, rows)
    assert "DELIVERED" not in out["quality_report"]["candidate_relation_types"]
    # Approving a flagged span is allowed for explicit review, not automatic.
    for row in rows:
        if row["surface"] == "largest tax cuts":
            row["decision"] = "accepted_mention"
    out2 = enrich_bundle(bundle, rows)
    assert out2["quality_report"]["candidate_relation_types"]["DELIVERED"] == 1
    # Reject an entity: no relation can use it.
    for row in rows:
        if row["surface"] == "Burisma":
            row["decision"] = "rejected_mention"
    out3 = enrich_bundle(bundle, rows)
    assert "BOARD_MEMBER_OF" not in out3["quality_report"]["candidate_relation_types"]


def test_unreviewed_qid_must_not_be_assigned():
    bundle, graph = make_input()
    rows = review_rows(graph)
    rows[0]["qid"] = "Q3630"
    with pytest.raises(ValueError, match="QIDs require"):
        enrich_bundle(bundle, rows)
    rows[0]["decision"] = "accepted_mention"
    rows[0]["qid"] = "P123"
    with pytest.raises(ValueError, match="QIDs require"):
        enrich_bundle(bundle, rows)


def test_tampered_review_table_fails_closed():
    bundle, graph = make_input()
    rows = review_rows(graph)
    with pytest.raises(ValueError, match="exactly"):
        enrich_bundle(bundle, rows[:-1])
    with pytest.raises(ValueError, match="Invalid review"):
        enrich_bundle(bundle, [{**rows[0], "decision": "verified_fact"}, *rows[1:]])


def test_llm_output_must_be_grounded_in_source_text_and_mention_indices():
    claim = "Hunter Biden joined the board of Burisma."
    mentions = [{"surface": "Hunter Biden"}, {"surface": "Burisma"}]
    raw = {"relations": [
        {"subject_index": 0, "object_index": 1, "predicate": "BOARD_MEMBER_OF", "evidence_quote": "joined the board of Burisma"},
        {"subject_index": 0, "object_index": 3, "predicate": "FOUNDED", "evidence_quote": "joined the board of Burisma"},
        {"subject_index": 0, "object_index": 1, "predicate": "OWNS", "evidence_quote": "He owns Burisma"},
        {"subject_index": True, "object_index": 1, "predicate": "OWNS", "evidence_quote": "joined the board of Burisma"},
    ]}
    actual = validate_model_relations(raw, claim, mentions)
    assert len(actual) == 1
    assert actual[0]["predicate"] == "BOARD_MEMBER_OF"
    with pytest.raises(ValueError):
        validate_model_relations({"nonsense": []}, claim, mentions)


def test_ollama_stays_loopback_and_does_not_call_network_for_invalid_target():
    with pytest.raises(ValueError, match="Only local"):
        ollama_relations("some claim", "en", [], base_url="https://attacker.com")


def test_optional_fake_model_injected_into_enrichment():
    bundle, graph = make_input()
    rows = review_rows(graph)
    calls = []

    def fake_ollama(claim, lang, mentions, model):
        calls.append((lang, model))
        if len(mentions) < 2:
            return []
        return [{"subject_index": 0, "object_index": 1, "predicate": "ASSERTS_X",
                 "evidence_text": claim, "method": "fake_test"}]

    out = enrich_bundle(bundle, rows, extractor="ollama", ollama_fn=fake_ollama)
    assert len(calls) == 3  # two claims have fewer than two safe mentions
    assert out["quality_report"]["candidate_relations"] == 3
    assert all(a.get("truth_status") == "unverified" for a in out["graph"]["links"])


def test_corrupted_bundle_rejected():
    with pytest.raises(ValueError):
        load_v04_bundle({"schema_version": "0.3", "graph": {}})
    with pytest.raises(ValueError):
        load_v04_bundle({"schema_version": "0.4", "graph": {"nodes": [], "links": [{"source": "a", "target": "b", "justification": "secret"}]}})
