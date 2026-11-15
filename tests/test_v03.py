"""v0.3 unit tests without remote model downloads."""
from networkx.readwrite import json_graph

from src.relation_extraction import extract_relation_candidates, build_assertion_graph


def make_entities(text: str, names_labels):
    """Build deterministic span-aligned fake NER output."""
    entities = []
    for name, label in names_labels:
        start = text.index(name)
        entities.append({"text": name, "start": start, "end": start + len(name), "label": label, "score": 0.98})
    return sorted(entities, key=lambda e: e["start"])


def test_indonesian_work_visit_extracts_one_unverified_visit():
    text = "Prabowo Subianto melakukan kunjungan kerja di Jakarta."
    ents = make_entities(text, [("Prabowo Subianto", "person"), ("kunjungan kerja", "event"), ("Jakarta", "location")])
    rels = extract_relation_candidates(text, "id", ents)
    assert len(rels) == 1
    assert (rels[0]["subject"], rels[0]["predicate"], rels[0]["object"]) == ("Prabowo Subianto", "VISITS", "Jakarta")
    assert rels[0]["truth_status"] == "unverified"
    assert text[rels[0]["trigger_start"]:rels[0]["trigger_end"]] == "melakukan kunjungan kerja"


def test_english_visit_and_meeting_extraction():
    text = "Barack Obama visited Paris and met Angela Merkel."
    ents = make_entities(text, [("Barack Obama", "person"), ("Paris", "location"), ("Angela Merkel", "person")])
    rels = extract_relation_candidates(text, "en", ents)
    assert [(x["predicate"], x["object"]) for x in rels] == [("VISITS", "Paris"), ("MEETS", "Angela Merkel")]


def test_chinese_place_before_trigger():
    text = "习近平在北京出席会议。"
    ents = make_entities(text, [("习近平", "person"), ("北京", "location")])
    rels = extract_relation_candidates(text, "zh", ents)
    assert len(rels) == 1
    assert rels[0]["predicate"] == "ATTENDS_IN"
    assert rels[0]["object"] == "北京"


def test_no_trigger_abstains_from_relation_generation():
    text = "Prabowo Subianto berada di Jakarta."
    ents = make_entities(text, [("Prabowo Subianto", "person"), ("Jakarta", "location")])
    assert extract_relation_candidates(text, "id", ents) == []


def test_graph_keeps_linking_distinct_from_truth_status():
    text = "Prabowo Subianto melakukan kunjungan kerja di Jakarta."
    ents = make_entities(text, [("Prabowo Subianto", "person"), ("Jakarta", "location")])
    candidates = extract_relation_candidates(text, "id", ents)
    links = {
        0: {"id": "Q2107268", "label": "Prabowo Subianto", "method": "manual"},
        1: {"id": "Q3630", "label": "Jakarta", "method": "manual"},
    }
    graph = build_assertion_graph(text, "id", ents, links, candidates, {0: "approved_extraction"})
    rel_edges = [(a, b, d) for a, b, d in graph.edges(data=True) if d.get("edge_kind") == "claim_assertion"]
    assert len(rel_edges) == 1
    subj, obj, edge = rel_edges[0]
    assert (subj, obj) == ("Q2107268", "Q3630")
    assert edge["truth_status"] == "unverified"
    assert edge["extraction_review"] == "approved_extraction"
    assert edge["provenance_claim_id"].startswith("claim:")
    payload = json_graph.node_link_data(graph, edges="links")
    assert payload["links"]


def test_rejected_relation_not_exported():
    text = "Barack Obama visited Paris."
    ents = make_entities(text, [("Barack Obama", "person"), ("Paris", "location")])
    candidates = extract_relation_candidates(text, "en", ents)
    graph = build_assertion_graph(text, "en", ents, {}, candidates, {0: "rejected"})
    assert all(d["relation"] == "MENTIONS" for _, _, d in graph.edges(data=True))


def test_input_language_validation():
    try:
        extract_relation_candidates("text", "fr", [])
        assert False
    except ValueError:
        pass
