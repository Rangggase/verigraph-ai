"""v0.4 tests: no network, no real model weights needed."""
import json
import pytest

from src.batch_pipeline import (
    load_records, normalize_records, approved_links_from_csv,
    build_batch_graph, export_graph_bundle, guess_column,
)


class FakeNER:
    """Deterministic test-double with GLiNER's public predict_entities signature."""
    def predict_entities(self, text, labels, threshold):
        found = []
        for term, label in [("Prabowo Subianto", "person"), ("Jakarta", "location"),
                            ("Barack Obama", "person"), ("Paris", "location")]:
            idx = text.find(term)
            if idx >= 0:
                found.append({"text": term, "label": label, "start": idx,
                              "end": idx + len(term), "score": .99})
        return found


def sample():
    return [
        {"claim_id": "1", "language": "id",
         "claim": "Prabowo Subianto melakukan kunjungan kerja di Jakarta.",
         "label": "FAKE", "justification": "DO NOT INCLUDE ME IN GRAPH"},
        {"claim_id": "2", "language": "id",
         "claim": "Prabowo Subianto mengunjungi Jakarta.",
         "label": "REAL", "justification": "PRIVATE GOLD EVIDENCE"},
        {"claim_id": "3", "language": "en", "claim": "Barack Obama visited Paris."},
    ]


def normalized():
    return normalize_records(sample(), claim_field="claim", id_field="claim_id",
                             language_field="language", dataset="demo", default_language="en", limit=3)


def test_import_csv_json_jsonl_and_autodetect():
    assert len(load_records(b'claim,label\n"Hello",fake\n', "input.csv")) == 1
    assert len(load_records(b'{"claim":"Hello"}\n', "input.jsonl")) == 1
    assert len(load_records(b'{"records":[{"claim":"Hello"}]}', "input.json")) == 1
    assert guess_column(["something", "claim"], "claim") == "claim"
    with pytest.raises(ValueError):
        load_records(b'[]', "input.exe")


def test_normalization_excludes_gold_labels_and_evidence():
    records, rejected = normalized()
    assert len(records) == 3
    assert not rejected
    assert all("label" not in r and "justification" not in r for r in records)
    assert len({r["claim_id"] for r in records}) == 3
    assert all(r["claim_id"].startswith("claim:") for r in records)


def test_batch_merge_only_reviewed_ids_and_keep_truth_unverified():
    approved = approved_links_from_csv(
        "language,surface,qid,label\nid,Prabowo Subianto,Q2107268,Prabowo Subianto\nid,Jakarta,Q3630,Jakarta\n".encode()
    )
    records, _ = normalized()
    graph, summary = build_batch_graph(records, model=FakeNER(), approved_links=approved)
    assert len(summary) == 3
    assert "Q2107268" in graph and "Q3630" in graph
    assert sum(1 for n in graph if n == "Q3630") == 1
    assert any(a.get("edge_kind") == "claim_assertion" for _, _, a in graph.edges(data=True))
    assert all(a.get("truth_status") == "unverified" for _, _, a in graph.edges(data=True))
    for _, attrs in graph.nodes(data=True):
        assert "justification" not in attrs and "gold_label" not in attrs
        assert "PRIVATE GOLD EVIDENCE" not in str(attrs)
    exported = export_graph_bundle(graph, summary)
    assert exported["stats"]["claims"] == 3
    assert exported["stats"]["edges"] >= 3
    assert "links" in exported["graph"]
    assert "PRIVATE GOLD EVIDENCE" not in json.dumps(exported)


def test_without_reviewed_qids_does_not_auto_merge_mentions():
    records, _ = normalized()
    graph, _ = build_batch_graph(records, model=FakeNER())
    assert "Q2107268" not in graph
    prabowo_nodes = [n for n, a in graph.nodes(data=True)
                     if a.get("label") == "Prabowo Subianto"]
    assert len(prabowo_nodes) == 2


def test_claim_only_mode_works_offline_and_explicit():
    records, _ = normalized()
    graph, summary = build_batch_graph(records, model=None)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 0
    assert all(x["entities"] == 0 for x in summary)


def test_bad_split_and_overlong_text_rejected():
    with pytest.raises(ValueError):
        normalize_records(sample(), claim_field="claim", dataset="x", default_language="en", split="test")
    rows, rejected = normalize_records([{"claim": "a" * 1501}, {"claim": "good"}],
                                       claim_field="claim", dataset="x", default_language="en")
    assert len(rows) == 1 and len(rejected) == 1


def test_batch_failure_keeps_claim_and_reports_error():
    class BrokenNER:
        def predict_entities(self, *_args, **_kwargs):
            raise RuntimeError("test error")
    records, _ = normalized()
    graph, summary = build_batch_graph(records, model=BrokenNER())
    assert graph.number_of_nodes() == 3
    assert len(summary) == 3
    assert all("test error" in r["error"] for r in summary)


def test_conflicting_manual_links_rejected():
    content = b'language,surface,qid,label\nid,Jakarta,Q3630,Jakarta\nid,Jakarta,Q1,Jakarta\n'
    with pytest.raises(ValueError):
        approved_links_from_csv(content)


def test_heldout_file_cannot_be_imported_as_train():
    with pytest.raises(ValueError, match="Held-out"):
        load_records(b"claim\nHi\n", "test.csv")


def test_large_jsonl_import_uses_head_preview_instead_of_failing():
    source = ("{\"claim\":\"hello\"}\n" * 1100).encode()
    assert len(load_records(source, "train.jsonl")) == 1000
