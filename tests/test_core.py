from src.core import make_graph, parse_demo_claim, read_csv


def test_graph_size():
    graph = make_graph()
    assert graph.number_of_nodes() == 6
    assert graph.number_of_edges() == 3


def test_language_examples():
    graph = make_graph()
    for example in read_csv("demo_claims.csv"):
        result = parse_demo_claim(example["text"], example["language"], graph)
        assert result["status"] == example["expected_relation"], example["claim_id"]


def test_unknown_data_does_not_become_fake():
    graph = make_graph()
    assert parse_demo_claim("Mars is the capital of France.", "en", graph)["status"] == "insufficient_evidence"


def test_unparsed_data_does_not_become_fake():
    graph = make_graph()
    assert parse_demo_claim("A completely new claim.", "en", graph)["status"] == "unparsed"
