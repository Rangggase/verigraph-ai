"""Smoke check on curated examples, NOT scientific accuracy or model evaluation."""
from collections import Counter
from src.core import read_csv, make_graph, parse_demo_claim


def main():
    graph = make_graph()
    items = read_csv("demo_claims.csv")
    results = []
    for item in items:
        prediction = parse_demo_claim(item["text"], item["language"], graph)
        results.append((item, prediction))
        print(f"{item['claim_id']}: {prediction['status']} | {item['text']}")
    n_pass = sum(pred["status"] == row["expected_relation"] for row, pred in results)
    print(f"Rule-based demo consistency: {n_pass}/{len(items)}")
    print("Count by language:", dict(Counter(r["language"] for r, _ in results)))
    print("NOTE: This is a fixture smoke check, NOT real fake-news detection performance.")
    if n_pass != len(items):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
