
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/cfever_zh_pilot_v061.json"
OUTPUT = ROOT / "outputs/cfever_zh_ner_review_v061.csv"

data = json.loads(SOURCE.read_text(encoding="utf-8"))

nodes = {
    node["id"]: node
    for node in data["graph"]["nodes"]
}

runs = data["runs"]
external_ids = {
    run["claim_id"]: run["external_id"]
    for run in runs
}

mentions = []
counts = Counter()

for edge in data["graph"]["links"]:
    if edge.get("edge_kind") != "claim_mention":
        continue

    claim_id = edge["source"]
    claim = nodes[claim_id]["label"]
    surface = edge["surface"]

    start = int(edge["start"])
    end = int(edge["end"])

    ratio = len(surface) / max(len(claim), 1)

    flags = []

    if ratio >= 0.50:
        flags.append("LONG_SPAN")

    if any(p in surface for p in ["，", "。", "；", "！", "？"]):
        flags.append("CONTAINS_PUNCTUATION")

    if claim[start:end] != surface:
        flags.append("OFFSET_MISMATCH")

    counts[claim_id] += 1

    mentions.append({
        "external_id": external_ids.get(claim_id, ""),
        "claim": claim,
        "entity": surface,
        "ner_label": edge.get("ner_label", ""),
        "score": edge.get("ner_score", ""),
        "span_ratio": round(ratio, 3),
        "quality_flags": ",".join(flags),
        "manual_decision": "",
        "review_note": "",
    })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

fields = list(mentions[0].keys()) if mentions else []

if fields:
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(mentions)

total_claims = len(runs)
covered = len(counts)
flagged = sum(bool(r["quality_flags"]) for r in mentions)

print("=== VERIGRAPH v0.6.1 NER QUALITY AUDIT ===")
print("Total claims:", total_claims)
print("Total mentions:", len(mentions))
print("Claims with mentions:", covered)
print("Claims without mentions:", total_claims - covered)
print("Claim coverage:", f"{covered / total_claims:.2%}")
print("Claims with >=2 mentions:",
      sum(n >= 2 for n in counts.values()))
print("Mentions flagged for review:", flagged)

print("\n=== TOP 10 LONGEST RELATIVE SPANS ===")

for row in sorted(
    mentions,
    key=lambda r: r["span_ratio"],
    reverse=True
)[:10]:
    print(
        row["external_id"],
        "| ratio:", row["span_ratio"],
        "| entity:", row["entity"],
        "| flags:", row["quality_flags"]
    )

print("\nReview CSV:", OUTPUT)
