"""CLI: enrich a v0.4 exported graph without re-running GLiNER.

Usage: python -m scripts.enrich_existing_batch --input outputs/batch_kg_v04.json
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from src.kg_enrichment import load_v04_bundle, review_rows, enrich_bundle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="outputs/batch_kg_v04.json")
    p.add_argument("--output", default="outputs/batch_kg_v041.json")
    p.add_argument("--template", default="outputs/mention_review_template.csv")
    p.add_argument("--review-csv", default=None)
    p.add_argument("--extractor", default="rules", choices=["rules", "none", "ollama"])
    p.add_argument("--ollama-model", default="qwen2.5:3b")
    args = p.parse_args()
    raw = Path(args.input).read_bytes()
    bundle, graph = load_v04_bundle(raw)
    rows = review_rows(graph)
    template = Path(args.template)
    template.parent.mkdir(parents=True, exist_ok=True)
    with template.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["mention_key","decision","qid","canonical_label"])
        writer.writeheader()
        writer.writerows(rows)
    if args.review_csv:
        with Path(args.review_csv).open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    enriched = enrich_bundle(bundle, rows, extractor=args.extractor, model_name=args.ollama_model)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(enriched["quality_report"], indent=2, ensure_ascii=False))
    print("Saved:", out)
    print("Review template:", template)


if __name__ == "__main__":
    main()
