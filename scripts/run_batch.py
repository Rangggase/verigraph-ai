"""Run v0.4 from CLI in the repository root.

Example:
python scripts/run_batch.py --input data/datasets/averitec/averitec_pilot_100.csv --language en --limit 5 --with-ner
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.batch_pipeline import (load_records, normalize_records, guess_column,
                                approved_links_from_csv, build_batch_graph, export_graph_bundle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch claim -> unverified candidate KG")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--claim-field", default="")
    parser.add_argument("--id-field", default="")
    parser.add_argument("--language-field", default="")
    parser.add_argument("--language", choices=["id", "en", "zh"], default="en")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=.35)
    parser.add_argument("--approved-links", type=Path)
    parser.add_argument("--with-ner", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "batch_kg_v04.json")
    args = parser.parse_args()
    raw = load_records(args.input.read_bytes(), args.input.name)
    if not raw:
        parser.error("Input dataset has no rows")
    cols = list(raw[0].keys())
    claim_field = args.claim_field or guess_column(cols, "claim")
    if not claim_field or claim_field not in cols:
        parser.error(f"Could not find claim column. Available: {cols}")
    id_field = args.id_field or guess_column(cols, "id")
    lang_field = args.language_field or guess_column(cols, "language")
    if id_field and id_field not in cols or lang_field and lang_field not in cols:
        parser.error("Invalid id or language field")
    dataset = args.dataset or args.input.stem
    normalized, rejected = normalize_records(
        raw, claim_field=claim_field, id_field=id_field,
        language_field=lang_field, default_language=args.language,
        split=args.split, limit=args.limit, dataset=dataset,
    )
    links = (approved_links_from_csv(args.approved_links.read_bytes()) if args.approved_links else {})
    model = None
    if args.with_ner:
        from src.ner_pipeline import load_ner_model
        print("Loading GLiNER (may download model on first run)...", flush=True)
        model = load_ner_model()
    graph, summary = build_batch_graph(normalized, model=model, threshold=args.threshold, approved_links=links)
    output = export_graph_bundle(graph, summary)
    output["rejected_rows"] = rejected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output["stats"], indent=2))
    print(f"Rejected rows: {len(rejected)}")
    print(f"Output: {args.output}")
    if not args.with_ner:
        print("CLAIM-ONLY DRY RUN: --with-ner is required for entity/relation graph")


if __name__ == "__main__":
    main()
