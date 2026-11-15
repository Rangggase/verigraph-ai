"""Build v0.4.2 using the existing v0.4.1 human decisions (no internet/NER)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Allow `python scripts/build_clean_kg.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reviewed_graph import build_reviewed_graph, get_mention_rows, get_relation_rows, load_v041


def main():
    cli = argparse.ArgumentParser(description="Build audit, reviewed clean, and bidirectional GNN graphs")
    cli.add_argument("--input", default="outputs/batch_kg_v041.json")
    cli.add_argument("--output", default="outputs/batch_kg_v042.json")
    options = cli.parse_args()
    src = Path(options.input)
    raw, graph = load_v041(src.read_bytes())
    # CLI respects existing review decisions: does NOT silently accept pending.
    result = build_reviewed_graph(raw, get_mention_rows(raw, graph), get_relation_rows(graph))
    out = Path(options.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "report": result["report"], "validation": result["validation"]}, indent=2))


if __name__ == "__main__":
    main()
