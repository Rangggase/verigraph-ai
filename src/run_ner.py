"""Optional CLI for testing multilingual NER without starting Streamlit.

Example:
    python -m src.run_ner --language id --text "Prabowo Subianto berada di Jakarta."
"""
from __future__ import annotations
import argparse
import json

from src.ner_pipeline import load_ner_model, extract_entities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["id", "en", "zh"], default="id")
    parser.add_argument("--text", required=True)
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()
    model = load_ner_model()
    entities = extract_entities(args.text, args.language, model, args.threshold)
    print(json.dumps(entities, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
