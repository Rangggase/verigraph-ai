# CLAUDE.md — VeriGraph AI

Portfolio project (Sistem Informasi): a **multilingual (id/en/zh) claim-mention knowledge-graph workbench** with human review, plus claim-only ML baselines. It does **not** verify truth; all claim-derived edges are `truth_status: unverified`.

## Stack
Python 3.11+ · Streamlit UI · NetworkX · pandas · scikit-learn/joblib (ML tab) · optional GLiNER + requests (NER, Wikidata) · pytest · ruff. No database and no user accounts: state is versioned JSON files in `outputs/` and `data/`.

## Commands
```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements-dev.txt                 # core + ml + pytest/ruff
python -m pytest -q                                 # must be 100% green
ruff check .
streamlit run app.py
python scripts/kg_coverage_report.py outputs/batch_kg_v042.json
```
Optional extras: `requirements-ner.txt` (GLiNER/Wikidata), `requirements-dl.txt` (transformers).

## Architecture (`src/`)
- `core.py`, `build_graph.py` — curated capital-of demo KG (v0.1).
- `ner_pipeline.py`, `entity_linking.py` — GLiNER NER + Wikidata *candidates* (never auto-linked).
- `relation_extraction.py`, `assertion_graph.py` — rule-based candidate relations → unverified assertion graph.
- `batch_pipeline.py` — CSV/JSON/JSONL import (size/row caps, test splits blocked) → batch KG.
- `kg_enrichment.py`, `reviewed_graph.py` — quality flags, human review, canonical QIDs, audit/clean/GNN-ready graphs (fail-closed validation).
- `ner_annotation*.py`, `ner_ui_v2_helpers.py`, `easy_home_ui.py` — human NER review of the 50-claim CFEVER pilot (atomic saves, source digest).
- `classification_*.py`, `model_integrity.py` — claim-only baselines; model files must carry a SHA-256 sidecar.
- `errors.py` — `friendly_error()`; every UI `except` must use it. `kg_coverage.py` — review/QID coverage metrics.
- `app.py` — Streamlit entry (8 tabs; "Mudah" mode shows the guided reviewer). Design tokens: `src/theme.py`.

## Rules
1. Never present output as fact verification. Keep the "unverified" warnings.
2. Never feed gold labels, evidence or justifications into graphs or features (`PROHIBITED_FIELDS`).
3. Never `joblib.load` anything except via `load_trusted_model()`; never accept uploaded pickles.
4. No secrets in code. Config via env vars (`.env.example`). `.env`, `.venv`, `outputs/`, `data/datasets/` are git-ignored.
5. User-visible text is Bahasa Indonesia; code, commits and docs headings in English. Commits: `feat:`/`fix:`/`refactor:`/`docs:`/`test:`/`chore:`.
6. Every change ships with a test; `pytest` and `ruff check` must pass before commit.
7. Local datasets (CFEVER, AVeriTeC) are not redistributed; check their licenses.
