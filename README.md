# VeriGraph AI v0.4.2 — Reviewed, Canonical & GNN-Ready Graphs

**New in v0.4.2:** an additional Streamlit tab separates the complete audit graph from a human-reviewed clean graph, supports manual canonical QIDs, accepts/rejects semantic extraction independently, and exports symmetric message-passing adjacency for future GNN experiments. **Nothing in this app fact-checks claims yet.**

Start (within `verigraph_ai_starter`):

```cmd
conda activate kg_e
python -m pytest -q
python -m streamlit run app.py
```

Open tab **6. Reviewed & Clean KG (v0.4.2)**. Saved input: `outputs/batch_kg_v041.json`. The original v0.1–v0.4.1 features remain available. For offline processing, run `python scripts/build_clean_kg.py`. Full instructions: [docs/UPGRADE_V042.md](docs/UPGRADE_V042.md).

---

# VeriGraph AI v0.4.1 — Multilingual Knowledge Graph Quality & Enrichment

**NEW: [Complete v0.4.1 quickstart, precautions and CLI](docs/UPGRADE_V041.md)**

Continue using the original v0.1–v0.4 tabs. The **fifth tab** loads a saved v0.4 JSON graph, flags low-quality NER output, accepts manual entity reviews and Wikidata QIDs, and proposes semantic relations using offline rules or an optional local Ollama LLM. All claim assertions remain **unverified**.

```
conda activate kg_e
python -m pytest -q
python -m streamlit run app.py
```

You can use the bundled `data/examples/averitec_5_v04.json` immediately, or upload your own v0.4 export. No need to rerun GLiNER.

---

> **Latest: VeriGraph AI v0.4** — batch CSV/JSONL import, GLiNER multilingual NER, unverified candidate relation KG, manually reviewed QID mapping, and JSON export. **Not a fake-news classifier or a truth verifier.** See [v0.4 upgrade guide](docs/UPGRADE_V04.md).

# VeriGraph AI — Multilingual Knowledge Graph (MVP Starter)

A portfolio project scaffold inspired by a 4-month multilingual KG fake-news detection roadmap, **adapted for a quick, testable engineering portfolio**.

**Three languages:** Indonesian (`id`), English (`en`), Chinese (`zh`).

> **Current scope / honesty statement:** v0.1 is an interactive, curated knowledge-graph demonstration using a narrow rule-based parser for capital-of claims. **It is not a trained fake-news classifier, not an LLM pipeline, and does not prove the truth of arbitrary claims.** `data/demo_claims.csv` is an illustrative fixture, NOT a research evaluation dataset.

## Quick start (Windows PowerShell, new environment)

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.build_graph
python -m src.evaluate_demo
python -m pytest -q
streamlit run app.py
```

If PowerShell activation is blocked, use Command Prompt (`.venv\Scripts\activate.bat`) or invoke `.venv\Scripts\python.exe` directly.

macOS/Linux: `source .venv/bin/activate`.

## What's working in v0.1?

- Stable Wikidata-aligned IDs shared across three languages (e.g. Japan = `Q17`)
- Curated node and relationship data with reference links
- A multilingual alias dictionary for a *single* simple claim pattern
- Demonstration graph exported as `GraphML` and JSON
- Interactive Streamlit visualization and inspection of evidence references
- Tests for 18 examples and safe "insufficient evidence" responses

## What was *not* implemented in v0.1? (historical baseline)

- General purpose multilingual NER and entity linking
- Automated fact extraction with an LLM
- Text-only machine-learning classifier
- GNN / text-graph fusion
- Full evidence provenance and date-aware retrieval
- Scientific multilingual / cross-lingual evaluation

## Original v0.1 project structure (historical; see v0.4 guide for current layout)

```text
verigraph_ai_starter/
├── app.py                       # interactive prototype
├── requirements.txt
├── data/
│   ├── entities.csv             # shared entity IDs and aliases
│   ├── relations.csv            # curated relations + references
│   └── demo_claims.csv          # illustrative examples, NOT benchmark
├── src/
│   ├── core.py                  # load, link, inspect demo relation
│   ├── build_graph.py           # GraphML and JSON export
│   └── evaluate_demo.py        # smoke fixture consistency only
├── tests/test_core.py
├── docs/
│   ├── ROADMAP_4_WEEKS.md
│   └── DATA_CONTRACT.md
└── outputs/                      # generated, gitignored
```

## Key design rules

1. **No data leakage:** keep gold verdicts and fact-check explanations out of retrieval knowledge accessible during test inference.
2. **No automatic trust:** triples mined from unverified claims are *assertions*, not trusted facts.
3. **Track sources:** attach provenance and, where relevant, dates to factual assertions.
4. **Open-world reasoning:** missing information cannot be treated as fake.
5. **Honest measurement:** rule-based demonstration accuracy on handcrafted fixtures is **not** a model metric.

## Next steps

See [4-week development plan](docs/ROADMAP_4_WEEKS.md).

## Source links

- Wikidata action API: https://www.wikidata.org/wiki/Wikidata:REST_API/Comparison/en
- Wikidata item identifiers in `data/entities.csv`
- Indonesian fact-check corpus candidate: https://huggingface.co/datasets/nlp-brin-id/fakenews-mafindo (has documented sampling limitations; audit before use)
- Multilingual spaCy NER candidate: https://spacy.io/models/xx (needs per-language quality checks)

## License

This starter code is a reusable portfolio scaffold. Validate third-party dataset/model licenses independently before distribution.

## Update v0.2: multilingual NER workbench

This ZIP also includes an **optional** NER + Wikidata workbench in Streamlit tab 2.
Install `requirements-ner.txt` before using tab 2, then see `docs/UPGRADE_V02.md`.
It is separate from the existing curated capital-of graph; new claim mentions
are unverified and **cannot** be interpreted as supporting or refuting a news claim.

## NEW — v0.4 Real Dataset Batch KG

Built as an extension of v0.3; see [docs/UPGRADE_V04.md](docs/UPGRADE_V04.md).

- Streamlit tab **4. Batch Dataset KG (v0.4)** imports CSV / JSONL / JSON,
  optionally runs GLiNER NER, uses rule-based candidate relation extraction,
  and exports an **unverified** multi-claim graph.
- Put your real datasets inside `data/datasets/averitec/` and `data/datasets/cfever/`
  below this project root.
- Start with just 5 claims on CPU. For CLI use `python scripts/run_batch.py --help`.
- **Do not claim fake-news detection yet**: v0.4 constructs candidate graphs,
  not fact verification or a trained classifier. Gold labels are excluded.
