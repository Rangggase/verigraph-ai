# VeriGraph AI v0.4 — Batch Dataset Candidate KG

This is a **portfolio MVP**, not verified fact checking, LLM-based triple extraction,
or a trained graph fake-news classifier. Built directly upon v0.3.

## Features

- Adds **tab 4** to the existing v0.3 Streamlit UI.
- Imports local datasets or uploaded **CSV/JSONL/JSON**, shows a preview of **up to the first 1,000 rows**, and offers configurable claim, ID and language fields (up to 75 MB source file).
- Processes at most 100 rows per UI batch (CPU: **start with 5**), selected from the first 1,000 rows. Supports `id`, `en`, `zh`. This deterministic head sample is for a smoke test, **not** a representative or stratified research sample.
- Runs **GLiNER** (optional, first run downloads model), then the v0.3 **rule-based** candidate relation extraction.
- Distinguishes `MENTIONS` edges from **unverified** claim assertion edges.
- Merges entity nodes across claims **only for explicitly human-reviewed QIDs** from a mapping CSV.
- Saves output in `outputs/batch_kg_v04.json`, including provenance claim IDs, edge status, model scores and an audit summary.
- Includes command-line processing for repeatable batch runs and automated tests.

## Recommended directory layout

```text
verigraph_ai_starter/
├─ app.py
├─ data/
│  ├─ entities.csv, relations.csv, demo_claims.csv
│  └─ datasets/
│     ├─ averitec/averitec_pilot_100.csv
│     └─ cfever/train.jsonl, dev.jsonl, test.jsonl
├─ scripts/run_batch.py
├─ src/batch_pipeline.py
└─ outputs/batch_kg_v04.json  (generated)
```

**You already downloaded datasets?** Move/copy your `data/datasets/` folder inside
`verigraph_ai_starter/data/` without replacing the existing curated demo files.

## Windows CMD quick start

```cmd
conda activate kg_e
cd verigraph_ai_starter
python -m pip install -r requirements-ner.txt
python -m pytest -q
python -m streamlit run app.py
```

Open **4. Batch Dataset KG (v0.4)**. Select your AVeriTeC pilot CSV, confirm
claim column, use `en`, `train`, `limit=5`, enable NER, and click **Build batch candidate KG**.

Without downloading GLiNER, validate import + export first:

```cmd
python scripts/run_batch.py --input data/datasets/averitec/averitec_pilot_100.csv --language en --limit 5
```

With GLiNER:

```cmd
python scripts/run_batch.py --input data/datasets/averitec/averitec_pilot_100.csv --language en --limit 5 --with-ner
```

CFEVER when the JSONL contains a string claim field:

```cmd
python scripts/run_batch.py --input data/datasets/cfever/train.jsonl --language zh --limit 5 --with-ner
```

If your particular dataset variant uses a different field name, check the
preview columns in Streamlit or add `--claim-field <name>` to CLI.

## Optional human-reviewed Wikidata links

The app supports optional reviewed-link CSV. **Do not bulk search and silently
assign the first candidate**; a Wikidata search result is not a validated identity.

Create your own file containing mappings you have checked:

```csv
language,surface,qid,label
id,Prabowo Subianto,Q2107268,Prabowo Subianto
id,Jakarta,Q3630,Jakarta
```

Then upload it in tab 4, or add `--approved-links <your-reviewed-file.csv>`.
If no mapping is provided, mentions are kept distinct between claims.
QID approval is **only entity identity linking, not verification of a relation or claim**.

## Safety and evaluation integrity

- Input labels, gold explanations and evidence are **not used** to build this KG.
- `test` / held-out splits are blocked by this pilot importer to reduce accidental leakage.
- All relations retain `truth_status=unverified`; all extracted edges have `extraction_review=pending`.
- A clause matching `VISITS` **does not prove a visit actually happened**. Rules can be semantically coarse.
- v0.4 currently doesn't import external evidence, retrieve Wikidata facts automatically, or perform fake-news classification.
- The output is a **candidate assertion graph** for data engineering and future experiments, not a trusted world-fact KG.
- For multilingual data, different datasets have different label semantics; **don't concatenate labels** without a documented mapping and held-out evaluation protocol.
- No deployment-specific authentication; use on local machine, not as a public file-processing endpoint.

## What comes after v0.4

**v0.5**: a proper text-only baseline, followed by a controlled text+graph experiment
with split-safe KG construction and ablations. Optional LLM relation extraction
can be evaluated as an additional module, not assumed correct.
