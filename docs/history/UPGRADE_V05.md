# VeriGraph AI v0.5 — Claim-only ML / DL baselines

This update extends the user's v0.4.2 source code, preserving tabs 1–6. Tab 7 reads **local trained artifacts**, without training in Streamlit.

## Why v0.5 starts with text only

The uploaded v0.4.2 clean KG has **7 nodes, 2 MENTIONS edges**, with 2 accepted and 16 pending mentions, 0 canonical QIDs, and 0 accepted semantic edges. That KG is insufficient to claim any benefit from GNN. Establish the text-only baseline first; do not claim an evidence-grounded fact checker yet.

## Safe setup in VS Code / Conda (Windows CMD)

Use your existing `kg_e` environment *only if you are comfortable updating its ML packages*. To avoid affecting another project, clone it first with `conda create -n verigraph_ml --clone kg_e` then `conda activate verigraph_ml`.

```cmd
cd path\to\verigraph_ai_starter
python -m pip install -r requirements-ml.txt
python -m pytest -q
```

No GPU or PyTorch reinstall needed for TF-IDF.

## A) Quick pilot — English AVeriTeC 100 claims (NOT official evaluation)

```cmd
python scripts/train_ml_baseline.py --dataset averitec --train data/datasets/averitec/averitec_pilot_100.csv --pilot --output outputs/ml_v05/averitec_pilot
```

This stratified 80/20 split **inside a small training sample** is only an installation/data-flow smoke test. `metrics.json` explicitly flags `PILOT_NOT_BENCHMARK`. Do not compare these scores with published AVeriTeC numbers.

## B) Official English split — AVeriTeC (recommended for portfolio)

The repository `chenxwh/AVeriTeC` has `data/train.json` and `data/dev.json` from the same benchmark. Confirm license (noncommercial) and provenance on the source site: https://huggingface.co/chenxwh/AVeriTeC/tree/main/data

```cmd
python -m pip install huggingface_hub
python scripts/download_averitec.py
python scripts/train_ml_baseline.py --dataset averitec --train data/datasets/averitec/train.json --dev data/datasets/averitec/dev.json --output outputs/ml_v05/averitec
```

Only `claim` text is used as input features; labels are targets only. Fact-check articles, gold evidence, and justification are intentionally excluded.

## C) Chinese — CFEVER (24,012 train / 3,000 dev already inside ZIP)

```cmd
python scripts/train_ml_baseline.py --dataset cfever --train data/datasets/cfever/train.jsonl --dev data/datasets/cfever/dev.jsonl --output outputs/ml_v05/cfever
```

Each dataset is evaluated **separately**, preserving its own label mapping. CFEVER test.jsonl has no gold labels; do not evaluate on it.

For faster debugging, add `--max-train 2000 --max-dev 500` but reports are explicitly marked `SUBSET RUN`.

## D) Optional Deep Learning (multilingual DistilBERT / GPU recommended)

Start only after the ML baseline works. CPU is technically supported but can be slow.

```cmd
python -m pip install -r requirements-dl.txt
python scripts/train_dl_baseline.py --dataset cfever --train data/datasets/cfever/train.jsonl --dev data/datasets/cfever/dev.jsonl --max-train 1000 --max-dev 300 --epochs 1 --batch-size 4 --output outputs/dl_v05/cfever_smoke
```

Use the **same train/dev splits** and evaluation metrics to compare DL to ML; run the full official split for final comparative results. The default checkpoint is `distilbert-base-multilingual-cased` (requires download and adequate RAM); pass `--checkpoint` to select another multilingual model. Large Transformers are much slower than ML on PyTorch CPU.

## Expected outputs

- `outputs/ml_v05/<dataset>/model.joblib` (local trusted pickle-like artifact; never load from untrusted uploads)
- `outputs/ml_v05/<dataset>/metrics.json` (macro-F1, per-class scores, confusion matrix, provenance and experimental warnings)
- `outputs/ml_v05/<dataset>/predictions.csv` (per-sample dev predictions)
- `outputs/dl_v05/<dataset>/model/` + `metrics.json` for optional fine-tuning
- Streamlit tab `7. ML/DL Baselines (v0.5)` displays local **ML** reports and inference only; DL reports remain available as JSON.

## Scientific limitations

- Predictions based solely on a claim are **not evidence-grounded fact verification**. This baseline can learn artifact, topic or source biases.
- No evidence retrieval, no KG information in the feature set, and no claim that adding KG improves performance.
- AVeriTeC labels and CFEVER labels are not interchangeable (4 classes versus 3); never aggregate accuracy across these datasets.
- `max_train` / `max_dev` simply take an initial subset; not a representative benchmark. Present full dev split results separately, with the training deduplication protocol disclosed.
- Training exact duplicate claims are deduplicated, conflicting-label claims are dropped, and train/dev overlapping claims are removed from training. All counts are reported. Official dev duplicates remain untouched. These interventions mean results are not directly comparable with published uncleaned training protocols; semantic near-duplicates still need separate auditing.
- Indonesian-language baseline still needs a licensed labeled dataset with reliable splits. Do not call CFEVER/AVeriTeC a three-language benchmark.
- For future GNN experiments, construct KG and features using training-allowed data only, keep gold verdicts out of prediction features, and verify message-passing direction.
