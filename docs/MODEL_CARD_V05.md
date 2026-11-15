# Model Card — VeriGraph AI v0.5 Text-Only ML Baseline

## Intended use

A reproducible **claim-text classification baseline** (not factual verification). It is a reference point for later evidence retrieval, graph learning, and text–graph fusion experiments.

## Supported experiments

- **CFEVER**: Simplified Chinese traditional claims; three labels: `supports`, `refutes`, `not enough info`. Official source train and dev files available locally.
- **AVeriTeC**: English; four labels. Only a 100-row sample is locally provided; its result is a **pilot** and must not be reported as a benchmark. Download official train/dev files with `scripts/download_averitec.py` for a meaningful split-based evaluation.
- **Indonesian**: not included yet; requires a separate labeled dataset and evaluation.

## Data audit and split safety — CFEVER

Input files are in `data/datasets/cfever/`. The training file has 24,012 rows; dev has 3,000 rows. Before training:

1. The `NOT ENOUGH INFO` label is canonicalized to `not enough info`.
2. Exact normalized duplicate claims with the **same** label are removed from training (610 redundant rows).
3. Conflicting-label claim groups are removed from training (101 rows).
4. Exact normalized train–dev claim overlaps are excluded from training (1 row).
5. The dev dataset is preserved in its original 3,000-row state (including 27 duplicate rows), rather than modifying the published dev composition. A source artifact can still contain semantic near-duplicates, not checked here.

Cleaned train: **23,300** rows. Zero exact normalized claim overlap remains. This is a **cleaned-source-split experimental protocol**, not a verbatim reimplementation of an official paper baseline; published numbers may not be directly comparable.

## Model and reproducibility

- TF-IDF character n-grams (2 to 4), max features 100,000, min_df 2, sublinear_tf true
- `LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)`
- Only feature: `claim` (text)
- Gold evidence, fact-checker justification, URLs and verdict strings are **not** input features
- Macro-F1 and per-class metrics on the full 3,000-row source dev split

## Locally reproduced metrics

| Metric | Value |
|---|---:|
| Macro-F1 | 0.4240 |
| Accuracy | 0.4273 |
| Majority macro-F1 | 0.1667 |
| Dev rows | 3,000 |

Class F1: `supports` 0.4552, `refutes` 0.4248, `not enough info` 0.3919. See `outputs/ml_v05/cfever/metrics.json` for the confusion matrix and all evaluation details.

This modest performance is expected from claim-only text and should **not** be interpreted as evidence that the claim was verified. Language/split-specific dataset patterns, class biases and source effects can influence predictions.

## Limitations and safety

- No evidence retrieval, temporal checks, real KG reasoning, or factual justification.
- Class labels are not universally comparable to AVeriTeC labels; do not combine their scores.
- The `.joblib` artifact is trusted *local training output* and uses Python pickling; **do not load third-party uploaded joblib files**.
- If scikit-learn package version differs between saved file and user environment, train locally to regenerate it.
- Train-only duplicate/conflict cleanup means method comparisons must use the same sanitized train split and unchanged dev split.
- The GLiNER review pipeline is independent from this baseline; no KG performance benefit has been established.

## Next steps

Fine-tune a multilingual Transformer on identical splits; compare Macro-F1 and error analysis. Only then build stronger training-allowed KG coverage, entity resolution, evidence provenance, and GNN integration with a separate model branch. Test whether graph representations actually improve performance using a text-only control and shuffled-KG ablation.
