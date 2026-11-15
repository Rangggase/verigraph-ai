# 4-week build plan (portfolio-focused)

This is an implementation target, not a promise that every research-grade model will be complete in four weeks.

## Days 1–2: runnable, transparent KG MVP

- [ ] Run the 18 multilingual fixtures; export graph JSON + GraphML
- [ ] Inspect 3 capital relations, linked Wikidata entity IDs and references
- [ ] Run UI demo and take a screenshot
- [ ] Add repository and clear current-vs-planned feature list to README

**Gate:** `python -m pytest -q` passes and `streamlit run app.py` works.

## Days 3–7: first real data and extraction

- [ ] Select one *real* labeled dataset and record license, sizes, languages, task and label semantics
- [ ] Split by event/claim family where possible; guard against duplicate-leakage
- [ ] Implement entity extraction baseline (multilingual spaCy, separate language-specific models, or a multilingual transformer)
- [ ] Entity-link to Wikidata where appropriate; log unknown entities (do not force mappings)
- [ ] Create *claim assertions* graph distinct from validated *evidence* graph
- [ ] Evaluate NER / linking quality manually on a small annotated sample

**Gate:** entities/relations for 30–50 real claims can be inspected with provenance.

## Week 2: text-only fake-news ML/DL baseline

- [ ] Train TF-IDF + Logistic Regression on a genuine annotated dataset
- [ ] Fine-tune multilingual encoder (e.g. XLM-R), with compute-aware configuration
- [ ] Report class distributions, macro-F1, confusion matrix, per-language breakdown
- [ ] Save configs, seeds, versions, model card and evaluation script

**Gate:** reproducible text-only result on a held-out test set.

## Week 3: add graph learning

- [ ] Construct train-only or time-valid KG, with cited evidence and leakage checks
- [ ] Baseline graph features (entity/relation counts or embeddings)
- [ ] Optional: GraphSAGE/R-GCN and text–graph fusion
- [ ] Compare text-only, graph-only, text+graph under fixed splits
- [ ] Analyze cases where KG helps/hurts

**Gate:** ablation demonstrates whether KG contributes, not merely that the code runs.

## Week 4: showcase

- [ ] Streamlit input, graph inspection, model predictions, evidence source panel
- [ ] Explicit distinction between evidence-based findings and low-confidence prediction
- [ ] Container or reproducible local run; basic tests and error paths
- [ ] GitHub README, architecture diagram, results table, limitations, demo video
- [ ] Optional generative explanation, strictly grounded in sourced evidence

**Gate:** a reviewer can clone, run, and see an honest end-to-end demo.

### Scope control

If 3-language annotated datasets are not comparable, keep each test set separate and build a *parallel/translated* diagnostic subset for cross-language consistency. Do not claim zero-shot performance from incomparable datasets.
