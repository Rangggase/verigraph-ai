# VeriGraph AI v0.4.2 — Reviewed & Canonical Knowledge Graph

## Motivation

v0.4.1 retains all detected mentions for auditing, including rejected ones. A raw
claim–mention graph **must not** be treated as a reliable factual KG or directly
used as a vetted GNN graph. The revised workflow preserves provenance and builds
a separate conservative graph from explicit human decisions.

## Starting point

- Use the included `outputs/batch_kg_v041.json` (5 claims, 18 mentions, 2 candidate relations), or upload your own v0.4.1 bundle.
- Your existing reviewed decisions are preserved (4 accepted, 2 rejected, 12 pending).
- No additional GLiNER download, GPU, external API, or `pip install` is necessary if v0.4.1 worked in your Conda environment.

### UI workflow

1. Activate the existing environment and open **the extracted `verigraph_ai_starter` folder**.
2. Run `python -m pytest -q` and `python -m streamlit run app.py`.
3. Open **6. Reviewed & Clean KG (v0.4.2)**. Select **Saved v0.4.1 output**.
4. Review mentions. Only `accepted_mention` enters the clean graph.
5. Optionally search Wikidata candidates; copy a QID **only after verifying it manually**, and only for `accepted_mention` rows. Leave ambiguous identities unresolved.
6. Review candidate semantic relations separately. `accepted_extraction` means the text *asserts* the relation; **it does not confirm its truth**.
7. Press **Build clean & audit graphs (v0.4.2)**. Inspect the metrics, validation, and side-by-side graph views.
8. Download the bundle and/or clean graph. Persistent export: `outputs/batch_kg_v042.json`.

### Offline CLI

From the project root:

```cmd
conda activate kg_e
python -m pytest -q
python scripts/build_clean_kg.py --input outputs/batch_kg_v041.json --output outputs/batch_kg_v042.json
```

The CLI **does not auto-approve pending items**. Use the UI to edit review decisions first.

## Outputs and semantics

| Field | Meaning |
| --- | --- |
| `audit_graph` | All original mentions, review decisions, and candidates; never delete the provenance trail |
| `clean_graph` | All five claim nodes; **only accepted** mention edges; **only accepted** candidate relations with accepted endpoints |
| `gnn_graph` | Adds explicitly labeled reverse **message-passing** edges without changing the semantic direction in `clean_graph` |
| `gnn_ready` | Deterministic `node_mapping`, `edge_index`, and `edge_type_mapping`; **no features/labels** |
| `validation` | Checks allowed relation states, direction of claim–mention edges, provenance, and unverified status |

Canonical identity rule: only a human-reviewed `Q123` ID causes cross-claim merging. If no QID is supplied, the original mention IDs remain distinct. The system validates the QID syntax but cannot independently certify that the chosen QID is semantically correct.

Graph links representing assertions always carry `truth_status: unverified`. `accepted_extraction` means human agreement with the **extraction**, not a fact check. The pipeline does not ingest gold labels or gold evidence as feature inputs.

### Expected first run on your 5-claim JSON

If you do not modify previous decisions or relations:

- 5 claim nodes; 18 raw mentions; 2 candidate relations still in the audit graph.
- Audit graph: 23 nodes / 20 edges.
- Clean graph: **9 nodes / 4 claim–mention edges** (4 accepted mentions; no reviewed QIDs).
- Accepted semantic edges in the clean graph: **0**, because both candidates are still `pending`.
- `gnn_graph`: 9 nodes / 8 edges, including inverse message-passing edges.

To test semantic edges, mark `BOARD_MEMBER_OF` as `accepted_extraction` **only after checking the claim text and both endpoint mentions**. Then it appears as an *unverified assertion* in the clean graph. `PLEDGED_TO` is excluded if either endpoint is still `pending`, even if the relation itself is marked accepted.

## Next stage — v0.5

Build text-only ML/DL baselines with reproducible splits and macro-F1 first. Once the text model is stable, integrate clean knowledge representations into a graph encoder. In the GNN, ensure node-to-claim message flow is active; the separate `gnn_graph` includes reverse edges for that purpose. Avoid treating unverified claim assertions as authoritative truth and avoid injecting benchmark gold labels or gold fact-checker justifications into prediction features.
