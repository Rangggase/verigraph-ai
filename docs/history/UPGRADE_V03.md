# VeriGraph AI v0.3 — Candidate Assertion Knowledge Graph

## What's new

- Keeps **all v0.1 & v0.2 functionality**, including manual Wikidata identity approval.
- Third Streamlit tab builds relation candidates from **auditable keyword + typed-entity templates**, not a generative LLM.
- Starter templates cover Indonesian, English, and Chinese: `VISITS`, `MEETS`, `ATTENDS_IN`. They intentionally **abstain** when there is no suitable pattern. These rules are limited and are not a general relation extraction solution.
- Each assertion edge carries `provenance_claim_id`, `evidence_text`, `evidence_trigger`, `trigger_start`, `trigger_end`, `extraction_method`, `extraction_review`, and `truth_status`.
- Manual review is about *whether text describes the candidate relation*, **not** whether it is factually true.
- Download a JSON graph with candidates and source provenance; approved Wikidata identities are reused where available.
- Tests run without network/model download.

## Install / run on Windows (Conda)

```cmd
conda activate kg_e
cd <your-v0.3-project-folder>\verigraph_ai_starter
python -m pip install -r requirements-ner.txt
python -m pytest -q
python -m streamlit run app.py
```

If v0.2 already works on your PC, you usually only need to start v0.3 after copying it into a new folder; **no new model download should be necessary** because model weights are cached by Hugging Face. Check the Conda environment before changing packages. If `kg_e` is used by another project, clone it before installing.

## Demo workflow

1. Tab **NER & Wikidata**: input "Prabowo Subianto melakukan kunjungan kerja di Jakarta." and click **Extract named entities**.
2. Review and approve candidate Wikidata identity for Prabowo Subianto (`Q2107268`) and Jakarta (`Q3630`), as shown in your v0.2 test. **The new Streamlit app starts a new in-memory session, so approval selections are not carried from the previous app unless you re-approve them.**
3. Tab **Candidate Relations**: check candidate `Prabowo Subianto — VISITS → Jakarta`.
4. Select `Accurate as a textual assertion (not verified)` if it represents the sentence accurately. This **does not** approve the fact as true.
5. Download JSON and inspect `graph.links`: the relation edge still has `truth_status: "unverified"`.
6. Repeat with English "Barack Obama visited Paris and met Angela Merkel." and Chinese "习近平在北京出席会议。"

## Limits and next steps

- **No claim truth verification** or real/fake classification yet.
- **No source-evidence retrieval** or fact comparison beyond the curated capital-city demo.
- Textual relations are unverified; even a correct entity QID does not validate the asserted relation.
- Current relation candidates are narrow, pattern-based heuristics. Future v0.4: structured LLM relation extraction with evidence spans, entity resolution validation, independent provenance corpus, and persisted claims from multiple documents.
- Session data is ephemeral; use Download JSON to preserve the output.
- Do not treat test-data claim outputs as independent verified evidence (avoid label leakage).
