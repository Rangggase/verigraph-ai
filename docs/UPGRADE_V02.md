# VeriGraph AI v0.2 — Upgrade instructions

This update preserves the curated v0.1 demo and adds a second tab for optional **model-based** entity extraction, candidate Wikidata linking, and unverified `CLAIM -> MENTIONS -> ENTITY` graph visualization.

## Requirements

- Python 3.10+ recommended, PyTorch available through GLiNER dependencies
- RAM/VRAM sufficient for a ~209M parameter model; GPU helpful but optional
- Internet on the **first** model load and for **user-triggered** Wikidata searches
- Install can modify torch/transformers dependencies. Prefer a separate/clone Conda environment if `kg_e` is used by other research projects

## Install & run (Windows CMD)

```cmd
conda activate kg_e
cd /d "D:\new another project\VeriGraph_AI_Starter_Kit\verigraph_ai_starter"
python -m pip check
python -m pip install -r requirements-ner.txt
python -m pip check
python -m pytest -q
python -m streamlit run app.py
```

If install conflicts with your existing environment, clone rather than damaging it:

```cmd
conda create -n verigraph_v02 --clone kg_e
conda activate verigraph_v02
python -m pip install -r requirements-ner.txt
```

## Use

1. Switch to **2. NER & Wikidata (v0.2)**.
2. Choose `id`, `en`, or `zh`; paste a short sentence.
3. Click **Extract named entities**. First run downloads the pretrained checkpoint and may take some time.
4. Recognized entities matching curated local aliases will link immediately.
5. For unknown entities, click **Search Wikidata candidates**, inspect labels/descriptions and follow the QID link, then manually **Approve selected Wikidata identity** if correct.
6. Inspect the separate *unverified mention graph*, then download JSON.

## What to look for

- NER extraction is model based; neither guaranteed recall nor precision is implied.
- Model card specifies multilingual but you should manually inspect NER for Indonesian, Chinese and English.
- Entity names are not the same as Wikidata entity identities; name collision and cross-language ambiguity require human review.
- Wikidata search results are **candidates**, not facts; this code does not assume the top result is correct.
- Claim mention graph is **not a verified fact graph**. A claim mentioning an organization does not mean a claim about that organization is true.
- No relationship extraction, classifier, KG neural encoder, or generative verifier has been implemented.

## If model download / load fails

Check Python, pip, torch, transformers version and available memory:

```cmd
python --version
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If the model cannot fit your hardware, start with a smaller checkpoint or use a cloud runtime. Do not overwrite your earlier v0.1 ZIP before saving the old version.

## Tests

The additional tests use a stub NER model and a mocked Wikidata request, so they don't need a GPU or internet. They ensure mentions are not inserted as trusted fact triples and that candidate search never auto-approves identities.
