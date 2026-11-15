# Data contract v0.1

## Entities (`data/entities.csv`)

- `entity_id`: stable canonical identifier (Wikidata Q-id in current fixture)
- `entity_type`: `city`, `country`, etc.
- `label_en`, `label_id`, `label_zh`: localized labels
- `aliases_*`: aliases separated by `|` within a language
- `reference_url`: human-readable entity reference

## Relations (`data/relations.csv`)

- `subject_id`, `predicate`, `object_id`: e.g. Q17 / capital / Q1490
- `reference_url`: pointer to provenance, not a guarantee a statement is correct for all time
- `source_type`: how the relation was collected
- `evidence_status`: here `curated_demo`; extend to independently verified/unverified later

**Future fields recommended:** `valid_from`, `valid_until`, `retrieved_at`, `claim_id`, `evidence_snippet`, `extractor_model`, `confidence`, `review_status`, `source_license`.

## Demo claims (`data/demo_claims.csv`)

- `claim_id`, `language`, `text`, `expected_relation`
- `expected_relation` describes relation consistency (`supported` or `conflict`) **within the curated example KG**. It is not a real fake-news ground-truth label.

## Leakage guardrails

- Do not construct test-time trusted KG facts by mining labeled evaluation claim verdicts.
- For a given test claim, avoid using its gold fact-check rationale as a model feature.
- Track entity/claim duplicates and close paraphrases across train/dev/test.
- Do not translate dataset labels and assume equivalence without auditing the original definitions.
