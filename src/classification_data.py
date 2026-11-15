"""Claim-only labelled examples. NEVER pass gold evidence/justifications into features."""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path
from collections import Counter
from typing import Any

AVERITEC_LABELS = [
    'Supported', 'Refuted', 'Not Enough Evidence',
    'Conflicting Evidence/Cherrypicking',
]
CFEVER_LABELS = ['supports', 'refutes', 'not enough info']
LABELS = {'averitec': AVERITEC_LABELS, 'cfever': CFEVER_LABELS}


def normalize_claim(claim: str) -> str:
    return re.sub(r'\s+', ' ', claim).strip().casefold()


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Dataset missing: {path}')
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    if path.suffix.lower() == '.jsonl':
        with path.open(encoding='utf-8-sig') as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.suffix.lower() == '.json':
        with path.open(encoding='utf-8-sig') as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        raise ValueError('Expected JSON array of claims')
    raise ValueError(f'Unsupported dataset extension: {path.suffix}')


def load_examples(path: str | Path, dataset: str, require_label=True) -> list[dict[str, str]]:
    if dataset not in LABELS:
        raise ValueError('Unsupported dataset')
    output = []
    for i, r in enumerate(read_rows(path)):
        if not isinstance(r, dict):
            raise ValueError(f'Row {i} is not an object')
        claim = r.get('claim', '')
        label = r.get('label')
        if dataset == 'cfever' and isinstance(label, str):
            label = label.strip().lower()  # e.g. official uppercase NOT ENOUGH INFO
        if not isinstance(claim, str) or not claim.strip():
            continue
        if require_label and label not in LABELS[dataset]:
            raise ValueError(f'Row {i} has invalid/missing label {label!r} for {dataset}')
        # CRITICAL: extract claim only; never include evidence, justification, links or verdict narrative.
        output.append({'claim': claim.strip(), 'label': str(label) if label is not None else '',
                       'source_id': str(r.get('claim_id', r.get('id', i)))})
    if not output:
        raise ValueError('No valid claims')
    return output


def sanitize_training(rows: list[dict]) -> tuple[list[dict], dict]:
    """Remove identical duplicated claims, and all conflicting-label duplicates, deterministically."""
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in rows:
        grouped[normalize_claim(row['claim'])].append(row)
    clean = []
    same_label_removed = 0
    conflicts_removed = 0
    for group in grouped.values():
        if len({r['label'] for r in group}) > 1:
            conflicts_removed += len(group)
            continue
        clean.append(group[0])
        same_label_removed += len(group)-1
    return clean, {'train_identical_duplicates_removed': same_label_removed,
                   'train_conflicting_rows_removed': conflicts_removed}


def verify_disjoint(train: list[dict], dev: list[dict]) -> dict:
    train_texts = [normalize_claim(r['claim']) for r in train]
    dev_texts = [normalize_claim(r['claim']) for r in dev]
    overlap = set(train_texts) & set(dev_texts)
    if overlap:
        raise ValueError(f'Data leakage: {len(overlap)} exact normalized claims appear in train and dev')
    return {'train_size': len(train), 'dev_size': len(dev),
            'train_labels': dict(Counter(r['label'] for r in train)),
            'dev_labels': dict(Counter(r['label'] for r in dev)),
            'normalized_claim_overlap': 0,
            'dev_duplicate_rows': len(dev_texts) - len(set(dev_texts))}


def prepare_split(dataset: str, train_path: str | Path, dev_path: str | Path | None,
                  pilot: bool = False, seed: int = 42, max_train: int | None = None,
                  max_dev: int | None = None):
    """Preserve official dev; sanitize training duplicates and prevent overlap before fitting.

    Pilot mode creates an 80/20 diagnostic split from the local 100-row sample.
    Actual official train/dev files are separate; no rerandomization of dev.
    """
    rows = load_examples(train_path, dataset)
    if pilot:
        if dev_path is not None:
            raise ValueError('Pilot mode expects --train only; split locally for DIAGNOSTICS')
        rows, cleanup = sanitize_training(rows)
        from sklearn.model_selection import train_test_split
        counts = Counter(r['label'] for r in rows)
        if len(rows) < 20 or min(counts.values()) < 2:
            raise ValueError('Pilot needs >=20 rows, >=2 samples per label')
        train, dev = train_test_split(rows, test_size=0.2, random_state=seed,
                                      stratify=[r['label'] for r in rows])
        mode = 'PILOT_NOT_BENCHMARK'
    else:
        if dev_path is None:
            raise ValueError('Official evaluation requires separate --dev; never evaluate on train')
        dev = load_examples(dev_path, dataset)
        rows, cleanup = sanitize_training(rows)
        dev_keys = {normalize_claim(r['claim']) for r in dev}
        original = len(rows)
        train = [r for r in rows if normalize_claim(r['claim']) not in dev_keys]
        cleanup['train_dev_overlapping_rows_removed'] = original-len(train)
        # Official dev contents remain intact. Training cleanup means not identical to source training protocol.
        mode = 'CLEANED_SOURCE_SPLIT'
    if max_train is not None:
        if max_train < 2: raise ValueError('max_train must be >=2')
        train = train[:max_train]
    if max_dev is not None:
        if max_dev < 1: raise ValueError('max_dev must be >=1')
        dev = dev[:max_dev]
    summary = verify_disjoint(train, dev)
    if len(set(r['label'] for r in train)) < 2:
        raise ValueError('Train subset has fewer than two classes')
    summary.update({'mode': mode, 'dataset': dataset, 'seed': seed,
                    'limited': max_train is not None or max_dev is not None,
                    'train_path': str(train_path), 'dev_path': str(dev_path) if dev_path else None,
                    'training_cleanup': cleanup})
    return train, dev, summary
