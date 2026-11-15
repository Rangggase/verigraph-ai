"""Additional, non-destructive quality checks for the v0.6.1 Chinese NER pilot.

The rules are conservative, explicit, and specific to the manually inspected pilot.
They check *annotation identity/types*, not the truth of the source claim.
No model inference, network calls, file writes, or automatic approvals occur here.
"""
from __future__ import annotations

from src.ner_annotation import CORE_LABELS, all_candidates, validate_span

# Known identity/label examples discussed during the first seven-claim audit.
# This is NOT an exhaustive Chinese entity gazetteer or a final ground truth.
PILOT_TYPE_HINTS = {
    '皮特·烏斯蒂諾夫': 'person',
    '北冰洋': 'location',
    '伊麗莎白二世': 'person',
}


def audit_claim(data: dict, state: dict, claim_id: str) -> list[dict]:
    """Return flags with code/severity/message and stable candidate IDs.

    'block': an obvious conflict that should prevent a *new* completion.
    Existing saved decisions and historical completions are never modified.
    """
    claims = {r['claim_id']: r['text'] for r in data['claims']}
    if claim_id not in claims:
        raise ValueError('Claim ID tidak ada dalam data anotasi.')
    text = claims[claim_id]
    result = []
    found_core = []
    rows = [r for r in all_candidates(data, state) if r['claim_id'] == claim_id]
    decisions = state.get('decisions', {})

    for r in rows:
        candidate_id = r['candidate_id']
        d = decisions.get(candidate_id, {})
        if d.get('action') not in ('ACCEPT', 'REVISE', 'OUT_OF_SCOPE'):
            continue
        label = d.get('label')
        start, end = d.get('start'), d.get('end')
        try:
            surface = validate_span(text, start, end, label)
        except (TypeError, ValueError) as exc:
            result.append({'code': 'INVALID_SPAN', 'severity': 'block',
                           'candidate_ids': [candidate_id],
                           'message': f'{candidate_id}: offset/tipe tidak valid ({exc}).'})
            continue
        if surface != d.get('surface'):
            result.append({'code': 'OFFSET_MISMATCH', 'severity': 'block',
                           'candidate_ids': [candidate_id],
                           'message': f'{candidate_id}: teks nama dan posisi karakter tidak cocok.'})
            continue
        if label in CORE_LABELS and d['action'] in ('ACCEPT', 'REVISE'):
            found_core.append((start, end, label, candidate_id, surface))
            expected = PILOT_TYPE_HINTS.get(surface)
            if expected and expected != label:
                result.append({'code': 'PILOT_LABEL_MISMATCH', 'severity': 'block',
                               'candidate_ids': [candidate_id],
                               'message': (f'{candidate_id}: {surface} tersimpan sebagai {label}; '
                                           f'contoh audit pilot mengidentifikasinya sebagai {expected}. '
                                           'Periksa dan revisi tipe melalui Streamlit.')})

    found_core.sort(key=lambda x: (x[0], x[1], x[3]))
    for idx, left in enumerate(found_core):
        for right in found_core[idx + 1:]:
            if right[0] >= left[1]:
                break
            result.append({'code': 'OVERLAPPING_CORE', 'severity': 'block',
                           'candidate_ids': [left[3], right[3]],
                           'message': (f'{left[3]} dan {right[3]} memiliki rentang karakter '
                                       'yang bertumpang tindih. Jika nama muncul dua kali, '
                                       'pilih kemunculan yang berbeda.')})

    # A specific high-confidence missing-person hint from the pilot audit. Only
    # assert a potential omission if the exact surface exists in claim text.
    surface = '伊麗莎白二世'
    if surface in text and not any(r[4] == surface and r[2] == 'person' for r in found_core):
        result.append({'code': 'PILOT_MISSING_PERSON', 'severity': 'block',
                       'candidate_ids': [r['candidate_id'] for r in rows if r['surface'] == surface],
                       'message': (f'Nama {surface} ada dalam kalimat tetapi belum tercatat '
                                   'sebagai person. Periksa kandidatnya atau tambahkan nama yang terlewat.')})
    return result


def audit_dataset(data: dict, state: dict) -> list[dict]:
    """Inspect every claim without changing decisions or completions."""
    return [{'claim_id': row['claim_id'], **issue}
            for row in data['claims'] for issue in audit_claim(data, state, row['claim_id'])]
