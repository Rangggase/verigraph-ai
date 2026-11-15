"""Pure presentation/navigation helpers for v0.6.1c guided review.

No model calls, network, file writes, or implicit approval of AI suggestions.
"""
from __future__ import annotations

import html

from src.ner_annotation import CORE_LABELS, all_candidates

LABEL_FRIENDLY = {
    'person': 'Orang', 'organization': 'Organisasi / lembaga',
    'location': 'Tempat / wilayah', 'event': 'Peristiwa',
}
DECISION_FRIENDLY = {
    'PENDING': 'Belum diperiksa', 'ACCEPT': 'Diterima',
    'REVISE': 'Sudah diperbaiki', 'REJECT': 'Ditolak',
    'OUT_OF_SCOPE': 'Kategori lain', 'NEEDS_CONTEXT': 'Perlu ditinjau lagi',
}


def readable_label(label: str) -> str:
    if label in LABEL_FRIENDLY:
        return LABEL_FRIENDLY[label]
    if label.startswith('OTHER:'):
        return 'Kategori lain: ' + label.split(':', 1)[1].replace('_', ' ').title()
    if label.startswith('AMBIGUOUS:'):
        return 'Belum pasti: ' + label.split(':', 1)[1].replace('_', ' ').title()
    return label or 'Belum dikategorikan'


def highlighted_html(text: str, start: int | None = None, end: int | None = None) -> str:
    """Escape all user-provided claim text before applying visual highlighting."""
    if (isinstance(start, int) and not isinstance(start, bool) and
        isinstance(end, int) and not isinstance(end, bool) and 0 <= start < end <= len(text)):
        rendered = (html.escape(text[:start]) +
                    '<mark class="vg-highlight">' + html.escape(text[start:end]) + '</mark>' +
                    html.escape(text[end:]))
    else:
        rendered = html.escape(text)
    return '<div class="vg-sentence" lang="zh">' + rendered + '</div>'


def next_candidate(rows: list[dict], state: dict, current_id: str | None = None) -> str | None:
    """Advance to truly pending work before deferrals; never treat deferrals as finished."""
    ids = [r['candidate_id'] for r in rows]
    if not ids:
        return None
    start = (ids.index(current_id) + 1) % len(ids) if current_id in ids else 0
    rotated = ids[start:] + ids[:start]
    decisions = state.get('decisions', {})
    for target in ('PENDING', 'NEEDS_CONTEXT'):
        for cid in rotated:
            if decisions.get(cid, {}).get('action', 'PENDING') == target:
                return cid
    return None


def first_unfinished_claim(data: dict, state: dict, after_id: str | None = None) -> str | None:
    claims = data['claims']
    if not claims:
        return None
    ids = [c['claim_id'] for c in claims]
    start = (ids.index(after_id) + 1) % len(ids) if after_id in ids else 0
    for cid in ids[start:] + ids[:start]:
        if state.get('claim_reviews', {}).get(cid, {}).get('action') not in (
            'CONFIRMED_COMPLETE', 'CONFIRMED_NO_CORE'
        ):
            return cid
    return None


def claim_progress(data: dict, state: dict, claim_id: str) -> dict:
    rows = [r for r in all_candidates(data, state) if r['claim_id'] == claim_id]
    decisions = state.get('decisions', {})
    waiting = sum(decisions.get(r['candidate_id'], {}).get('action', 'PENDING') in
                  ('PENDING', 'NEEDS_CONTEXT') for r in rows)
    return {'total': len(rows), 'decided': len(rows) - waiting, 'waiting': waiting,
            'confirmed': state.get('claim_reviews', {}).get(claim_id, {}).get('action')
            in ('CONFIRMED_COMPLETE', 'CONFIRMED_NO_CORE')}


def is_core(label: str) -> bool:
    return label in CORE_LABELS


def focus_candidate(rows: list[dict], state: dict, requested: str | None = None,
                    review_saved: bool = False) -> dict | None:
    """Select a work item without unexpectedly reopening a completed decision.

    Explicit requests from the 'Ubah keputusan lama' control are allowed.
    This is pure navigation, with no writes and no automatic AI approvals.
    """
    decisions = state.get('decisions', {})
    for row in rows:
        if row['candidate_id'] == requested:
            status = decisions.get(requested, {}).get('action', 'PENDING')
            if review_saved or status in ('PENDING', 'NEEDS_CONTEXT'):
                return row
            break
    target = next_candidate(rows, state)
    return next((r for r in rows if r['candidate_id'] == target), None)
