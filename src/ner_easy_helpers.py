"""Pure helpers for the optional Indonesian guided review UI.

State and validation remain delegated to ner_annotation.py. No AI approval here.
"""
from __future__ import annotations

from src.ner_annotation import CORE_LABELS, all_candidates

LABEL_ID = {
    "person": "Orang / tokoh",
    "organization": "Organisasi / lembaga",
    "location": "Lokasi / tempat",
    "event": "Peristiwa / kegiatan",
}

ACTION_ID = {
    "PENDING": "Belum diputuskan",
    "ACCEPT": "Terima nama & jenis ini",
    "REVISE": "Perbaiki nama atau jenis",
    "REJECT": "Tolak — bukan nama entitas yang tepat",
    "OUT_OF_SCOPE": "Di luar 4 kategori utama",
    "NEEDS_CONTEXT": "Tunda — perlu pemeriksaan lanjutan",
}

SUGGESTION_ID = {
    "ACCEPT_RECOMMENDED": "AI menyarankan diterima",
    "REVISE_RECOMMENDED": "AI menyarankan diperbaiki",
    "REJECT_RECOMMENDED": "AI menyarankan ditolak",
    "NEEDS_ADJUDICATION": "AI meminta pemeriksaan lebih lanjut",
    "NONE": "Tidak ada rekomendasi AI",
}


def find_surface_positions(claim: str, surface: str) -> list[tuple[int,int]]:
    """Locate exact Chinese substrings, including repeated and overlapping occurrences."""
    if not isinstance(surface, str) or not surface.strip():
        return []
    if surface != surface.strip():
        return []
    found = []
    at = 0
    while True:
        pos = claim.find(surface, at)
        if pos < 0:
            return found
        found.append((pos, pos + len(surface)))
        at = pos + 1


def candidate_counts_for_claim(claim_id: str, data: dict, state: dict) -> dict:
    rows = [x for x in all_candidates(data, state) if x['claim_id'] == claim_id]
    decisions = state.get('decisions', {})
    todo = [r for r in rows if decisions.get(r['candidate_id'], {}).get('action', 'PENDING') in ('PENDING', 'NEEDS_CONTEXT')]
    core = [r for r in rows if decisions.get(r['candidate_id'], {}).get('action') in ('ACCEPT', 'REVISE')
            and decisions[r['candidate_id']].get('label') in CORE_LABELS]
    confirmed = state.get('claim_reviews', {}).get(claim_id, {}).get('action') in ('CONFIRMED_COMPLETE', 'CONFIRMED_NO_CORE')
    return {'total':len(rows), 'todo':len(todo), 'core':len(core), 'confirmed':confirmed}


def first_unresolved(rows: list[dict], state: dict, after_id: str | None = None) -> str | None:
    """Find the next unresolved candidate, preferring rows after the current one."""
    if not rows:
        return None
    start = 0
    if after_id is not None:
        ids = [r['candidate_id'] for r in rows]
        if after_id in ids:
            start = (ids.index(after_id) + 1) % len(rows)
    for j in range(len(rows)):
        candidate = rows[(start+j) % len(rows)]
        action = state.get('decisions',{}).get(candidate['candidate_id'],{}).get('action','PENDING')
        if action in ('PENDING', 'NEEDS_CONTEXT'):
            return candidate['candidate_id']
    return None


def candidate_is_core(label: str) -> bool:
    return label in CORE_LABELS
