"""v0.6.1d UX behavior without a browser; does not auto-approve AI."""
from pathlib import Path

from src.ner_annotation import (load_candidates, load_state, save_candidate_decision,
                                save_claim_review, review_summary)
from src.ner_ui_v2_helpers import (focus_candidate, highlighted_html,
                                  first_unfinished_claim, next_candidate)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/annotations/cfever_zh_review_candidates_v061a.json'


def test_focus_first_pending_instead_of_last_saved():
    rows = [{'candidate_id': x} for x in ('a', 'b', 'c')]
    state = {'decisions': {'a': {'action': 'ACCEPT'}}}
    assert focus_candidate(rows, state, 'a')['candidate_id'] == 'b'


def test_focus_explicit_reopens_saved_decision():
    rows = [{'candidate_id': x} for x in ('a', 'b')]
    state = {'decisions': {'a': {'action': 'REJECT'}}}
    assert focus_candidate(rows, state, 'a', review_saved=True)['candidate_id'] == 'a'


def test_focus_returns_none_if_everything_finished():
    rows = [{'candidate_id': x} for x in ('a', 'b')]
    state = {'decisions': {'a': {'action': 'ACCEPT'}, 'b': {'action': 'OUT_OF_SCOPE'}}}
    assert focus_candidate(rows, state) is None


def test_focus_prefers_unchecked_over_deferred():
    rows = [{'candidate_id': x} for x in ('a', 'b')]
    state = {'decisions': {'a': {'action': 'NEEDS_CONTEXT'}}}
    assert focus_candidate(rows, state, 'a')['candidate_id'] == 'a'  # explicit deferred review
    assert focus_candidate(rows, state)['candidate_id'] == 'b'


def test_all_user_content_is_escaped():
    claim = '西安<script>alert(1)</script>城'
    html = highlighted_html(claim, 0, 2)
    assert '&lt;script&gt;' in html and '<script>' not in html
    assert '<mark class="vg-highlight">西安</mark>' in html


def test_review_changes_only_json_state_not_source(tmp_path):
    data = load_candidates(DATA)
    source_digest = data['source_digest']
    path = tmp_path / 'decisions.json'
    first = data['candidates'][0]
    save_candidate_decision(path, data, first['candidate_id'], 'ACCEPT',
                            start=first['start'], end=first['end'], label=first['label'])
    state = load_state(path, data)
    summary = review_summary(data, state)
    assert summary['decided'] == 1 and summary['claims_confirmed'] == 0
    assert data['source_digest'] == source_digest
    assert state['events'][0]['kind'] == 'candidate_decision'
    assert first_unfinished_claim(data, state) == data['claims'][0]['claim_id']


def test_deferred_is_never_counted_as_final(tmp_path):
    data = load_candidates(DATA)
    path = tmp_path / 'decisions.json'
    first = data['candidates'][0]
    save_candidate_decision(path, data, first['candidate_id'], 'NEEDS_CONTEXT')
    summary = review_summary(data, load_state(path, data))
    assert summary['decided'] == 0 and summary['needs_context'] == 1
    assert next_candidate([first], load_state(path, data), first['candidate_id']) == first['candidate_id']


def test_claim_completion_invalidation_still_enforced(tmp_path):
    data = load_candidates(DATA)
    path = tmp_path / 'decisions.json'
    cid = data['claims'][0]['claim_id']
    rows = [r for r in data['candidates'] if r['claim_id'] == cid]
    # Must not be possible to bypass the backend review gate with UI changes.
    import pytest
    with pytest.raises(ValueError):
        save_claim_review(path, data, cid, 'CONFIRMED_NO_CORE')
    for r in rows:
        save_candidate_decision(path, data, r['candidate_id'], 'REJECT')
    save_claim_review(path, data, cid, 'CONFIRMED_NO_CORE')
    assert load_state(path, data)['claim_reviews'][cid]['action'] == 'CONFIRMED_NO_CORE'
