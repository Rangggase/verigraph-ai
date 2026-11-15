"""UI regression tests that do not require opening a browser."""
from pathlib import Path

from src.ner_annotation import (
    load_candidates, load_state, save_candidate_decision, save_claim_review,
)
from src.ner_ui_v2_helpers import (
    readable_label, highlighted_html, next_candidate, first_unfinished_claim,
    claim_progress,
)
from src.ner_annotation_ui import _has_accepted_core, _recommendation_friendly

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / 'data/annotations/cfever_zh_review_candidates_v061a.json'


def test_html_escapes_chinese_claim_and_user_text():
    text = '北京<bad>在北京'
    marked = highlighted_html(text, 0, 2)
    assert '<mark class="vg-highlight">北京</mark>' in marked
    assert '&lt;bad&gt;' in marked
    assert '<bad>' not in marked


def test_invalid_offsets_render_no_highlight():
    assert '<mark' not in highlighted_html('臺北', -1, 2)
    assert '<mark' not in highlighted_html('臺北', 0, 20)
    assert '<mark' not in highlighted_html('臺北', True, 1)


def test_labels_simple_and_other():
    assert readable_label('person') == 'Orang'
    assert readable_label('organization') == 'Organisasi / lembaga'
    assert 'Kategori lain:' in readable_label('OTHER:WORK')


def test_navigation_prefers_new_pending_over_deferred():
    rows = [{'candidate_id':x} for x in ('a','b','c')]
    state = {'decisions': {'a': {'action':'NEEDS_CONTEXT'}, 'b':{'action':'REJECT'}}}
    assert next_candidate(rows, state, 'a') == 'c'
    state['decisions']['c'] = {'action':'ACCEPT'}
    assert next_candidate(rows, state, 'c') == 'a'
    state['decisions']['a'] = {'action':'REJECT'}
    assert next_candidate(rows, state, 'a') is None


def test_restarts_from_first_unfinished_claim(tmp_path):
    data = load_candidates(CANDIDATES)
    state = load_state(tmp_path/'empty.json',data)
    assert first_unfinished_claim(data,state) == data['claims'][0]['claim_id']
    first = data['claims'][0]['claim_id']
    state['claim_reviews'][first] = {'action':'CONFIRMED_COMPLETE'}
    assert first_unfinished_claim(data,state) == data['claims'][1]['claim_id']


def test_saved_decision_and_no_gold_autopromotion(tmp_path):
    data = load_candidates(CANDIDATES)
    path = tmp_path/'decisions.json'
    row = data['candidates'][0]
    save_candidate_decision(path,data,row['candidate_id'],'ACCEPT',
                            start=row['start'],end=row['end'],label=row['label'])
    state = load_state(path,data)
    assert claim_progress(data,state,row['claim_id'])['decided'] == 1
    assert not claim_progress(data,state,row['claim_id'])['confirmed']
    assert _has_accepted_core(data,state,row['claim_id'])
    assert len(state['events']) == 1
    assert _recommendation_friendly('ACCEPT_RECOMMENDED') == 'AI menyarankan diterima'


def test_confirming_empty_core_requires_completed_decisions(tmp_path):
    data = load_candidates(CANDIDATES)
    path = tmp_path/'decisions.json'
    cid = data['claims'][0]['claim_id']
    rows = [r for r in data['candidates'] if r['claim_id']==cid]
    for row in rows:
        save_candidate_decision(path,data,row['candidate_id'],'REJECT')
    assert not _has_accepted_core(data,load_state(path,data),cid)
    save_claim_review(path,data,cid,'CONFIRMED_NO_CORE')
    assert claim_progress(data,load_state(path,data),cid)['confirmed']
