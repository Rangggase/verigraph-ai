"""Regression tests for the beginner UI's non-Streamlit logic."""
from pathlib import Path

from src.ner_annotation import (
    load_candidates, load_state, save_candidate_decision, save_claim_review,
)
from src.ner_easy_helpers import (
    candidate_counts_for_claim, find_surface_positions, first_unresolved,
)

ROOT=Path(__file__).resolve().parents[1]


def test_exact_mandarin_substring_search():
    text='北京在北京，北京'
    assert find_surface_positions(text,'北京')==[(0,2),(3,5),(6,8)]
    assert find_surface_positions(text,'上海')==[]
    assert find_surface_positions(text,'')==[]
    assert find_surface_positions(text,' 北京')==[]


def test_overlapping_repeated_strings():
    assert find_surface_positions('aaaa','aa')==[(0,2),(1,3),(2,4)]


def test_counts_on_real_50_claim_snapshot(tmp_path):
    data=load_candidates(ROOT/'data/annotations/cfever_zh_review_candidates_v061a.json')
    state=load_state(tmp_path/'d.json',data)
    claim=data['claims'][0]['claim_id']
    counts=candidate_counts_for_claim(claim,data,state)
    assert counts['total']>=1
    assert counts['todo']==counts['total']
    assert counts['confirmed'] is False


def test_save_then_resume_preserves_existing_decision(tmp_path):
    data=load_candidates(ROOT/'data/annotations/cfever_zh_review_candidates_v061a.json')
    path=tmp_path/'d.json'
    row=data['candidates'][0]
    save_candidate_decision(path,data,row['candidate_id'],'ACCEPT',
                            start=row['start'],end=row['end'],label=row['label'])
    state=load_state(path,data)
    assert state['decisions'][row['candidate_id']]['action']=='ACCEPT'
    assert candidate_counts_for_claim(row['claim_id'],data,state)['todo'] < candidate_counts_for_claim(row['claim_id'],data,load_state(tmp_path/'empty.json',data))['todo']
    assert len(state['events'])==1


def test_find_next_unresolved_respects_saved_decisions():
    rows=[{'candidate_id':'one'},{'candidate_id':'two'},{'candidate_id':'three'}]
    state={'decisions':{'one':{'action':'ACCEPT'},'two':{'action':'NEEDS_CONTEXT'},'three':{'action':'REJECT'}}}
    assert first_unresolved(rows,state,'one')=='two'
    state['decisions']['two']['action']='REJECT'
    assert first_unresolved(rows,state,'one') is None


def test_confirmed_claim_can_be_reopened_after_edit(tmp_path):
    data=load_candidates(ROOT/'data/annotations/cfever_zh_review_candidates_v061a.json')
    path=tmp_path/'d.json'
    first=data['claims'][0]['claim_id']
    rows=[r for r in data['candidates'] if r['claim_id']==first]
    for row in rows:
        save_candidate_decision(path,data,row['candidate_id'],'REJECT')
    save_claim_review(path,data,first,'CONFIRMED_NO_CORE')
    assert candidate_counts_for_claim(first,data,load_state(path,data))['confirmed']
    save_candidate_decision(path,data,rows[0]['candidate_id'],'ACCEPT',
                            start=rows[0]['start'],end=rows[0]['end'],label=rows[0]['label'])
    assert not candidate_counts_for_claim(first,data,load_state(path,data))['confirmed']
