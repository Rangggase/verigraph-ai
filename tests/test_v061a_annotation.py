"""Regression tests for v0.6.1a review, independent from model weights/Internet."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.ner_annotation import (
    CORE_LABELS, add_manual_mention, all_candidates, export_check,
    export_reviewed_jsonl, load_candidates, load_state, new_state,
    review_summary, save_candidate_decision, save_claim_review,
    validate_span,
)

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"data/annotations/cfever_zh_review_candidates_v061a.json"


@pytest.fixture
def data():
    return load_candidates(SOURCE)


def test_source_contains_50_claims_122_distinct_candidates(data):
    assert len(data['claims'])==50
    assert len(data['candidates'])==122
    assert len({r['candidate_id'] for r in data['candidates']})==122
    assert len({(r['claim_id'],r['start'],r['end'],r['label']) for r in data['candidates']})==122


def test_ai_recommendations_are_never_implicitly_human_decisions(tmp_path,data):
    state=load_state(tmp_path/'decisions.json',data)
    assert not state['decisions']
    assert review_summary(data,state)['pending']==122
    assert review_summary(data,state)['claims_confirmed']==0
    assert len(export_check(data,state))==172


def test_all_candidate_offsets_match_original_claim(data):
    claims={r['claim_id']:r['text'] for r in data['claims']}
    for row in data['candidates']:
        assert validate_span(claims[row['claim_id']],row['start'],row['end'],row['label'])==row['surface']


def test_three_excel_offsets_are_preserved_as_separate_ai_revision(data):
    repaired=[r for r in data['candidates'] if r.get('excel_offset_was_for_ai_span')]
    assert len(repaired)==3
    assert {r['candidate_id'] for r in repaired}=={'main:0014','main:0042','main:0070'}
    assert all(r['ai_offsets'] is not None for r in repaired)


def test_reject_then_revise_and_event_history(tmp_path,data):
    state_path=tmp_path/'decisions.json'
    r=data['candidates'][0]
    save_candidate_decision(state_path,data,r['candidate_id'],'REJECT',note='Deskripsi bukan entitas')
    current=load_state(state_path,data)
    assert current['decisions'][r['candidate_id']]['action']=='REJECT'
    assert current['events'][0]['kind']=='candidate_decision'
    save_candidate_decision(state_path,data,r['candidate_id'],'ACCEPT',
                            start=r['start'],end=r['end'],label=r['label'])
    assert load_state(state_path,data)['decisions'][r['candidate_id']]['surface']==r['surface']
    assert len(load_state(state_path,data)['events'])==2


def test_cannot_accept_changed_span_without_explicit_revise(tmp_path,data):
    r=data['candidates'][0]
    with pytest.raises(ValueError,match='REVISE'):
        save_candidate_decision(tmp_path/'d.json',data,r['candidate_id'],'ACCEPT',
                                start=r['start'],end=r['end']+1,label=r['label'])
    result=save_candidate_decision(tmp_path/'d.json',data,r['candidate_id'],'REVISE',
                                start=r['start'],end=r['end']+1,label=r['label'],note='Manual boundary update')
    assert result['decisions'][r['candidate_id']]['surface'] != r['surface']


def test_invalid_and_unscoped_label_rejected():
    with pytest.raises(ValueError):validate_span('你好世界',3,3,'person')
    with pytest.raises(ValueError):validate_span('你好世界',0,2,'disease')
    assert validate_span('你好世界',0,2,'OTHER:CONCEPT')=='你好'


def test_manual_addition_and_duplicate_rejection(tmp_path,data):
    path=tmp_path/'decisions.json'
    claim=data['claims'][0]
    start=0;end=1;label='person'
    # A distinct new span of the claim, regardless of linguistic validity (this is an IO test).
    save=add_manual_mention(path,data,claim['claim_id'],start,end,label,'Manual check')
    assert len(save['manual_additions'])==1
    assert list(save['decisions'].values())[0]['action']=='ACCEPT'
    with pytest.raises(ValueError,match='sudah tercatat'):
        add_manual_mention(path,data,claim['claim_id'],start,end,label)
    assert review_summary(data,load_state(path,data))['candidates']==123


def test_no_core_claim_must_finish_its_candidates(tmp_path,data):
    path=tmp_path/'decisions.json'
    r=data['candidates'][0]
    with pytest.raises(ValueError,match='belum diputuskan'):
        save_claim_review(path,data,r['claim_id'],'CONFIRMED_NO_CORE')
    rows=[item for item in data['candidates'] if item['claim_id']==r['claim_id']]
    for item in rows:
        save_candidate_decision(path,data,item['candidate_id'],'REJECT')
    state=save_claim_review(path,data,r['claim_id'],'CONFIRMED_NO_CORE')
    assert state['claim_reviews'][r['claim_id']]['action']=='CONFIRMED_NO_CORE'


def test_changes_invalidate_prior_claim_confirmation(tmp_path,data):
    path=tmp_path/'decisions.json'
    r=data['candidates'][0]
    rows=[item for item in data['candidates'] if item['claim_id']==r['claim_id']]
    for item in rows:save_candidate_decision(path,data,item['candidate_id'],'REJECT')
    save_claim_review(path,data,r['claim_id'],'CONFIRMED_NO_CORE')
    assert r['claim_id'] in load_state(path,data)['claim_reviews']
    save_candidate_decision(path,data,r['candidate_id'],'ACCEPT',
                            start=r['start'],end=r['end'],label=r['label'])
    assert r['claim_id'] not in load_state(path,data)['claim_reviews']


def test_snapshot_mismatch_protects_existing_review(tmp_path,data):
    path=tmp_path/'decisions.json'
    r=data['candidates'][0]
    save_candidate_decision(path,data,r['candidate_id'],'REJECT')
    fake=deepcopy(data)
    fake['source_digest']='different-data-snapshot'
    with pytest.raises(ValueError,match='tidak cocok'):
        load_state(path,fake)


def test_export_blocked_until_all_candidates_and_claims_decided(data):
    state=new_state(data)
    with pytest.raises(ValueError,match='belum siap'):
        export_reviewed_jsonl(data,state)
    for c in data['candidates']:
        state['decisions'][c['candidate_id']]={'action':'REJECT','label':None,'start':None,'end':None}
    for c in data['claims']:
        state['claim_reviews'][c['claim_id']]={'action':'CONFIRMED_NO_CORE'}
    assert export_check(data,state)==[]
    payload=export_reviewed_jsonl(data,state)
    records=[json.loads(x) for x in payload.splitlines()]
    assert len(records)==50
    assert records[0]['entities']==[]
    assert all(r['annotation_status']!='gold' for r in records)


def test_export_blocks_overlapping_core_mentions(data):
    state=new_state(data)
    for c in data['candidates']:
        state['decisions'][c['candidate_id']]={'action':'REJECT','label':None,'start':None,'end':None}
    for c in data['claims']:
        state['claim_reviews'][c['claim_id']]={'action':'CONFIRMED_NO_CORE'}
    r=data['candidates'][0]
    clone=deepcopy(r)
    clone['candidate_id']='manual:overlap-test'
    clone['start']=r['start']
    clone['end']=r['end']
    state['manual_additions'][clone['candidate_id']]=clone
    for candidate in [r,clone]:
        state['decisions'][candidate['candidate_id']]={'action':'ACCEPT',
                            'label':candidate['label'],'start':candidate['start'],
                            'end':candidate['end'],'surface':candidate['surface']}
    assert any('tumpang tindih' in s for s in export_check(data,state))
