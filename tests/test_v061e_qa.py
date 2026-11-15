"""Safety and consistency checks for the seven-claim NER QA pilot."""
import json
from pathlib import Path

import pytest

from src.ner_annotation import (load_candidates, load_state, new_state, save_candidate_decision,
                                save_claim_review)
from src.ner_annotation_qa import audit_claim, audit_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = load_candidates(ROOT / 'data/annotations/cfever_zh_review_candidates_v061a.json')


def _row(cid):
    return next(r for r in DATA['candidates'] if r['candidate_id'] == cid)


def _set(state, candidate_id, action, surface, label, start, end):
    state['decisions'][candidate_id] = {
        'action': action, 'surface': surface, 'label': label,
        'start': start, 'end': end,
    }


def test_wrong_type_peter_from_audit_flagged():
    state = new_state(DATA)
    _set(state,'main:0002','REVISE','皮特·烏斯蒂諾夫','location',0,8)
    issues = audit_claim(DATA,state,_row('main:0002')['claim_id'])
    assert any(r['code']=='PILOT_LABEL_MISMATCH' and 'main:0002' in r['candidate_ids'] for r in issues)


def test_wrong_type_arctic_from_audit_flagged():
    state = new_state(DATA)
    _set(state,'main:0003','REVISE','北冰洋','organization',0,3)
    issues = audit_claim(DATA,state,_row('main:0003')['claim_id'])
    assert any(r['code']=='PILOT_LABEL_MISMATCH' for r in issues)


def test_elizabeth_rejection_flagged_until_corrected():
    state = new_state(DATA)
    cid = _row('main:0033')['claim_id']
    _set(state,'main:0033','REJECT',None,None,None,None)
    issues = audit_claim(DATA,state,cid)
    assert any(r['code']=='PILOT_MISSING_PERSON' for r in issues)
    claim = next(r['text'] for r in DATA['claims'] if r['claim_id']==cid)
    pos = claim.index('伊麗莎白二世')
    _set(state,'main:0033','REVISE','伊麗莎白二世','person',pos,pos+len('伊麗莎白二世'))
    assert not any(r['code']=='PILOT_MISSING_PERSON' for r in audit_claim(DATA,state,cid))


def test_overlapping_uk_mentioned_twice_is_flagged_and_can_be_fixed():
    state = new_state(DATA)
    cid = _row('main:0034')['claim_id']
    claim = next(r['text'] for r in DATA['claims'] if r['claim_id']==cid)
    positions = []
    index=0
    while (i:=claim.find('英國',index))>=0:
        positions.append(i);index=i+2
    assert len(positions)==2
    for candidate in ('extra:0001','main:0034'):
        _set(state,candidate,'REVISE','英國','location',positions[1],positions[1]+2)
    assert any(r['code']=='OVERLAPPING_CORE' for r in audit_claim(DATA,state,cid))
    _set(state,'main:0034','REVISE','英國','location',positions[0],positions[0]+2)
    assert not any(r['code']=='OVERLAPPING_CORE' for r in audit_claim(DATA,state,cid))


def test_current_real_decisions_report_is_read_only():
    path = ROOT / 'data/annotations/cfever_zh_review_candidates_v061a.json'
    before = path.read_bytes()
    state = new_state(DATA)
    snapshot = json.dumps(state, ensure_ascii=False, sort_keys=True)
    findings = audit_dataset(DATA,state)
    assert isinstance(findings, list)
    assert json.dumps(state, ensure_ascii=False, sort_keys=True) == snapshot
    assert path.read_bytes() == before


def test_backend_rejects_confirmed_wrong_type(tmp_path):
    path = tmp_path/'decisions.json'
    cid = _row('main:0003')['claim_id']
    rows = [r for r in DATA['candidates'] if r['claim_id']==cid]
    assert len(rows)==1
    save_candidate_decision(path,DATA,rows[0]['candidate_id'],'REVISE',
                            start=0,end=3,label='organization')
    with pytest.raises(ValueError,match='QA anotasi'):
        save_claim_review(path,DATA,cid,'CONFIRMED_COMPLETE')
    assert cid not in load_state(path,DATA)['claim_reviews']
    save_candidate_decision(path,DATA,rows[0]['candidate_id'],'REVISE',
                            start=0,end=3,label='location')
    save_claim_review(path,DATA,cid,'CONFIRMED_COMPLETE')
    assert load_state(path,DATA)['claim_reviews'][cid]['action']=='CONFIRMED_COMPLETE'


def test_confirm_unchanged_non_special_claim(tmp_path):
    path = tmp_path/'decisions.json'
    cid=DATA['claims'][0]['claim_id']
    rows=[r for r in DATA['candidates'] if r['claim_id']==cid]
    for r in rows:
        save_candidate_decision(path,DATA,r['candidate_id'],'REJECT')
    save_claim_review(path,DATA,cid,'CONFIRMED_NO_CORE')
    assert load_state(path,DATA)['claim_reviews'][cid]['action']=='CONFIRMED_NO_CORE'
