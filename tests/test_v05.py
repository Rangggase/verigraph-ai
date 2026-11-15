"""v0.5 data leakage protection and reproducibility tests."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.classification_data import load_examples, prepare_split, sanitize_training, verify_disjoint
from scripts.train_ml_baseline import train


def test_cfever_normalizes_uppercase_nei(tmp_path):
    path=tmp_path/'d.jsonl'
    path.write_text('{"id":1,"claim":"北京是城市","label":"NOT ENOUGH INFO","evidence":"gold"}\n',encoding='utf8')
    rows=load_examples(path,'cfever')
    assert rows == [{'source_id':'1','claim':'北京是城市','label':'not enough info'}]
    assert 'evidence' not in rows[0]


def test_label_harmonization_does_not_cross_datasets(tmp_path):
    path=tmp_path/'d.jsonl'
    path.write_text('{"claim":"北京是城市","label":"supports"}\n',encoding='utf8')
    with pytest.raises(ValueError,match='invalid/missing label'):
        load_examples(path,'averitec')


def test_training_dedupe_drops_conflicting_labels():
    rows=[{'claim':'Apple has offices','label':'Supported'},
          {'claim':' Apple   has offices ','label':'Supported'},
          {'claim':'London is in UK','label':'Refuted'},
          {'claim':'London is in UK','label':'Supported'}]
    clean,report=sanitize_training(rows)
    assert len(clean)==1
    assert report['train_identical_duplicates_removed']==1
    assert report['train_conflicting_rows_removed']==2


def test_split_overlap_is_detected():
    with pytest.raises(ValueError,match='Data leakage'):
        verify_disjoint([{'claim':'A  B','label':'Supported'}],[{'claim':' a b ','label':'Supported'}])


def test_pilot_baseline_has_no_evidence_features(tmp_path):
    inp=tmp_path/'train.csv'
    labels=['Supported','Refuted','Not Enough Evidence','Conflicting Evidence/Cherrypicking']
    with inp.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['claim','label','justification','evidence_json'])
        w.writeheader()
        for i in range(40):
            w.writerow({'claim':f'Novel synthetic statement number {i} about Earth and apples',
                        'label':labels[i%4],'justification':'LEAK_SECRET'+str(i),
                        'evidence_json':'LEAK_SECRET'})
    out=tmp_path/'out'
    report=train(SimpleNamespace(dataset='averitec',train=str(inp),dev=None,pilot=True,
                  seed=42,max_train=None,max_dev=None,output=str(out)))
    assert report['data']['mode']=='PILOT_NOT_BENCHMARK'
    assert report['feature_columns']==['claim']
    assert report['data']['normalized_claim_overlap']==0
    assert 'LEAK_SECRET' not in (out/'metrics.json').read_text(encoding='utf8')
    assert (out/'model.joblib').is_file()


def test_reviewed_graph_preserves_current_export():
    p=Path(__file__).resolve().parents[1]/'outputs'/'batch_kg_v042.json'
    data=json.loads(p.read_text(encoding='utf8'))
    assert data['validation']['passed']
    assert data['report']['mention_decisions']['accepted_mention']==5
    assert data['report']['clean_edges']==5
    assert data['report']['canonical_qids']==0
