"""v0.6.0: review edits must survive JSON save, resume and second rebuild."""
import json
from pathlib import Path
from src.reviewed_graph import (
    build_reviewed_graph, load_v041, get_mention_rows, get_relation_rows,
    resume_v042_as_v041,
)
INPUT = Path(__file__).resolve().parent / 'fixtures' / 'batch_kg_v041_v060_reference.json'


def test_edited_review_decisions_roundtrip():
    raw, graph = load_v041(INPUT.read_bytes())
    mentions, relations = get_mention_rows(raw, graph), get_relation_rows(graph)
    next(r for r in mentions if r['surface'] == 'Biden')['decision'] = 'accepted_mention'
    next(r for r in relations if r['predicate'] == 'BOARD_MEMBER_OF')['decision'] = 'accepted_extraction'
    result = build_reviewed_graph(raw, mentions, relations)
    stored = {r['surface']:r for r in result['mention_reviews']}
    assert stored['Biden']['decision'] == 'accepted_mention'
    stored_rel = {r['predicate']:r for r in result['relation_reviews']}
    assert stored_rel['BOARD_MEMBER_OF']['decision'] == 'accepted_extraction'
    assert result['report']['accepted_relation_edges_in_clean'] == 1

    new_raw, new_graph = resume_v042_as_v041(json.dumps(result).encode())
    m2, r2 = get_mention_rows(new_raw, new_graph), get_relation_rows(new_graph)
    assert next(r for r in m2 if r['surface']=='Biden')['decision'] == 'accepted_mention'
    assert next(r for r in r2 if r['predicate']=='BOARD_MEMBER_OF')['decision'] == 'accepted_extraction'
    result2 = build_reviewed_graph(new_raw, m2, r2)
    assert result2['report'] == result['report']


def test_persisted_qid_is_not_fact_verification():
    raw, graph = load_v041(INPUT.read_bytes())
    mentions, relations = get_mention_rows(raw, graph), get_relation_rows(graph)
    row = next(r for r in mentions if r['surface'] == 'Hunter Biden')
    row['qid'] = 'Q999888'
    row['canonical_label'] = 'human-reviewed hypothetical identity'
    result = build_reviewed_graph(raw, mentions, relations)
    assert next(r for r in result['mention_reviews'] if r['surface']=='Hunter Biden')['qid']=='Q999888'
    assert result['clean_graph']['graph']['truth_status']=='unverified'
