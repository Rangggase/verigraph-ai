from unittest.mock import Mock, patch

from src.assertion_graph import build_mention_graph
from src.core import make_graph
from src.entity_linking import local_link, search_wikidata_candidates
from src.ner_pipeline import extract_entities


def test_fake_ner_predictions_do_not_require_model_download():
    model = Mock()
    text = 'Barack Obama visited Paris.'
    model.predict_entities.return_value = [
        {'text': 'Barack Obama', 'label': 'person', 'start': 0, 'end': 12, 'score': 0.93},
        {'text': 'Paris', 'label': 'location', 'start': 21, 'end': 26, 'score': 0.91},
    ]
    ents = extract_entities(text, 'en', model)
    assert [e['text'] for e in ents] == ['Barack Obama', 'Paris']
    assert model.predict_entities.called


def test_local_link_is_exact_and_language_specific():
    graph = make_graph()
    assert local_link('Prancis', 'id', graph)['id'] == 'Q142'
    assert local_link('UNKNOWN', 'id', graph) is None


def test_unresolved_mentions_never_become_trusted_facts():
    ents = [{'text': 'Jakarta', 'label': 'location', 'start': 0, 'end': 7, 'score': 0.9}]
    g = build_mention_graph('Jakarta', 'id', ents, {})
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1
    _, _, edge = next(iter(g.edges(data=True)))
    assert edge['relation'] == 'MENTIONS'
    assert edge['claim_status'] == 'unverified'
    assert not any(attr.get('node_type') == 'linked_entity' for _, attr in g.nodes(data=True))


def test_manually_approved_wikidata_candidate_creates_linked_entity_only():
    ents = [{'text': 'Jakarta', 'label': 'location', 'start': 0, 'end': 7, 'score': 0.9}]
    g = build_mention_graph('Jakarta', 'id', ents,
                            {0: {'id': 'Q3630', 'label': 'Jakarta', 'method': 'manually_approved_wikidata_candidate'}})
    assert 'Q3630' in g
    assert g.nodes['Q3630']['node_type'] == 'linked_entity'
    assert next(iter(g.edges(data=True)))[2]['claim_status'] == 'unverified'


def test_search_candidates_does_not_auto_assign_first_hit():
    with patch('requests.get') as get:
        response = Mock()
        response.json.return_value = {
            'search': [{'id': 'Q3630', 'label': 'Jakarta', 'description': 'capital city of Indonesia'}]
        }
        get.return_value = response
        candidates = search_wikidata_candidates('Jakarta', 'id')
    assert candidates[0]['method'] == 'candidate_requires_review'
    assert candidates[0]['id'] == 'Q3630'


def test_unsupported_language_rejected():
    try:
        extract_entities('test', 'fr', Mock())
        assert False, 'must reject unsupported language'
    except ValueError:
        pass
