"""Read-only quality check for the v0.6.1a Chinese NER review workspace."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.ner_annotation import load_candidates, load_state, review_summary, export_check

DATA=ROOT/'data/annotations/cfever_zh_review_candidates_v061a.json'
STATE=ROOT/'outputs/ner_review_v061a/decisions.json'


def main():
    data=load_candidates(DATA)
    state=load_state(STATE,data)
    summary=review_summary(data,state)
    print('=== VERIGRAPH AI v0.6.1a — REVIEW STATUS ===')
    for k in ('claims','candidates','decided','pending','needs_context','claims_confirmed','manual_additions'):
        print(f'{k}: {summary[k]}')
    print('AI recommendations are NOT automatically human-approved.')
    print('Annotation source digest:',data['source_digest'][:16])
    print('Decision log:',str(STATE))
    print('Export blocking issues:',len(export_check(data,state)))
    return 0


if __name__=='__main__':
    sys.exit(main())
