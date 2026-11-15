"""Read-only, targeted quality audit for VeriGraph Chinese NER decisions."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ner_annotation import load_candidates, load_state, review_summary
from src.ner_annotation_qa import audit_dataset

DATA = ROOT / 'data/annotations/cfever_zh_review_candidates_v061a.json'
DECISIONS = ROOT / 'outputs/ner_review_v061a/decisions.json'
OUTPUT = ROOT / 'outputs/ner_review_v061a/quality_flags_v061e.json'


def run():
    data = load_candidates(DATA)
    state = load_state(DECISIONS, data)
    summary = review_summary(data, state)
    findings = audit_dataset(data, state)
    print('=== VERIGRAPH AI v0.6.1e — READ-ONLY QA ===')
    print(f"Klaim selesai: {summary['claims_confirmed']} / {summary['claims']}")
    print(f"Keputusan final: {summary['decided']} / {summary['candidates']}")
    print('Temuan QA:', len(findings))
    for item in findings:
        print(f"- {item['claim_id']} | {item['code']} | {item['message']}")
    print('File keputusan TIDAK diubah.')
    # No write by default. The stdout report is enough for this pilot.


if __name__ == '__main__':
    run()
