"""Coverage metrics for a reviewed v0.4.2 bundle.

Measures *review progress* and *identity-link coverage* only. Neither number says
anything about whether a claim is true.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

QID_OK = "Q"


def _pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 1) if whole else 0.0


def coverage_report(bundle: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(bundle, dict) or bundle.get("schema_version") != "0.4.2":
        raise ValueError("Berkas harus berupa bundle KG v0.4.2.")
    mentions = bundle.get("mention_reviews")
    if not isinstance(mentions, list):
        raise ValueError("Bundle tidak memuat mention_reviews.")
    decisions = Counter(m.get("decision", "pending") for m in mentions)
    total = len(mentions)
    decided = total - decisions.get("pending", 0)
    accepted = decisions.get("accepted_mention", 0)
    with_qid = sum(1 for m in mentions
                   if m.get("decision") == "accepted_mention" and str(m.get("qid", "")).startswith(QID_OK))
    claims = {m.get("claim_id") for m in mentions}
    claims_with_accepted = {m.get("claim_id") for m in mentions if m.get("decision") == "accepted_mention"}
    by_label = Counter(m.get("ner_label", "unknown") for m in mentions)
    return {
        "claims": len(claims),
        "mentions": total,
        "decisions": dict(decisions),
        "review_completion_pct": _pct(decided, total),
        "accepted_mentions": accepted,
        "accepted_with_qid": with_qid,
        "qid_coverage_of_accepted_pct": _pct(with_qid, accepted),
        "claims_with_accepted_mention_pct": _pct(len(claims_with_accepted), len(claims)),
        "mentions_by_ner_label": dict(by_label),
        "note": "Review and linking coverage only; not evidence that any claim is true.",
    }
