"""VeriGraph AI v0.6.1a: auditable human NER review for a 50-claim CFEVER pilot.

All suggestions from the source workbook remain AI *suggestions* until a human
explicitly saves a decision. The review is claim-text-only and does not verify
claim truth or Wikidata identities.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import os
import tempfile
import uuid

CORE_LABELS = frozenset(("person", "organization", "location", "event"))
ACTIONS = ("PENDING", "ACCEPT", "REVISE", "REJECT", "OUT_OF_SCOPE", "NEEDS_CONTEXT")
CLAIM_ACTIONS = ("PENDING", "CONFIRMED_COMPLETE", "CONFIRMED_NO_CORE", "NEEDS_CONTEXT")
DATA_SCHEMA = "verigraph-ner-review-candidates-v061a"
STATE_SCHEMA = "verigraph-ner-review-decisions-v061a"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_json(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def validate_span(text: str, start: int, end: int, label: str) -> str:
    if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool):
        raise ValueError("Start dan end harus bilangan bulat (offset 0-based).")
    if not 0 <= start < end <= len(text):
        raise ValueError(f"Offset tidak valid: [{start}:{end}] untuk panjang klaim {len(text)}.")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Tipe entitas wajib diisi.")
    if label != label.strip():
        raise ValueError("Tipe entitas tidak boleh mengandung spasi di tepi.")
    if label not in CORE_LABELS and not label.startswith(("OTHER:", "AMBIGUOUS:")):
        raise ValueError("Tipe di luar empat kategori inti harus berawalan OTHER: atau AMBIGUOUS:.")
    return text[start:end]


def load_candidates(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != DATA_SCHEMA:
        raise ValueError("Skema kandidat tidak kompatibel.")
    claims = data.get("claims", [])
    candidates = data.get("candidates", [])
    if len(claims) != 50 or len(candidates) != 122:
        raise ValueError("Snapshot pilot memerlukan 50 klaim dan 122 kandidat.")
    claim_by_id = {r["claim_id"]:r["text"] for r in claims}
    if len(claim_by_id) != len(claims):
        raise ValueError("Claim ID duplikat pada snapshot.")
    ids = set()
    spans = set()
    for row in candidates:
        cid = row["candidate_id"]
        if cid in ids:
            raise ValueError(f"Candidate ID duplikat: {cid}")
        ids.add(cid)
        if row["claim_id"] not in claim_by_id:
            raise ValueError(f"Candidate mengacu pada klaim yang tidak tersedia: {cid}")
        text = claim_by_id[row["claim_id"]]
        observed = validate_span(text, row["start"], row["end"], row["label"])
        if observed != row["surface"]:
            raise ValueError(f"Surface vs offset tidak cocok: {cid}")
        key = (row["claim_id"], row["start"], row["end"], row["label"])
        if key in spans:
            raise ValueError(f"Kandidat duplikat berdasarkan claim/offset/label: {cid}")
        spans.add(key)
    data["source_digest"] = sha256(_canonical_json({k:v for k,v in data.items() if k!="source_digest"})).hexdigest()
    return data


def new_state(data: dict) -> dict:
    return {
        "schema_version":STATE_SCHEMA,
        "source_digest":data["source_digest"],
        "created_at":utc_now(),
        "updated_at":utc_now(),
        "revision":0,
        "decisions":{},
        "manual_additions":{},
        "claim_reviews":{},
        "events":[],
        "notice":"Human decisions are separate from AI recommendations. Not a verified claim or a finalized gold dataset.",
    }


def load_state(path: Path, data: dict) -> dict:
    path = Path(path)
    if not path.exists():
        return new_state(data)
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != STATE_SCHEMA or state.get("source_digest") != data["source_digest"]:
        raise ValueError("File keputusan tidak cocok dengan snapshot kandidat. JANGAN digabung otomatis; backup dan audit dulu.")
    if not isinstance(state.get("decisions"), dict) or not isinstance(state.get("events"), list):
        raise ValueError("File state tidak memiliki struktur yang valid.")
    return state


def _atomic_save(path: Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = utc_now()
    state["revision"] = state.get("revision", 0) + 1
    # Save to the same directory and atomically rename, protecting from partial files.
    fd, tmp = tempfile.mkstemp(prefix=".v061a_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _record_event(state: dict, kind: str, target: str, previous: Any, current: Any) -> None:
    state["events"].append({"at":utc_now(),"kind":kind,"target":target,"before":previous,"after":current})


def all_candidates(data: dict, state: dict) -> list[dict]:
    return data["candidates"] + list(state["manual_additions"].values())


def save_candidate_decision(path: Path, data: dict, candidate_id: str, action: str,
                            start: int | None = None, end: int | None = None,
                            label: str | None = None, note: str = "") -> dict:
    if action not in ACTIONS:
        raise ValueError("Aksi keputusan tidak dikenal.")
    state = load_state(path, data)
    candidate_by_id = {x["candidate_id"]:x for x in all_candidates(data,state)}
    if candidate_id not in candidate_by_id:
        raise ValueError("Candidate ID tidak ditemukan.")
    candidate = candidate_by_id[candidate_id]
    cid = candidate["claim_id"]
    claim = next(row["text"] for row in data["claims"] if row["claim_id"] == cid)
    if action in ("ACCEPT", "REVISE", "OUT_OF_SCOPE"):
        if start is None or end is None or label is None:
            raise ValueError("Keputusan ini memerlukan offset dan tipe final.")
        surface = validate_span(claim, start, end, label)
        if action == "ACCEPT" and (start,end,label) != (candidate["start"],candidate["end"],candidate["label"]):
            raise ValueError("Span/label berbeda dari kandidat awal; gunakan REVISE.")
        if action == "OUT_OF_SCOPE" and label in CORE_LABELS:
            raise ValueError("OUT_OF_SCOPE wajib memakai tipe OTHER:* atau AMBIGUOUS:*.")
    else:
        start,end,label,surface = None,None,None,None
    updated = {"action":action,"start":start,"end":end,"label":label,"surface":surface,
               "note":str(note).strip(),"reviewed_at":utc_now()}
    old = state["decisions"].get(candidate_id)
    state["decisions"][candidate_id] = updated
    _record_event(state,"candidate_decision",candidate_id,old,updated)
    old_claim = state["claim_reviews"].pop(cid,None)
    if old_claim is not None:
        _record_event(state,"claim_review_invalidated",cid,old_claim,None)
    _atomic_save(path,state)
    return state


def add_manual_mention(path: Path, data: dict, claim_id: str, start: int, end: int,
                       label: str, note: str = "") -> dict:
    state = load_state(path,data)
    claims={r["claim_id"]:r["text"] for r in data["claims"]}
    if claim_id not in claims:
        raise ValueError("Claim ID tidak tersedia.")
    surface=validate_span(claims[claim_id],start,end,label)
    for r in all_candidates(data,state):
        if (r["claim_id"],r["start"],r["end"],r["label"]) == (claim_id,start,end,label):
            raise ValueError(f"Entitas ini sudah tercatat: {r['candidate_id']}")
    new_id="manual:"+uuid.uuid4().hex
    candidate={"candidate_id":new_id,"claim_id":claim_id,"surface":surface,"label":label,
               "start":start,"end":end,"origin":"USER_ADDED", "priority":"MANUAL",
               "source_sheet":None,"source_excel_row":None,"ai_recommendation":"NONE",
               "ai_span":"","ai_label":"","ai_offsets":None,"ai_note":""}
    state["manual_additions"][new_id]=candidate
    action="ACCEPT" if label in CORE_LABELS else "OUT_OF_SCOPE"
    state["decisions"][new_id]={"action":action,"start":start,"end":end,
                              "label":label,"surface":surface,"note":str(note).strip(),
                              "reviewed_at":utc_now()}
    _record_event(state,"manual_addition",new_id,None,candidate)
    old_claim=state["claim_reviews"].pop(claim_id,None)
    if old_claim is not None:
        _record_event(state,"claim_review_invalidated",claim_id,old_claim,None)
    _atomic_save(path,state)
    return state


def save_claim_review(path: Path, data: dict, claim_id: str, action: str, note: str="") -> dict:
    if action not in CLAIM_ACTIONS:
        raise ValueError("Keputusan klaim tidak dikenal.")
    if claim_id not in {r["claim_id"] for r in data["claims"]}:
        raise ValueError("Claim ID tidak tersedia.")
    state=load_state(path,data)
    rows=[r for r in all_candidates(data,state) if r["claim_id"]==claim_id]
    if action in ("CONFIRMED_COMPLETE","CONFIRMED_NO_CORE"):
        pending=[r["candidate_id"] for r in rows
                 if state["decisions"].get(r["candidate_id"],{}).get("action","PENDING")
                 in ("PENDING","NEEDS_CONTEXT")]
        if pending:
            raise ValueError(f"Masih ada {len(pending)} kandidat belum diputuskan pada klaim ini.")
        accepted_core=[r for r in rows if state["decisions"].get(r["candidate_id"],{}).get("action") in ("ACCEPT","REVISE")
                       and state["decisions"][r["candidate_id"]]["label"] in CORE_LABELS]
        if action=="CONFIRMED_NO_CORE" and accepted_core:
            raise ValueError("Tidak boleh menandai NO_CORE jika ada entitas inti yang diterima.")
        if action=="CONFIRMED_COMPLETE" and not accepted_core:
            raise ValueError("Jika tidak ada entitas inti, pilih CONFIRMED_NO_CORE.")
        # A final check applies to both beginner and technical interfaces.
        # Import here to avoid a circular import at module load time.
        from src.ner_annotation_qa import audit_claim
        problems = [r for r in audit_claim(data, state, claim_id) if r['severity'] == 'block']
        if problems:
            raise ValueError('QA anotasi belum selesai: ' + problems[0]['message'])
    old=state["claim_reviews"].get(claim_id)
    updated={"action":action,"note":str(note).strip(),"reviewed_at":utc_now()}
    state["claim_reviews"][claim_id]=updated
    _record_event(state,"claim_review",claim_id,old,updated)
    _atomic_save(path,state)
    return state


def review_summary(data: dict, state: dict) -> dict:
    rows=all_candidates(data,state)
    actions=Counter(state["decisions"].get(row["candidate_id"],{}).get("action","PENDING") for row in rows)
    claim_actions=Counter(state["claim_reviews"].get(r["claim_id"],{}).get("action","PENDING") for r in data["claims"])
    return {"claims":len(data["claims"]),"candidates":len(rows),
            "decided":len(rows)-actions["PENDING"]-actions["NEEDS_CONTEXT"],
            "pending":actions["PENDING"],"needs_context":actions["NEEDS_CONTEXT"],
            "claims_confirmed":claim_actions["CONFIRMED_COMPLETE"]+claim_actions["CONFIRMED_NO_CORE"],
            "manual_additions":len(state["manual_additions"]),"actions":dict(actions),
            "claim_actions":dict(claim_actions)}


def export_check(data: dict, state: dict) -> list[str]:
    issues=[]
    all_rows=all_candidates(data,state)
    for r in all_rows:
        d=state["decisions"].get(r["candidate_id"],{})
        action=d.get("action","PENDING")
        if action in ("PENDING","NEEDS_CONTEXT"):
            issues.append(f"Kandidat {r['candidate_id']} belum final ({action}).")
    for claim in data["claims"]:
        cid=claim["claim_id"]
        if state["claim_reviews"].get(cid,{}).get("action") not in ("CONFIRMED_COMPLETE","CONFIRMED_NO_CORE"):
            issues.append(f"Klaim {cid} belum dikonfirmasi lengkap/tanpa entitas inti.")
    for claim in data["claims"]:
        cid=claim["claim_id"]
        spans=[]
        for r in all_rows:
            if r["claim_id"]!=cid:continue
            d=state["decisions"].get(r["candidate_id"],{})
            if d.get("action") in ("ACCEPT","REVISE") and d.get("label") in CORE_LABELS:
                try:
                    surface=validate_span(claim["text"],d["start"],d["end"],d["label"])
                    if d.get("surface")!=surface:
                        issues.append(f"Surface/offset hasil keputusan tidak cocok: {r['candidate_id']}.")
                    spans.append((d["start"],d["end"],d["label"],r["candidate_id"]))
                except (ValueError,TypeError,KeyError) as exc:
                    issues.append(f"Offset hasil keputusan tidak valid: {r['candidate_id']}: {exc}")
        spans.sort()
        for i,a in enumerate(spans):
            for b in spans[i+1:]:
                if b[0]>=a[1]:break
                issues.append(f"Entitas inti tumpang tindih {cid}: {a[3]} dan {b[3]}.")
    return issues


def export_reviewed_jsonl(data: dict,state:dict) -> str:
    issues=export_check(data,state)
    if issues:
        raise ValueError(f"Review belum siap diekspor: {len(issues)} masalah. Contoh: {issues[0]}")
    rows=all_candidates(data,state)
    lines=[]
    for claim in data["claims"]:
        entities=[]
        for r in rows:
            if r["claim_id"]!=claim["claim_id"]:continue
            d=state["decisions"].get(r["candidate_id"],{})
            if d.get("action") in ("ACCEPT","REVISE") and d.get("label") in CORE_LABELS:
                entities.append({"text":d["surface"],"label":d["label"],
                                "start":d["start"],"end":d["end"],
                                "candidate_id":r["candidate_id"]})
        entities.sort(key=lambda e:(e["start"],e["end"],e["label"]))
        lines.append(json.dumps({"claim_id":claim["claim_id"],"text":claim["text"],
                                 "language":"zh","source_split":"train", "entities":entities,
                                 "annotation_status":"reviewed_for_evaluation_pending_independent_audit",
                                 "claim_truth_status":"unverified"},ensure_ascii=False))
    return "\n".join(lines)+"\n"
