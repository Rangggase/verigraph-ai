"""Sprint 1-2 hardening: model integrity, friendly errors, KG coverage."""
import json

import pytest

from src.errors import friendly_error
from src.kg_coverage import coverage_report
from src.model_integrity import (ModelIntegrityError, load_trusted_model,
                                 sidecar_path, verify_model, write_sidecar)


def test_unsigned_model_is_refused(tmp_path):
    model = tmp_path / "model.joblib"
    model.write_bytes(b"not really a model")
    with pytest.raises(ModelIntegrityError, match="no integrity signature"):
        verify_model(model)


def test_tampered_model_is_refused_before_unpickling(tmp_path):
    model = tmp_path / "model.joblib"
    model.write_bytes(b"original")
    write_sidecar(model)
    verify_model(model)
    model.write_bytes(b"tampered")
    with pytest.raises(ModelIntegrityError, match="does not match"):
        load_trusted_model(model)


def test_signed_model_roundtrip(tmp_path):
    import joblib
    model = tmp_path / "m.joblib"
    joblib.dump({"ok": 1}, model)
    write_sidecar(model)
    assert sidecar_path(model).is_file()
    assert load_trusted_model(model) == {"ok": 1}


def test_missing_model_is_refused(tmp_path):
    with pytest.raises(ModelIntegrityError):
        verify_model(tmp_path / "absent.joblib")


def test_friendly_error_hides_internals():
    secret = RuntimeError(r"boom at C:\Users\secret\path.py line 3")
    message = friendly_error(secret, "Prediksi")
    assert "secret" not in message and "Prediksi gagal" in message


def test_friendly_error_keeps_intentional_validation_messages():
    assert "Offset tidak valid" in friendly_error(ValueError("Offset tidak valid"), "Simpan")


def test_friendly_error_network_and_files():
    assert "jaringan" in friendly_error(ConnectionError("x"), "Cari")
    assert "tidak ditemukan" in friendly_error(FileNotFoundError(r"C:\x"), "Buka")


def _bundle(mentions):
    return {"schema_version": "0.4.2", "mention_reviews": mentions}


def test_coverage_report_counts():
    report = coverage_report(_bundle([
        {"claim_id": "a", "decision": "accepted_mention", "qid": "Q42", "ner_label": "person"},
        {"claim_id": "a", "decision": "accepted_mention", "qid": "", "ner_label": "event"},
        {"claim_id": "b", "decision": "pending", "qid": "", "ner_label": "person"},
        {"claim_id": "b", "decision": "rejected_mention", "qid": "", "ner_label": "location"},
    ]))
    assert report["claims"] == 2 and report["mentions"] == 4
    assert report["review_completion_pct"] == 75.0
    assert report["accepted_with_qid"] == 1
    assert report["qid_coverage_of_accepted_pct"] == 50.0
    assert report["claims_with_accepted_mention_pct"] == 50.0


def test_coverage_report_handles_empty_and_rejects_wrong_schema():
    assert coverage_report(_bundle([]))["review_completion_pct"] == 0.0
    with pytest.raises(ValueError):
        coverage_report({"schema_version": "0.4.1"})
    with pytest.raises(ValueError):
        coverage_report(json.loads('{"schema_version":"0.4.2"}'))
