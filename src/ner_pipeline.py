"""Optional model-backed multilingual named entity extraction.

This is *entity detection*, not relation extraction or fake-news verification.
The model is only fetched/loaded when this component is used.
"""
from __future__ import annotations

from typing import Any

MODEL_ID = "urchade/gliner_multi-v2.1"
ENTITY_LABELS = ["person", "organization", "location", "event"]
SUPPORTED_LANGUAGES = {"id", "en", "zh"}


def load_ner_model(model_id: str = MODEL_ID) -> Any:
    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise RuntimeError(
            "GLiNER belum terpasang. Jalankan: python -m pip install -r requirements-ner.txt"
        ) from exc
    return GLiNER.from_pretrained(model_id)


def extract_entities(text: str, language: str, model: Any, threshold: float = 0.35) -> list[dict]:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    if not text.strip():
        return []
    if len(text) > 1500:
        raise ValueError("Maksimal 1.500 karakter per input untuk prototipe.")
    predictions = model.predict_entities(text, ENTITY_LABELS, threshold=threshold)
    result = []
    seen = set()
    for ent in predictions:
        surface = str(ent.get("text", "")).strip()
        if not surface:
            continue
        start = int(ent.get("start", -1))
        end = int(ent.get("end", -1))
        label = str(ent.get("label", "unknown"))
        if start < 0 or end > len(text) or end <= start:
            continue
        if (start, end, label) in seen:
            continue
        seen.add((start, end, label))
        result.append({
            "text": text[start:end],
            "label": label,
            "start": start,
            "end": end,
            "score": round(float(ent.get("score", 0)), 4),
        })
    return sorted(result, key=lambda x: (x["start"], x["end"], x["label"]))
