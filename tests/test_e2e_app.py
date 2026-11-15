"""End-to-end UI flows driven through Streamlit's AppTest (headless, no browser)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
MODE_KEY = "vg_app_mode_v061b"
RESEARCHER = "Mode Peneliti (fitur lengkap)"


def _run(researcher: bool = False) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    if researcher:
        at.radio(key=MODE_KEY).set_value(RESEARCHER).run()
    return at


def test_guided_home_renders_without_errors():
    at = _run()
    assert not at.exception
    assert any("Bantu AI mengenali" in m.value for m in at.markdown)


def test_researcher_mode_shows_all_tabs_and_unverified_banner():
    at = _run(researcher=True)
    assert not at.exception
    assert len(at.tabs) == 8
    assert any("belum terverifikasi" in m.value for m in at.markdown)


def test_curated_kg_flow_supports_known_claim():
    at = _run(researcher=True)
    at.selectbox(key="demo_language").set_value("en").run()
    claim_box = at.selectbox[1]
    claim_box.select("Paris is the capital of France.").run()
    next(b for b in at.button if b.label == "Periksa relasi").click().run()
    assert not at.exception
    assert at.session_state["demo_result"]["status"] == "supported"


def test_curated_kg_flow_flags_wrong_capital():
    at = _run(researcher=True)
    at.selectbox(key="demo_language").set_value("en").run()
    box = at.text_area[0]
    box.set_value("Tokyo is the capital of France.")
    next(b for b in at.button if b.label == "Periksa relasi").click().run()
    assert not at.exception
    assert at.session_state["demo_result"]["status"] == "conflict"


def test_ner_tab_empty_state_is_friendly():
    at = _run(researcher=True)
    assert not at.exception
    assert any("Belum ada ekstraksi" in i.value for i in at.info)


@pytest.mark.parametrize("mode", [False, True])
def test_no_raw_exception_text_in_either_mode(mode):
    at = _run(researcher=mode)
    assert not at.error
