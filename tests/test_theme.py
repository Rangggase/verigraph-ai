"""Design tokens must meet WCAG AA and the CSS must keep accessibility guards."""
import pytest

from src.theme import CONTRAST_PAIRS, GLOBAL_CSS, TOKENS


def _lum(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    chans = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in chans]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def ratio(fg: str, bg: str) -> float:
    hi, lo = sorted((_lum(fg), _lum(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("fg,bg,minimum", CONTRAST_PAIRS)
def test_contrast_pairs_meet_wcag_aa(fg, bg, minimum):
    assert ratio(TOKENS[fg], TOKENS[bg]) >= minimum


def test_css_has_accessibility_guards():
    assert "focus-visible" in GLOBAL_CSS
    assert "prefers-reduced-motion" in GLOBAL_CSS
    assert "--vg-target:44px" in GLOBAL_CSS


def test_tokens_are_valid_hex():
    assert all(len(v) == 7 and v.startswith("#") for v in TOKENS.values())
