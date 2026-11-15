"""VeriGraph design system: CSS tokens + global styles for Streamlit.

Concept: an evidence desk. Cool paper-grey surfaces, deep teal ink for action,
and a dashed amber "belum terverifikasi" marker that echoes the dashed edges of
unverified assertions in the graph views. Every text/background pair below has
been checked for WCAG AA (see tests/test_theme.py).
"""
from __future__ import annotations

# name -> hex. Pairs checked in tests: (text, surface).
TOKENS = {
    "paper": "#F4F6F5",       # page background
    "surface": "#FFFFFF",     # cards / inputs
    "ink": "#14262B",         # primary text           14.4:1 on paper
    "ink-muted": "#4A5B60",   # secondary text          6.5:1 on paper
    "teal": "#0B6B6B",        # primary action / links  5.8:1 on paper
    "teal-ink": "#FFFFFF",    # text on teal            6.3:1
    "line": "#6F8387",        # control borders         3.7:1 (non-text 3:1)
    "hairline": "#D5DDDC",    # decorative dividers only
    "unverified-bg": "#FFF1CC",
    "unverified-ink": "#6B4200",   # 7.8:1 on unverified-bg
    "verified-bg": "#DCF3E3",
    "verified-ink": "#14532D",     # 7.8:1
    "danger": "#9B1C1C",           # 7.5:1 on paper
    "mark": "#F7DB76",             # entity highlight (ink text on it)
}

# (foreground token, background token, minimum ratio)
CONTRAST_PAIRS = [
    ("ink", "paper", 4.5), ("ink", "surface", 4.5), ("ink-muted", "paper", 4.5),
    ("ink-muted", "surface", 4.5), ("teal", "paper", 4.5), ("teal-ink", "teal", 4.5),
    ("unverified-ink", "unverified-bg", 4.5), ("verified-ink", "verified-bg", 4.5),
    ("danger", "paper", 4.5), ("ink", "mark", 4.5), ("line", "paper", 3.0),
]


def _root_css() -> str:
    return ":root{" + ";".join(f"--vg-{k}:{v}" for k, v in TOKENS.items()) + ";" \
        "--vg-space-1:4px;--vg-space-2:8px;--vg-space-3:16px;--vg-space-4:24px;--vg-space-5:40px;" \
        "--vg-radius:10px;--vg-target:44px}"


GLOBAL_CSS = _root_css() + """
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;700&family=Schibsted+Grotesk:wght@600;800&display=swap');
html, body, [class*="css"], .stApp { font-family:'Public Sans', system-ui, sans-serif; color:var(--vg-ink); }
.stApp { background:var(--vg-paper); }
h1, h2, h3, .vg-headline { font-family:'Schibsted Grotesk', 'Public Sans', sans-serif; font-weight:800;
  letter-spacing:-.01em; line-height:1.2; color:var(--vg-ink); text-wrap:balance; }
h1 { font-size:clamp(1.5rem, 2.6vw, 2rem); }
p, li, .stMarkdown { line-height:1.6; max-width:75ch; }
.block-container { padding-top:var(--vg-space-4); padding-bottom:var(--vg-space-5); }

/* Controls: 44px touch targets, visible focus, no hover-only affordances */
.stButton > button, .stDownloadButton > button { min-height:var(--vg-target); border-radius:var(--vg-radius);
  border:1.5px solid var(--vg-line); font-weight:700; background:var(--vg-surface); color:var(--vg-ink);
  transition:background-color .15s ease, border-color .15s ease; }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] { background:var(--vg-teal);
  color:var(--vg-teal-ink); border-color:var(--vg-teal); }
.stButton > button:hover { border-color:var(--vg-teal); }
:is(button, a, input, textarea, select, [role="tab"], [role="radio"]):focus-visible {
  outline:3px solid var(--vg-teal) !important; outline-offset:2px !important; }
.stTabs [role="tab"] { min-height:var(--vg-target); padding-inline:var(--vg-space-3); font-weight:600; color:var(--vg-ink); }
.stTabs [aria-selected="true"] { color:var(--vg-teal); }
[data-baseweb="input"], [data-baseweb="select"] > div, textarea { border-color:var(--vg-line) !important; }
[data-testid="stMetricValue"] { font-family:'Schibsted Grotesk', sans-serif; font-variant-numeric:tabular-nums; }

/* The signature element: honest status markers */
.vg-unverified, .vg-verified { display:inline-block; padding:2px 10px; border-radius:6px; font-size:.85rem; font-weight:700; }
.vg-unverified { background:var(--vg-unverified-bg); color:var(--vg-unverified-ink); border:1.5px dashed var(--vg-unverified-ink); }
.vg-verified { background:var(--vg-verified-bg); color:var(--vg-verified-ink); border:1.5px solid var(--vg-verified-ink); }
.vg-note { border-left:4px solid var(--vg-teal); background:var(--vg-surface); padding:var(--vg-space-3);
  border-radius:0 var(--vg-radius) var(--vg-radius) 0; color:var(--vg-ink); }
.vg-banner { background:var(--vg-unverified-bg); color:var(--vg-unverified-ink); border:1.5px dashed var(--vg-unverified-ink);
  border-radius:var(--vg-radius); padding:var(--vg-space-3); margin:0 0 var(--vg-space-3); font-weight:500; }
.vg-highlight { background:var(--vg-mark); color:var(--vg-ink); border-radius:4px; padding:2px 3px; font-weight:700; }

@media (max-width:650px) { .block-container { padding-inline:16px; } h1 { font-size:1.6rem; } }
@media (prefers-reduced-motion:reduce) { * { transition:none !important; animation:none !important; } }
"""


def apply_global_theme(st) -> None:
    st.markdown(f"<style>{GLOBAL_CSS}</style>", unsafe_allow_html=True)


def unverified_banner(st, text: str) -> None:
    st.markdown(f'<div class="vg-banner" role="note">{text}</div>', unsafe_allow_html=True)
