"""VeriGraph v0.6.1d: simple landing page; technical workflows remain separate."""
from __future__ import annotations

from pathlib import Path

from src.ner_annotation import load_candidates, load_state, review_summary
from src.ner_annotation_ui import render_ner_annotation_tab, _apply_theme


def _show_review():
    import streamlit as st
    st.session_state['vg_easy_page'] = 'Periksa nama'


def render_easy_home(root: Path) -> None:
    import streamlit as st
    _apply_theme()
    page = st.radio('Menu utama', ['Beranda', 'Periksa nama'], horizontal=True,
                    key='vg_easy_page', label_visibility='collapsed')
    if page == 'Periksa nama':
        render_ner_annotation_tab(root, show_mode_toggle=False)
        return

    st.markdown('<div class="vg-kicker">VERIGRAPH AI · MODE MUDAH</div>', unsafe_allow_html=True)
    st.markdown('<div class="vg-headline">Bantu AI mengenali nama dalam kalimat.</div>', unsafe_allow_html=True)
    st.markdown('<div class="vg-subheadline">Tidak perlu memahami kode. Baca kalimat Mandarin, lihat nama yang disorot, lalu pilih jawaban.</div>', unsafe_allow_html=True)

    try:
        data = load_candidates(root / 'data/annotations/cfever_zh_review_candidates_v061a.json')
        state = load_state(root / 'outputs/ner_review_v061a/decisions.json', data)
        summary = review_summary(data, state)
    except Exception as exc:
        st.error(f'Data review tidak dapat dibaca: {exc}')
        return

    # Primary CTA deliberately appears before metrics/instructions.
    st.button('▶ Mulai / Lanjutkan pemeriksaan', type='primary', use_container_width=True,
              on_click=_show_review, key='vg_v2_home_start',
              help='Lanjut ke kalimat yang belum selesai. Progres sebelumnya tetap tersimpan.')

    st.markdown('### Progres kamu')
    a, b, c = st.columns(3)
    a.metric('Nama sudah diperiksa', f"{summary['decided']} / {summary['candidates']}")
    b.metric('Kalimat sudah selesai', f"{summary['claims_confirmed']} / {summary['claims']}")
    c.metric('Masih menunggu', summary['pending'] + summary['needs_context'])
    st.progress(summary['decided'] / max(summary['candidates'], 1))
    st.caption('Jawaban disimpan ke komputer ini ketika tombol keputusan ditekan.')

    with st.expander('Cara menggunakan (3 langkah)'):
        st.write('**1. Baca** kalimat Mandarin dan lihat bagian kuning.')
        st.write('**2. Pilih** Benar, Perbaiki, atau Bukan nama. Kalau ragu, pilih Periksa nanti.')
        st.write('**3. Selesaikan** kalimat setelah semua nama diperiksa dan tidak ada yang terlewat.')
    with st.expander('Apa yang diperiksa aplikasi ini?'):
        st.write('Nama orang, lembaga, lokasi, dan peristiwa. '
                 '**Ini bukan pengecekan kebenaran berita atau klaim.**')
    st.caption('Mode Peneliti di panel kiri tetap menyimpan fitur KG, model, dan audit teknis.')
