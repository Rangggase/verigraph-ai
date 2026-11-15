"""VeriGraph AI v0.6.1d: compact, accessible guided entity review with explicit decisions.

The v0.6.1a annotation schema, audit log, source data and technical UI stay intact.
No recommendation is automatically approved, and no claim truth is inferred.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.ner_annotation import (
    CORE_LABELS, add_manual_mention, all_candidates, export_check,
    export_reviewed_jsonl, load_candidates, load_state, review_summary,
    save_candidate_decision, save_claim_review,
)
from src.ner_easy_helpers import find_surface_positions
from src.ner_annotation_qa import audit_claim
from src.ner_ui_v2_helpers import (
    DECISION_FRIENDLY, claim_progress, first_unfinished_claim,
    highlighted_html, is_core, next_candidate, focus_candidate, readable_label,
)

STYLE = '''<style>
/* Beginner mode only. Technical review remains a separate path. */
.block-container { max-width: 990px !important; padding-top: .65rem !important; padding-bottom: 2.8rem !important; }
.vg-kicker {font-size:.76rem; font-weight:800; letter-spacing:.1em; color:#27c6b6; margin:.45rem 0 .25rem;}
.vg-headline {font-size:clamp(1.4rem, 2.5vw, 1.9rem); font-weight:800; line-height:1.24; margin:0 0 .22rem;}
.vg-subheadline {font-size:.94rem; color:#b9c4cb; line-height:1.45; margin:0 0 .75rem;}
.vg-sentence {font-size:clamp(1.2rem,2.05vw,1.52rem); line-height:2.05;letter-spacing:.012em;
  padding:19px 23px; background:rgba(100,136,154,.11); border:1px solid rgba(127,169,173,.4);
  border-radius:15px; overflow-wrap:anywhere; margin:.34rem 0 .65rem;}
.vg-highlight {background:#f7db76; color:#1d262d; border-radius:4px; padding:2px 2px;
 box-decoration-break:clone; -webkit-box-decoration-break:clone; font-weight:800;}
.vg-small {color:#b6c3c8; font-size:.90rem; line-height:1.5;}
.vg-status {color:#43d3bf; font-weight:750; font-size:.86rem;}
.vg-question {font-size:1.13rem; font-weight:750; margin:.48rem 0 .35rem;}
.vg-current {font-size:1.08rem;font-weight:750;margin:.15rem 0 .45rem;}
.vg-pill {background:rgba(48,145,142,.15); border:1px solid rgba(60,165,157,.4); border-radius:12px;
 padding:.52rem .72rem; margin-bottom:.65rem; font-size:.91rem;}
/* Only improve touch targets; preserve Streamlit focus rings and keyboard semantics. */
.stButton > button {min-height:48px; border-radius:11px; font-weight:700; font-size:.99rem;}
.stButton > button:focus-visible {outline:3px solid #58d8d0; outline-offset:2px;}
[data-testid="stMetricValue"] {font-size:1.38rem;}
@media (max-width:650px) { .vg-sentence{padding:13px 15px;} .vg-headline{font-size:1.5rem;} }
</style>'''

LABEL_CHOICES = [
    'person','organization','location','event','OTHER:WORK','OTHER:CELESTIAL_BODY',
    'OTHER:AWARD','OTHER:BUILDING','OTHER:RELIGION','OTHER:CONCEPT','OTHER:ETHNIC_GROUP',
    'OTHER:FIELD','OTHER:CHEMICAL','OTHER:FOOD','OTHER:SPECIES','OTHER:TECHNOLOGY',
    'OTHER:INFRASTRUCTURE','OTHER:ROAD','OTHER:PROGRAM','OTHER:MYTHOLOGICAL_FIGURE',
    'OTHER:VEHICLE_MODEL','OTHER:CUISINE','OTHER:OTHER','AMBIGUOUS:PERSON_OR_LOCATION',
]


def _apply_theme() -> None:
    import streamlit as st
    st.markdown(STYLE, unsafe_allow_html=True)


def _labels(*extra: str) -> list[str]:
    choices = list(LABEL_CHOICES)
    for label in extra:
        if label and label not in choices:
            choices.append(label)
    return choices


def _go_claim(cid: str) -> None:
    import streamlit as st
    st.session_state['vg_v2_claim'] = cid
    st.session_state['vg_v2_candidate'] = None
    st.session_state['vg_v2_claim_revision'] = st.session_state.get('vg_v2_claim_revision', 0) + 1
    st.session_state['vg_v2_candidate_revision'] = st.session_state.get('vg_v2_candidate_revision', 0) + 1
    st.session_state['vg_v2_edit'] = None
    st.session_state['vg_v2_review_saved'] = False
    st.rerun()


def _after_candidate_saved(rows: list[dict], state: dict, current_id: str, message: str) -> None:
    import streamlit as st
    st.session_state['vg_v2_candidate'] = next_candidate(rows, state, current_id)
    st.session_state['vg_v2_candidate_revision'] = st.session_state.get('vg_v2_candidate_revision', 0) + 1
    st.session_state['vg_v2_edit'] = None
    st.session_state['vg_v2_review_saved'] = False
    st.session_state['vg_v2_flash'] = message
    st.rerun()


def _save_simple(path: Path, data: dict, rows: list[dict], row: dict, action: str, note: str = '') -> None:
    import streamlit as st
    try:
        kwargs = {'start': row['start'], 'end': row['end'], 'label': row['label']} if action in ('ACCEPT','OUT_OF_SCOPE') else {}
        state = save_candidate_decision(path, data, row['candidate_id'], action, note=note, **kwargs)
        # If the last open candidate was deferred, move to another claim rather
        # than immediately showing the same uncertain name again.
        if action == 'NEEDS_CONTEXT' and next_candidate(rows, state, row['candidate_id']) == row['candidate_id']:
            following = first_unfinished_claim(data, state, after_id=row['claim_id'])
            if following and following != row['claim_id']:
                st.session_state['vg_v2_claim'] = following
                st.session_state['vg_v2_candidate'] = None
                st.session_state['vg_v2_edit'] = None
                st.session_state['vg_v2_review_saved'] = False
                st.session_state['vg_v2_flash'] = 'Nama ditunda. Beralih ke kalimat lain; keputusan belum dianggap final.'
                st.rerun()
                return
        _after_candidate_saved(rows, state, row['candidate_id'], 'Jawaban berhasil disimpan.')
    except Exception as exc:
        st.error(f'Jawaban belum disimpan: {exc}')


def _revision_form(claim: str, row: dict, saved: dict, path: Path, data: dict, rows: list[dict], rev: int) -> None:
    import streamlit as st
    cid = row['candidate_id']
    st.markdown('#### Perbaiki nama atau jenis')
    st.caption('Salin bagian yang tepat dari kalimat di atas. Nama harus muncul sama persis.')
    initial_surface = saved.get('surface') or row['surface']
    initial_label = saved.get('label') or row['label']
    name = st.text_input('Nama yang benar', value=initial_surface, key=f'vg_v2_edit_surface_{cid}_{rev}')
    labels = _labels(row.get('label',''), row.get('ai_label',''), initial_label)
    label = st.selectbox('Nama ini termasuk jenis apa?', labels,
                         index=labels.index(initial_label), format_func=readable_label,
                         key=f'vg_v2_edit_label_{cid}_{rev}')
    matches = find_surface_positions(claim, name)
    if matches and len(matches) > 1:
        occurrence = st.selectbox('Nama muncul lebih dari sekali. Pilih yang dimaksud:',
                                  range(len(matches)), format_func=lambda x:f'Kemunculan {x+1}',
                                  key=f'vg_v2_edit_occ_{cid}_{rev}')
    else:
        occurrence = 0
    if matches:
        start,end = matches[occurrence]
        st.markdown(highlighted_html(claim, start, end), unsafe_allow_html=True)
    elif name:
        st.warning('Nama belum ditemukan persis di kalimat. Periksa penulisan dan tanda baca.')
    if row.get('ai_span') and row['ai_span'] != row['surface']:
        st.caption(f"Contoh usulan AI (hanya referensi): {row['ai_span']}. Periksa sendiri sebelum menyimpan.")
    note = st.text_input('Catatan (opsional)', value=saved.get('note',''), key=f'vg_v2_edit_note_{cid}_{rev}')
    can_save = bool(matches)
    if can_save and (start,end,label)==(row['start'],row['end'],row['label']):
        st.info('Nama dan jenis masih sama dengan usulan awal. Gunakan tombol **Benar** untuk menerimanya.')
        can_save = False
    a,b = st.columns([2,1])
    if a.button('✓ Simpan perbaikan', type='primary', disabled=not can_save, use_container_width=True,
                key=f'vg_v2_save_edit_{cid}'):
        try:
            action = 'REVISE' if is_core(label) else 'OUT_OF_SCOPE'
            state = save_candidate_decision(path, data, cid, action,
                                            start=start, end=end, label=label, note=note)
            _after_candidate_saved(rows, state, cid, 'Perbaikan berhasil disimpan.')
        except Exception as exc:
            st.error(f'Perbaikan belum disimpan: {exc}')
    if b.button('Batal', use_container_width=True, key=f'vg_v2_cancel_edit_{cid}'):
        st.session_state['vg_v2_edit'] = None
        st.rerun()


def _render_guided(root: Path) -> None:
    """Show the sentence and decision controls above the fold.

    More advanced tools are available below in collapsed sections. Human
    decisions continue to use the original v061a auditable JSON schema.
    """
    import streamlit as st
    _apply_theme()
    path = root / 'outputs/ner_review_v061a/decisions.json'
    try:
        data = load_candidates(root / 'data/annotations/cfever_zh_review_candidates_v061a.json')
        state = load_state(path, data)
    except Exception as exc:
        st.error(f'Data review tidak dapat dibuka: {exc}')
        return

    summary = review_summary(data, state)
    claims = data['claims']
    ids = [c['claim_id'] for c in claims]
    if not ids:
        st.warning('Belum ada kalimat untuk diperiksa.')
        return

    selected_id = st.session_state.get('vg_v2_claim')
    if selected_id not in ids:
        selected_id = first_unfinished_claim(data, state) or ids[0]
    st.session_state['vg_v2_claim'] = selected_id
    claim_index = ids.index(selected_id)
    claim = claims[claim_index]['text']
    rows = [r for r in all_candidates(data, state) if r['claim_id'] == selected_id]
    counts = claim_progress(data, state, selected_id)
    quality_flags = audit_claim(data, state, selected_id)

    st.markdown('<div class="vg-kicker">VERIGRAPH AI · PEMERIKSAAN TERPANDU</div>', unsafe_allow_html=True)
    st.markdown('<div class="vg-headline">Periksa nama dalam kalimat</div>', unsafe_allow_html=True)
    st.markdown('<div class="vg-subheadline">Baca bagian kuning, pilih jawaban, lalu lanjut ke nama berikutnya. Ini bukan penilaian benar atau hoaks.</div>', unsafe_allow_html=True)

    a, b = st.columns([2, 1])
    a.markdown(f'<span class="vg-status">Kalimat {claim_index + 1}/{len(ids)} · {counts["decided"]}/{counts["total"]} nama diperiksa</span>', unsafe_allow_html=True)
    b.caption(f'Total proyek: {summary["decided"]}/{summary["candidates"]} nama')
    st.progress(summary['decided'] / max(summary['candidates'], 1), text='Progres tersimpan setelah setiap keputusan')

    flash = st.session_state.pop('vg_v2_flash', None)
    if flash:
        st.success(flash)
    if quality_flags:
        st.warning(f'Pemeriksaan kualitas: {len(quality_flags)} hal perlu diperiksa pada kalimat ini.')
        with st.expander('Lihat temuan dan cara memperbaikinya'):
            for flag in quality_flags:
                st.write('• ' + flag['message'])
            st.caption('Peringatan adalah aturan QA pada pilot, bukan penilaian kebenaran klaim.')

    row_ids = [r['candidate_id'] for r in rows]
    current_row = focus_candidate(rows, state,
        requested=st.session_state.get('vg_v2_candidate'),
        review_saved=st.session_state.get('vg_v2_review_saved', False))
    selected_candidate = current_row['candidate_id'] if current_row else None
    st.markdown(highlighted_html(claim, current_row['start'], current_row['end'])
                if current_row else highlighted_html(claim), unsafe_allow_html=True)

    if current_row is None:
        if counts['confirmed']:
            st.success('Kalimat ini sudah selesai. Kamu bisa membuka kalimat lain di bawah.')
        elif counts['waiting'] == 0:
            st.success('Semua nama sudah diperiksa. Sekarang pastikan tidak ada nama yang terlewat.')
        else:
            st.info('Belum ada nama yang dipilih. Buka daftar nama pada bagian navigasi.')
    else:
        cid = current_row['candidate_id']
        saved = state['decisions'].get(cid, {})
        action = saved.get('action', 'PENDING')
        st.markdown('<div class="vg-question">Apakah bagian kuning sudah tepat sebagai nama entitas?</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="vg-pill">{_esc(current_row["surface"])} &nbsp; · &nbsp; {_esc(readable_label(current_row["label"]))}</div>', unsafe_allow_html=True)
        if action != 'PENDING':
            st.caption('Keputusan sebelumnya: '+ DECISION_FRIENDLY.get(action, action) + '. Kamu boleh mengubahnya.')
        elif current_row['label'] in CORE_LABELS:
            st.caption('"Benar" berarti **nama yang disorot DAN jenisnya** sudah tepat, bukan bahwa klaimnya benar.')
        else:
            st.caption('Jenis ini berada di luar empat kategori nama utama atau belum pasti. Periksa sebelum memilih.')

        if current_row['label'] in CORE_LABELS:
            a, b, c = st.columns(3, gap='small')
            if a.button('✓ Benar', type='primary', use_container_width=True, key=f'vg_v2_accept_{cid}',
                        help='Nama dan kategori sesuai. Tidak memvalidasi kebenaran kalimat.'):
                _save_simple(path, data, rows, current_row, 'ACCEPT')
            if b.button('✎ Perbaiki', use_container_width=True, key=f'vg_v2_fix_{cid}',
                        help='Nama terlalu panjang, terlalu pendek, atau kategorinya keliru.'):
                st.session_state['vg_v2_edit'] = cid
                st.rerun()
            if c.button('✕ Bukan nama', use_container_width=True, key=f'vg_v2_reject_{cid}',
                        help='Potongan ini bukan entitas bernama dalam skema anotasi.'):
                _save_simple(path, data, rows, current_row, 'REJECT')
        elif current_row['label'].startswith('OTHER:'):
            a, b, c = st.columns(3, gap='small')
            if a.button('✓ Kategori lain', type='primary', use_container_width=True, key=f'vg_v2_other_{cid}',
                        help='Simpan nama ini tetapi kecualikan dari evaluasi empat kategori inti.'):
                _save_simple(path, data, rows, current_row, 'OUT_OF_SCOPE')
            if b.button('✎ Perbaiki', use_container_width=True, key=f'vg_v2_fix_{cid}'):
                st.session_state['vg_v2_edit'] = cid
                st.rerun()
            if c.button('✕ Bukan nama', use_container_width=True, key=f'vg_v2_reject_{cid}'):
                _save_simple(path, data, rows, current_row, 'REJECT')
        else:
            a, b = st.columns(2, gap='small')
            if a.button('✎ Tentukan jenis', type='primary', use_container_width=True, key=f'vg_v2_fix_{cid}'):
                st.session_state['vg_v2_edit'] = cid
                st.rerun()
            if b.button('✕ Bukan nama', use_container_width=True, key=f'vg_v2_reject_{cid}'):
                _save_simple(path, data, rows, current_row, 'REJECT')
        if st.session_state.get('vg_v2_edit') == cid:
            with st.container(border=True):
                _revision_form(claim, current_row, saved, path, data, rows, state.get('revision', 0))
        if st.button('⏱ Belum yakin — periksa nanti', key=f'vg_v2_defer_{cid}',
                     help='Tunda tanpa menganggap anotasi sudah selesai.'):
            _save_simple(path, data, rows, current_row, 'NEEDS_CONTEXT')
        with st.expander('Mengapa AI memilih nama ini? · Bantuan'):
            st.write('**Benar:** hanya untuk nama dan jenis yang tepat. **Perbaiki:** ubah potongan nama atau jenis. '
                     '**Bukan nama:** prediksi bukan entitas bernama. **Periksa nanti:** belum diputuskan.')
            ai = current_row.get('ai_recommendation', 'NONE')
            st.caption('Saran AI: ' + _recommendation_friendly(ai) + '. Saran AI tidak otomatis disetujui.')
            if current_row.get('ai_note'):
                st.caption('Alasan AI: ' + current_row['ai_note'])

    # Put claim completion immediately after the last candidate, not buried
    # beneath navigation or technical controls.
    if counts['waiting'] == 0 and not counts['confirmed']:
        st.markdown('#### Selesaikan kalimat')
        if quality_flags:
            st.info('Perbaiki dahulu temuan QA di atas sebelum menyelesaikan kalimat.')
        check = st.checkbox('Saya sudah membaca ulang kalimat dan memeriksa apakah ada nama yang terlewat.',
                            key=f'vg_v2_confirm_{selected_id}')
        if st.button('✓ Tandai kalimat selesai', disabled=not check or bool(quality_flags), type='primary',
                     use_container_width=True, key=f'vg_v2_done_{selected_id}'):
            try:
                fresh = load_state(path, data)
                action = 'CONFIRMED_COMPLETE' if _has_accepted_core(data, fresh, selected_id) else 'CONFIRMED_NO_CORE'
                save_claim_review(path, data, selected_id, action)
                fresh = load_state(path, data)
                following = first_unfinished_claim(data, fresh, after_id=selected_id)
                st.session_state['vg_v2_flash'] = 'Kalimat selesai dan tersimpan.'
                if following is not None and following != selected_id:
                    st.session_state['vg_v2_claim'] = following
                st.session_state['vg_v2_candidate'] = None
                st.session_state['vg_v2_review_saved'] = False
                st.session_state['vg_v2_candidate_revision'] = st.session_state.get('vg_v2_candidate_revision', 0) + 1
                st.rerun()
            except Exception as exc:
                st.error(f'Kalimat belum dapat diselesaikan: {exc}')
    elif counts['waiting'] > 0 and current_row is None:
        st.info(f'Masih ada {counts["waiting"]} nama menunggu keputusan.')

    # These are useful but not mandatory for each decision: keep them collapsed.
    with st.expander('＋ Tambah nama yang terlewat'):
        _add_missing_form(claim, selected_id, data, path)
    with st.expander('Lihat nama lain dan ubah keputusan lama'):
        if rows:
            nav = st.session_state.get('vg_v2_candidate_revision', 0)
            chosen = st.selectbox('Pilih nama dalam kalimat', row_ids,
                index=row_ids.index(selected_candidate) if selected_candidate in row_ids else 0,
                format_func=lambda choice: (
                    next(r['surface'] for r in rows if r['candidate_id']==choice) + ' · ' +
                    DECISION_FRIENDLY.get(state['decisions'].get(choice, {}).get('action', 'PENDING'), 'Belum diperiksa')),
                key=f'vg_v2_pick_{selected_id}_{nav}')
            if st.button('Buka nama terpilih', key=f'vg_v2_open_candidate_{selected_id}'):
                st.session_state['vg_v2_candidate'] = chosen
                st.session_state['vg_v2_review_saved'] = True
                st.session_state['vg_v2_edit'] = None
                st.rerun()
        else:
            st.caption('Tidak ada nama dari hasil AI. Gunakan Tambah nama jika ada yang terlewat.')

    st.divider()
    previous, following = st.columns(2)
    if previous.button('← Kalimat sebelumnya', disabled=claim_index==0,
                       use_container_width=True, key='vg_v2_prev_claim'):
        _go_claim(ids[claim_index - 1])
    if following.button('Kalimat berikutnya →', disabled=claim_index==len(ids)-1,
                        use_container_width=True, key='vg_v2_next_claim'):
        _go_claim(ids[claim_index + 1])
    with st.expander('Cari kalimat tertentu atau lanjutkan yang belum selesai'):
        nav = st.session_state.get('vg_v2_claim_revision', 0)
        picked = st.selectbox('Pilih kalimat', ids, index=claim_index,
            format_func=lambda x: f'{ids.index(x)+1}. '+next(c['text'] for c in claims if c['claim_id']==x)[:38],
            key=f'vg_v2_claim_select_{nav}')
        if st.button('Buka kalimat terpilih', key='vg_v2_open_claim'):
            _go_claim(picked)
        unfinished = first_unfinished_claim(data, state)
        if unfinished and st.button('↪ Lanjutkan kalimat yang belum selesai', key='vg_v2_open_unfinished'):
            _go_claim(unfinished)
    with st.expander('Backup dan status teknis (opsional)'):
        fresh = load_state(path, data)
        st.download_button('Unduh backup keputusan (JSON)',
            json.dumps(fresh, ensure_ascii=False, indent=2),
            file_name='verigraph_ner_review_backup.json', mime='application/json', key='vg_v2_backup')
        issues = export_check(data, fresh)
        if issues:
            st.caption(f'{len(issues)} pemeriksaan belum selesai. Wajar selama proses anotasi berlangsung.')
        else:
            st.download_button('Unduh hasil review (perlu audit independen)',
                export_reviewed_jsonl(data, fresh), file_name='cfever_zh_reviewed_pending_audit.jsonl',
                mime='application/x-ndjson', key='vg_v2_export')
        st.caption('Keputusan tersimpan di: outputs/ner_review_v061a/decisions.json')


def _add_missing_form(claim: str, cid: str, data: dict, path: Path) -> None:
    """Inline form used inside the optional expander."""
    import streamlit as st
    st.write('Salin nama dari kalimat. Tidak perlu memasukkan angka posisi karakter.')
    new_name = st.text_input('Nama yang terlewat', key=f'vg_v2_missing_{cid}')
    new_label = st.selectbox('Jenis nama', _labels(), format_func=readable_label,
                             key=f'vg_v2_missing_label_{cid}')
    matches = find_surface_positions(claim, new_name)
    if new_name and not matches:
        st.warning('Nama tidak ditemukan persis di kalimat.')
    if matches and len(matches) > 1:
        at = st.selectbox('Pilih kemunculan', range(len(matches)),
                          format_func=lambda x: f'Kemunculan {x+1}', key=f'vg_v2_missing_occ_{cid}')
    else:
        at = 0
    if matches:
        st.markdown(highlighted_html(claim, *matches[at]), unsafe_allow_html=True)
    if st.button('＋ Simpan nama tambahan', disabled=not matches,
                 key=f'vg_v2_missing_save_{cid}'):
        try:
            add_manual_mention(path, data, cid, *matches[at], new_label)
            st.session_state['vg_v2_flash'] = 'Nama tambahan berhasil disimpan.'
            st.session_state['vg_v2_candidate'] = None
            st.session_state['vg_v2_review_saved'] = False
            st.session_state['vg_v2_candidate_revision'] = st.session_state.get('vg_v2_candidate_revision', 0) + 1
            st.rerun()
        except Exception as exc:
            st.error(f'Nama belum ditambahkan: {exc}')

def _esc(value: str) -> str:
    import html
    return html.escape(str(value))


def _recommendation_friendly(reco: str) -> str:
    return {
        'ACCEPT_RECOMMENDED':'AI menyarankan diterima',
        'REVISE_RECOMMENDED':'AI menyarankan diperbaiki',
        'REJECT_RECOMMENDED':'AI menyarankan ditolak',
        'NEEDS_ADJUDICATION':'Perlu pemeriksaan tambahan',
        'NONE':'Tidak ada saran AI',
    }.get(reco, 'Belum ada informasi')


def _has_accepted_core(data: dict, state: dict, claim_id: str) -> bool:
    return any(r['claim_id']==claim_id and state['decisions'].get(r['candidate_id'],{}).get('action')
               in ('ACCEPT','REVISE') and state['decisions'][r['candidate_id']].get('label') in CORE_LABELS
               for r in all_candidates(data,state))


def render_ner_annotation_tab(root: Path, show_mode_toggle: bool = True) -> None:
    import streamlit as st
    if show_mode_toggle:
        choice = st.radio('Tampilan review',['Review terpandu','Review teknis (lama)'],
                          horizontal=True,key='vg_v2_ui_mode')
        if choice=='Review teknis (lama)':
            from src.ner_annotation_ui_advanced import render_ner_annotation_tab as legacy_render
            legacy_render(root)
            return
    _render_guided(Path(root))
