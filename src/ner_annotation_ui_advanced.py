"""Streamlit interface for v0.6.1a Mandarin NER human annotation review."""
from __future__ import annotations

from pathlib import Path
import json

from src.ner_annotation import (
    ACTIONS, CLAIM_ACTIONS, CORE_LABELS, add_manual_mention, all_candidates,
    export_check, export_reviewed_jsonl, load_candidates, load_state,
    review_summary, save_candidate_decision, save_claim_review,
)


def _label_options(*values: str) -> list[str]:
    base = ["person", "organization", "location", "event",
            "OTHER:WORK", "OTHER:CELESTIAL_BODY", "OTHER:AWARD",
            "OTHER:BUILDING", "OTHER:RELIGION", "OTHER:CONCEPT",
            "OTHER:ETHNIC_GROUP", "OTHER:FIELD", "OTHER:OTHER",
            "AMBIGUOUS:PERSON_OR_LOCATION"]
    for v in values:
        if v and v not in base:
            base.append(v)
    return base


def render_ner_annotation_tab(root: Path) -> None:
    import streamlit as st

    root = Path(root)
    candidate_path = root / "data" / "annotations" / "cfever_zh_review_candidates_v061a.json"
    state_path = root / "outputs" / "ner_review_v061a" / "decisions.json"

    st.subheader("8 — Chinese NER Annotation Review (v0.6.1a)")
    st.caption("Pilot CFEVER train: 50 klaim · 122 kandidat AI · empat kelas inti. "
               "Review ini BUKAN verifikasi kebenaran klaim, persetujuan QID, atau gold otomatis.")
    try:
        data = load_candidates(candidate_path)
        state = load_state(state_path, data)
    except Exception as exc:
        st.error(f"Gagal membaca data anotasi: {exc}")
        return
    counts=review_summary(data,state)
    k1,k2,k3,k4=st.columns(4)
    k1.metric("Klaim pilot",counts["claims"])
    k2.metric("Kandidat (termasuk tambahan)",counts["candidates"])
    k3.metric("Keputusan final kandidat",f'{counts["decided"]}/{counts["candidates"]}')
    k4.metric("Klaim dikonfirmasi",f'{counts["claims_confirmed"]}/{counts["claims"]}')
    st.progress(counts["decided"]/counts["candidates"] if counts["candidates"] else 0)
    st.info("Keputusan awal tetap PENDING. Rekomendasi AI hanyalah referensi. "
            "Setiap tombol Simpan akan menulis keputusan dan riwayatnya ke JSON lokal.")

    claims=data["claims"]
    priorities={r["claim_id"] for r in data["candidates"]
                if r["ai_recommendation"] in ("REVISE_RECOMMENDED","NEEDS_ADJUDICATION")}
    only_priorities=st.checkbox("Tampilkan hanya klaim dengan kasus adjudikasi/revisi AI",value=False,
                                key="v061a_only_priority")
    choices=[c for c in claims if not only_priorities or c["claim_id"] in priorities]
    if not choices:
        st.info("Tidak ada klaim untuk filter ini.")
        return
    claim_index={c["claim_id"]:c for c in claims}
    selected_id=st.selectbox("Pilih klaim",[c["claim_id"] for c in choices],
                             format_func=lambda v:f'{v} · {claim_index[v]["text"][:55]}',
                             key="v061a_selected_claim")
    claim=claim_index[selected_id]["text"]
    st.markdown("**Teks klaim Mandarin (tidak dapat diubah)**")
    st.code(claim,language=None,wrap_lines=True)

    source_notes=[r for r in data.get("claims_without_candidate_ai_notes",[]) if r["claim_id"]==selected_id]
    if source_notes:
        st.caption(f'Klaim awal tanpa kandidat: {source_notes[0].get("ai_recommendation","")} — '
                   f'{source_notes[0].get("ai_note","")}')

    filtered=[r for r in all_candidates(data,state) if r["claim_id"]==selected_id]
    if filtered:
        st.markdown(f"**Kandidat untuk klaim ini: {len(filtered)}**")
        selected_candidate=st.selectbox(
            "Pilih kandidat",[r["candidate_id"] for r in filtered],
            format_func=lambda cid: (
                f'{next(r for r in filtered if r["candidate_id"]==cid)["surface"]} '
                f'[{next(r for r in filtered if r["candidate_id"]==cid)["label"]}] · '
                f'{state["decisions"].get(cid,{}).get("action","PENDING")}'
            ),key=f"v061a_pick_{selected_id}",
        )
        r=next(r for r in filtered if r["candidate_id"]==selected_candidate)
        current=state["decisions"].get(selected_candidate,{})
        suggestion=r.get("ai_recommendation","NONE")
        st.write(f"**Saran AI (belum disetujui):** `{suggestion}`")
        if r.get("ai_note"):
            st.caption(r["ai_note"])
        if r.get("excel_offset_was_for_ai_span"):
            st.warning("Offset di workbook tahap 2 merujuk ke span revisi AI, bukan span kandidat awal. "
                       "Keduanya disimpan terpisah. Periksa offset pilihanmu sebelum menyimpan.")
        st.write(f'Kandidat asal: `{r["surface"]}` ({r["label"]}) '
                 f'[{r["start"]}:{r["end"]}]')
        ai_start_end=r.get("ai_offsets")
        st.caption(f'Saran span/tipe AI: {r.get("ai_span","")} '
                   f'({r.get("ai_label","")}) — offset: {ai_start_end if ai_start_end else "perlu pilih manual"}')

        current_action=current.get("action","PENDING")
        with st.form(key=f'v061a_form_{selected_candidate}'):
            decision=st.selectbox("Keputusan anotator (eksplisit)",ACTIONS,
                                  index=ACTIONS.index(current_action),
                                  help="ACCEPT = terima kandidat asli; REVISE = ubah span atau label; "
                                       "OUT_OF_SCOPE = bukan empat kategori inti.")
            default_start=current.get("start") if current.get("start") is not None else r["start"]
            default_end=current.get("end") if current.get("end") is not None else r["end"]
            start=int(st.number_input("Start (0-based)",min_value=0,max_value=len(claim)-1,
                                      value=int(default_start),step=1))
            end=int(st.number_input("End (exclusive)",min_value=1,max_value=len(claim),
                                    value=int(default_end),step=1))
            label_opts=_label_options(r["label"],r.get("ai_label"),current.get("label"))
            default_label=current.get("label") or r["label"]
            label=st.selectbox("Tipe final",label_opts,index=label_opts.index(default_label))
            note=st.text_area("Catatan keputusan",value=current.get("note", ""),height=70)
            save=st.form_submit_button("Simpan keputusan kandidat",type="primary")
        if 0<=start<end<=len(claim):
            st.caption(f'Pratinjau offset yang dimasukkan: `{claim[:start]}【{claim[start:end]}】{claim[end:]}`')
        else:
            st.warning("Start harus lebih kecil dari end.")
        if save:
            try:
                save_candidate_decision(state_path,data,selected_candidate,decision,
                                        start=start,end=end,label=label,note=note)
                st.success("Keputusan tersimpan di JSON dengan riwayat perubahan.")
                st.rerun()
            except Exception as exc:
                st.error(f"Keputusan tidak disimpan: {exc}")
    else:
        st.warning("Belum ada kandidat dalam klaim ini. Periksa teks dan tambahkan mention jika diperlukan.")

    with st.expander("Tambah entitas yang terlewat (oleh anotator)",expanded=not filtered):
        st.caption("Gunakan offset karakter pada teks asli. Kandidat baru diberi sumber USER_ADDED dan "
                   "dicatat sebagai keputusan manusia, bukan AI.")
        with st.form(key=f'v061a_add_{selected_id}'):
            a=int(st.number_input("Start entitas baru",min_value=0,max_value=len(claim)-1,value=0,step=1))
            b=int(st.number_input("End entitas baru (exclusive)",min_value=1,max_value=len(claim),
                                  value=min(2,len(claim)),step=1))
            new_label=st.selectbox("Tipe entitas baru",_label_options())
            add_note=st.text_area("Alasan anotasi baru",height=70)
            add=st.form_submit_button("Tambah dan simpan mention")
        if 0<=a<b<=len(claim):
            st.caption(f'Pratinjau: {claim[:a]}【{claim[a:b]}】{claim[b:]}')
        if add:
            try:
                add_manual_mention(state_path,data,selected_id,a,b,new_label,add_note)
                st.success("Mention tambahan tersimpan; status kelengkapan klaim direset bila sebelumnya sudah dikonfirmasi.")
                st.rerun()
            except Exception as exc:
                st.error(f"Tidak dapat menambah mention: {exc}")

    st.markdown("### Konfirmasi kelengkapan klaim")
    st.caption("Setelah **semua kandidat** pada klaim ini diputuskan, tinjau lagi teks penuh "
               "untuk memastikan tidak ada entitas inti yang terlewat.")
    existing=state["claim_reviews"].get(selected_id,{})
    existing_action=existing.get("action","PENDING")
    with st.form(key=f'v061a_claim_form_{selected_id}'):
        claim_decision=st.selectbox("Status pemeriksaan seluruh klaim",CLAIM_ACTIONS,
                                    index=CLAIM_ACTIONS.index(existing_action))
        claim_note=st.text_area("Catatan klaim",value=existing.get("note", ""),height=65)
        confirm=st.form_submit_button("Simpan konfirmasi klaim")
    if confirm:
        try:
            save_claim_review(state_path,data,selected_id,claim_decision,claim_note)
            st.success("Pemeriksaan kelengkapan klaim tersimpan.")
            st.rerun()
        except Exception as exc:
            st.error(f"Konfirmasi belum dapat disimpan: {exc}")

    st.divider()
    st.subheader("Ekspor dan backup keputusan")
    state_json=json.dumps(state,ensure_ascii=False,indent=2)
    st.download_button("Download backup keputusan (JSON)",state_json,
                       file_name="verigraph_ner_review_v061a_decisions_BACKUP.json",
                       mime="application/json",key="v061a_backup")
    issues=export_check(data,state)
    if issues:
        st.warning(f"Ekspor dataset review inti masih ditahan: {len(issues)} persoalan belum selesai. "
                   "Jangan hitung metrik NER final sekarang.")
        with st.expander("Lihat contoh masalah (maksimal 25)"):
            for item in issues[:25]:
                st.write("-",item)
    else:
        exported=export_reviewed_jsonl(data,state)
        st.success("Semua kandidat dan seluruh klaim selesai diperiksa. "
                   "Dataset review siap untuk AUDIT INDEPENDEN sebelum dibekukan sebagai gold.")
        st.download_button("Download annotated pilot JSONL (reviewed, belum gold)",exported,
                           file_name="cfever_zh_50_ner_reviewed_v061a.jsonl",
                           mime="application/x-ndjson",key="v061a_export")
    st.caption(f'File keputusan lokal: `{state_path.relative_to(root)}`. '
               "Saat update proyek, jangan mengganti atau menghapus folder outputs/ner_review_v061a.")
