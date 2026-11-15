"""VeriGraph AI v0.4.1 — post-processing and review of saved v0.4 batch graphs."""
from __future__ import annotations
import json
from pathlib import Path

import streamlit as st
from networkx.readwrite import json_graph

from src.assertion_graph import build_mention_graph
from src.core import make_graph, parse_demo_claim, read_csv
from src.entity_linking import local_link, search_wikidata_candidates
from src.ner_pipeline import load_ner_model, extract_entities, MODEL_ID
from src.relation_extraction import extract_relation_candidates, build_assertion_graph
from src.batch_pipeline import (load_records, guess_column, normalize_records, approved_links_from_csv,
                                build_batch_graph, export_graph_bundle)
from src.errors import friendly_error
from src.theme import apply_global_theme, unverified_banner
from src.quality_ui import render_quality_tab
from src.reviewed_ui import render_reviewed_tab
from src.classification_ui import render_classification_tab
from src.ner_annotation_ui import render_ner_annotation_tab

st.set_page_config(page_title="VeriGraph AI | KG multibahasa", page_icon="🕸️", layout="wide")
apply_global_theme(st)

# BEGIN VERIGRAPH V061B BEGINNER ENTRYPOINT
from src.easy_home_ui import render_easy_home
_vg_mode = st.sidebar.radio(
    "Tampilan aplikasi",
    ["Mudah digunakan", "Mode Peneliti (fitur lengkap)"],
    help="Mode mudah untuk review entitas. Mode Peneliti menampilkan semua fitur lama.",
    key="vg_app_mode_v061b",
)
if _vg_mode == "Mudah digunakan":
    render_easy_home(Path(__file__).resolve().parent)
    st.stop()
# END VERIGRAPH V061B BEGINNER ENTRYPOINT

st.title("VeriGraph AI — knowledge graph klaim multibahasa")
unverified_banner(st, "Alat ini membangun graf dari <strong>klaim yang belum terverifikasi</strong>. Ia tidak memeriksa kebenaran klaim.")

graph = make_graph()
curated_tab, ner_tab, relation_tab, batch_tab, quality_tab, reviewed_tab, classification_tab, annotation_tab = st.tabs(["1. KG contoh", "2. NER & Wikidata", "3. Relasi kandidat", "4. KG dari dataset", "5. Kualitas KG", "6. KG bersih hasil review", "7. Baseline ML", "8. Review NER Mandarin"])


@st.cache_resource(show_spinner="Loading multilingual NER model (first use downloads weights)...")
def cached_ner():
    return load_ner_model()


def dot_escape(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


with curated_tab:
    st.info("Demo awal ini hanya membaca relasi ibu kota dan memakai graf fakta kecil yang disusun manual.")
    claims = read_csv("demo_claims.csv")
    left, right = st.columns([1, 1])
    with left:
        language = st.selectbox("Bahasa", ["id", "en", "zh"],
                                format_func={"id": "Bahasa Indonesia", "en": "English", "zh": "中文"}.get,
                                key="demo_language")
        samples = [row for row in claims if row["language"] == language]
        demo = st.selectbox("Contoh klaim", samples, format_func=lambda r: r["text"])
        text = st.text_area("Klaim (dapat diubah)", value=demo["text"], height=90,
                            key=f"demo_{demo['claim_id']}")
        if st.button("Periksa relasi", type="primary"):
            st.session_state["demo_result"] = parse_demo_claim(text, language, graph)
    with right:
        st.subheader("Hasil pemeriksaan KG")
        result = st.session_state.get("demo_result")
        if result:
            if result["status"] == "supported":
                st.success("SESUAI DENGAN KG DEMO")
            elif result["status"] == "conflict":
                st.warning("BERTENTANGAN DENGAN KG DEMO")
            else:
                st.info(result["status"].upper().replace("_", " "))
            st.write(result.get("message", ""))
            if result.get("stored_capital"):
                st.write("Stored capital:", result["stored_capital"])
            if result.get("source_url"):
                st.write("Reference:", result["source_url"])
            with st.expander("Structured output"):
                st.json(result)
        else:
            st.info("Pilih contoh klaim lalu tekan Periksa relasi.")

    st.subheader("Graf contoh multibahasa")
    label_key = {"id": "label_id", "en": "label_en", "zh": "label_zh"}[language]
    dot_lines = ["digraph G {", "rankdir=LR;", "node [shape=box];"]
    for entity_id, attrs in graph.nodes(data=True):
        dot_lines.append(f'{dot_escape(entity_id)} [label={dot_escape(attrs[label_key] + " | " + entity_id)}];')
    for a, b, _ in graph.edges(data=True):
        dot_lines.append(f'{dot_escape(a)} -> {dot_escape(b)} [label="capital"];')
    dot_lines.append("}")
    st.graphviz_chart("\n".join(dot_lines), use_container_width=True)
    with st.expander("Curated KG sources"):
        st.dataframe(read_csv("relations.csv"), use_container_width=True)

with ner_tab:
    st.subheader("Automatic multilingual NER")
    st.caption("Mendeteksi entitas, menyarankan kandidat Wikidata, dan membuat graf klaim–entitas yang BELUM TERVERIFIKASI. Bukan pendeteksi hoaks.")
    lang = st.selectbox("Bahasa input", ["id", "en", "zh"],
                        format_func={"id": "Bahasa Indonesia", "en": "English", "zh": "中文"}.get,
                        key="ner_language")
    examples = {
        "id": "Prabowo Subianto melakukan kunjungan kerja di Jakarta.",
        "en": "Barack Obama visited Paris and met Angela Merkel.",
        "zh": "习近平在北京出席会议。",
    }
    user_text = st.text_area("Teks bebas (maks. 1.500 karakter)",
                             value=examples[lang], height=120, key=f"ner_text_{lang}")
    threshold = st.slider("NER confidence threshold", min_value=0.1, max_value=0.9,
                          value=0.35, step=0.05)
    st.caption(f"Checkpoint: {MODEL_ID}. First use downloads model weights; CPU inference may be slow. Evaluate entity quality for all three languages.")
    if st.button("Ekstrak entitas", type="primary"):
        try:
            with st.spinner("Running model-based entity extraction..."):
                recognized = extract_entities(user_text, lang, cached_ner(), threshold)
            st.session_state["ner_output"] = {
                "text": user_text, "language": lang, "entities": recognized,
                "links": {}, "candidates": {},
            }
        except Exception as exc:
            st.error(friendly_error(exc, "Ekstraksi entitas"))

    state = st.session_state.get("ner_output")
    if state and (state["text"] != user_text or state["language"] != lang):
        st.info("Input berubah. Tekan 'Ekstrak entitas' untuk memperbarui graf.")
        state = None

    if state:
        recognized = state["entities"]
        st.write(f"Detected entities: **{len(recognized)}**")
        if not recognized:
            st.warning("Tidak ada entitas di atas ambang kepercayaan ini. Turunkan ambang atau gunakan teks lebih panjang.")
        else:
            st.dataframe(recognized, use_container_width=True)
            st.subheader("Entity linking & human review")
            st.caption("Exact matches to the curated alias list are linked locally. Online Wikidata hits are only candidates until you explicitly approve them.")
            for i, ent in enumerate(recognized):
                with st.expander(f"{i + 1}. {ent['text']}  ({ent['label']}, score={ent['score']:.2f})"):
                    existing_link = local_link(ent["text"], lang, graph)
                    if existing_link is not None:
                        state["links"][i] = existing_link
                        st.success(f"Curated local alias match: {existing_link['id']}")
                        st.write(existing_link["url"])
                    elif i in state["links"]:
                        link = state["links"][i]
                        st.success(f"Manually approved candidate: {link['id']} ({link['label']})")
                        st.write(link["url"])
                    else:
                        st.info("Unresolved mention: no automatic Wikidata ID assignment.")
                        if st.button("Cari kandidat Wikidata", key=f"search_{i}"):
                            try:
                                state["candidates"][i] = search_wikidata_candidates(ent["text"], lang)
                            except Exception as exc:
                                st.warning(friendly_error(exc, "Pencarian Wikidata"))
                        if i in state["candidates"]:
                            hits = state["candidates"][i]
                            if not hits:
                                st.warning("No Wikidata candidates found.")
                            else:
                                choice = st.selectbox(
                                    "Review candidate identity", list(range(len(hits))),
                                    format_func=lambda j: f"{hits[j]['id']} — {hits[j]['label']} — {hits[j]['description']}",
                                    key=f"candidate_{i}",
                                )
                                selected = hits[choice]
                                st.write(selected["url"])
                                if st.button("Setujui identitas Wikidata terpilih", key=f"approve_{i}"):
                                    state["links"][i] = {
                                        **selected,
                                        "method": "manually_approved_wikidata_candidate",
                                    }
                                    st.rerun()

            mention_graph = build_mention_graph(user_text, lang, recognized, state["links"])
            st.subheader("Unverified claim–entity mention graph")
            dot_lines = ["digraph G {", "rankdir=LR;", "node [shape=ellipse];"]
            for node_id, attrs in mention_graph.nodes(data=True):
                display = "CLAIM (unverified)" if attrs.get("node_type") == "unverified_claim" else f"{attrs.get('label', '')}\n{node_id}"
                dot_lines.append(f'{dot_escape(node_id)} [label={dot_escape(display)}];')
            for a, b, edge_attrs in mention_graph.edges(data=True):
                dot_lines.append(f'{dot_escape(a)} -> {dot_escape(b)} [label={dot_escape(edge_attrs["relation"])}];')
            dot_lines.append("}")
            st.graphviz_chart("\n".join(dot_lines), use_container_width=True)
            output = json.dumps(json_graph.node_link_data(mention_graph, edges="links"),
                                ensure_ascii=False, indent=2)
            st.download_button("Unduh graf mention (JSON)", output,
                               file_name="claim_mentions.json", mime="application/json")
            st.warning("MENTIONS berarti klaim memuat nama entitas. Ini TIDAK membuktikan klaim atau relasi apa pun benar.")

with relation_tab:
    st.subheader("Candidate relation extraction & KG construction")
    st.warning(
        "Experimental RULE-BASED relation extractor, NOT an LLM and NOT a fake-news detector. "
        "Its edges represent unverified assertions made by the input text; do not treat them as factual truth."
    )
    st.caption(
        "First extract named entities in tab 2 and approve relevant Wikidata identities there. "
        "Then inspect the candidate relationships below."
    )
    kg_state = st.session_state.get("ner_output")
    if not kg_state:
        st.info("Belum ada ekstraksi. Buka tab 2, masukkan klaim, lalu tekan 'Ekstrak entitas'.")
    elif kg_state["text"] != user_text or kg_state["language"] != lang:
        st.info("Current input was edited after extraction. Re-run NER in tab 2 first.")
    else:
        kg_text = kg_state["text"]
        kg_lang = kg_state["language"]
        kg_ents = kg_state["entities"]
        candidates = extract_relation_candidates(kg_text, kg_lang, kg_ents)
        st.write(f"Candidate relations found: **{len(candidates)}**")
        if candidates:
            for i, cand in enumerate(candidates):
                with st.expander(f"{i + 1}. {cand['subject']} — {cand['predicate']} → {cand['object']}", expanded=True):
                    st.caption(f"Matched phrase: '{cand['trigger']}' • Rule: {cand['method']} • Language: {kg_lang}")
                    st.code(cand["evidence_text"], language=None)
                    st.caption("Review below checks extraction accuracy only; it does not verify the claim's truth.")
                    st.selectbox(
                        f"Extraction review for relation {i + 1}",
                        ["pending", "approved_extraction", "rejected"],
                        key=f"relation_review_{kg_lang}_{i}_{kg_text}",
                        format_func={
                            "pending": "Pending review",
                            "approved_extraction": "Accurate as a textual assertion (not verified)",
                            "rejected": "Reject extracted relation",
                        }.get,
                    )
        else:
            st.info(
                "No supported relation template matched. v0.3 deliberately abstains instead of "
                "inventing an edge. Try 'Prabowo Subianto melakukan kunjungan kerja di Jakarta.'"
            )
        review_map = {
            i: st.session_state.get(f"relation_review_{kg_lang}_{i}_{kg_text}", "pending")
            for i in range(len(candidates))
        }
        rel_graph = build_assertion_graph(kg_text, kg_lang, kg_ents, kg_state["links"], candidates, review_map)
        st.subheader("Claim assertion graph (unverified)")
        dot_lines = ["digraph G {", "rankdir=LR;", 'node [shape=box, style="rounded"];']
        for node_id, attrs in rel_graph.nodes(data=True):
            if attrs.get("node_type") == "unverified_claim":
                disp = "CLAIM (UNVERIFIED)"
            else:
                disp = f"{attrs.get('label', '')}\n{node_id}"
            dot_lines.append(f'{dot_escape(node_id)} [label={dot_escape(disp)}];')
        for a, b, attrs in rel_graph.edges(data=True):
            is_assertion = attrs.get("edge_kind") == "claim_assertion"
            relation = str(attrs.get("relation", ""))
            desc = relation + (" [REVIEWED]" if attrs.get("extraction_review") == "approved_extraction" else " [PENDING]") if is_assertion else relation
            style = "dashed" if is_assertion else "solid"
            dot_lines.append(
                f'{dot_escape(a)} -> {dot_escape(b)} '
                f'[label={dot_escape(desc)}, style={dot_escape(style)}];'
            )
        dot_lines.append("}")
        st.graphviz_chart("\n".join(dot_lines), use_container_width=True)
        payload = {
            "schema_version": "0.3",
            "warning": "All extracted assertions are unverified; identity approval is NOT factual verification.",
            "input": {"language": kg_lang, "text": kg_text},
            "candidates": [{**c, "extraction_review": review_map[i]} for i, c in enumerate(candidates)],
            "graph": json_graph.node_link_data(rel_graph, edges="links"),
        }
        st.download_button(
            "Unduh KG kandidat + provenans (JSON)",
            data=json.dumps(payload, indent=2, ensure_ascii=False),
            file_name="verigraph_candidate_kg_v03.json",
            mime="application/json",
        )
        st.caption(
            "Review persists only during the current Streamlit session. Export JSON before restarting. "
            "In a later release we will add multi-document persistence and independently verified evidence."
        )

with batch_tab:
    st.subheader("v0.4 — Import datasets & build batch candidate KG")
    st.warning(
        "Batch KG contains ONLY unverified claim assertions and detected mentions. "
        "It is NOT a fake-news detector. Gold labels, justifications and gold evidence "
        "are not ingested into the graph."
    )
    st.caption(
        "Start with 5 claims on CPU. Supported: CSV, JSONL, JSON. "
        "This pilot intentionally blocks test/held-out splits. "
        "Unknown entities remain unlinked unless you supply HUMAN-REVIEWED QIDs."
    )

    root = Path(__file__).resolve().parent
    local_options = {
        "AVeriTeC pilot CSV (local)": root / "data/datasets/averitec/averitec_pilot_100.csv",
        "CFEVER train JSONL (local)": root / "data/datasets/cfever/train.jsonl",
        "CFEVER dev JSONL (local)": root / "data/datasets/cfever/dev.jsonl",
        "Built-in synthetic examples": root / "data/demo_claims.csv",
    }
    selection = st.radio("Dataset source", ["Local dataset", "Upload file"], horizontal=True)
    raw_content = None
    filename = ""
    suggested_dataset = ""
    suggested_language = "en"
    suggested_split = "train"
    if selection == "Local dataset":
        chosen = st.selectbox("Dataset location", list(local_options))
        path = local_options[chosen]
        filename = path.name
        suggested_dataset = "averitec" if "AVeriTeC" in chosen else "cfever" if "CFEVER" in chosen else "synthetic_demo"
        suggested_language = "zh" if "CFEVER" in chosen else "en"
        suggested_split = "dev" if " dev " in chosen else "train"
        if path.is_file():
            raw_content = path.read_bytes()
            st.caption(f"Found: {path.relative_to(root)}")
        else:
            st.info(f"File not found: {path.relative_to(root)}. Copy your dataset to this location or use Upload file.")
    else:
        upload = st.file_uploader("Upload a CSV, JSONL or JSON", type=["csv", "jsonl", "json"], key="v04_dataset_upload")
        if upload is not None:
            raw_content = upload.getvalue()
            filename = upload.name
            suggested_dataset = Path(filename).stem

    if raw_content is not None:
        try:
            raw_rows = load_records(raw_content, filename)
        except Exception as exc:
            st.error(friendly_error(exc, "Impor dataset"))
            raw_rows = []
        if raw_rows:
            st.success(f"Loaded a preview of {len(raw_rows)} rows (maximum first 1,000); no model run yet.")
            cols = list(dict.fromkeys(str(k) for row in raw_rows[:25] for k in row.keys()))
            st.dataframe(raw_rows[:5], use_container_width=True)
            inferred_claim = guess_column(cols, "claim")
            inferred_id = guess_column(cols, "id")
            inferred_language = guess_column(cols, "language")
            c1, c2, c3 = st.columns(3)
            with c1:
                claim_field = st.selectbox("Claim/text column", cols,
                                           index=cols.index(inferred_claim) if inferred_claim else 0,
                                           key="v04_claim_field")
                dataset_name = st.text_input("Dataset name", value=suggested_dataset, key="v04_dataset_name")
            with c2:
                id_field = st.selectbox("External ID field (optional)", ["(row number)"] + cols,
                                        index=(1 + cols.index(inferred_id)) if inferred_id else 0,
                                        key="v04_id_field")
                language_field = st.selectbox("Language field (optional)", ["(fixed language)"] + cols,
                                              index=(1 + cols.index(inferred_language)) if inferred_language else 0,
                                              key="v04_lang_field")
            with c3:
                default_language = st.selectbox("Fixed/default language", ["en", "id", "zh"],
                                                index=["en", "id", "zh"].index(suggested_language),
                                                key="v04_default_language")
                split = st.selectbox("Dataset split", ["train", "dev", "validation"],
                                     index=1 if suggested_split == "dev" else 0,
                                     key="v04_split")
            limit = st.number_input("Maximum claims in this batch (CPU: start with 5)",
                                    min_value=1, max_value=min(100, len(raw_rows)), value=min(5, len(raw_rows)),
                                    step=1, key="v04_limit")
            run_ner = st.checkbox("Run GLiNER NER + rule-based candidate relations", value=True,
                                  help="Uncheck for a fast claim-only importer test.", key="v04_run_ner")
            ner_threshold = st.slider("Batch NER confidence", min_value=.10, max_value=.90,
                                      value=.35, step=.05, key="v04_threshold")
            st.markdown("**Optional: reviewed Wikidata mappings for cross-claim entity merging**")
            st.caption("No automatic batch Wikidata search or guessed QIDs. A CSV mapping is applied only if you reviewed it.")
            st.download_button("Download reviewed links CSV template",
                               "language,surface,qid,label\n",
                               file_name="human_approved_qids.csv", mime="text/csv",
                               key="v04_link_template")
            links_upload = st.file_uploader("Human-reviewed links CSV", type=["csv"],
                                            key="v04_links_upload")
            if st.button("Bangun KG kandidat", type="primary", key="v04_build"):
                try:
                    mapped_links = (approved_links_from_csv(links_upload.getvalue())
                                    if links_upload is not None else {})
                    normalized, rejected = normalize_records(
                        raw_rows, claim_field=claim_field, dataset=dataset_name,
                        default_language=default_language, split=split,
                        id_field=None if id_field == "(row number)" else id_field,
                        language_field=None if language_field == "(fixed language)" else language_field,
                        limit=int(limit),
                    )
                    model = cached_ner() if run_ner else None
                    progress = st.progress(0, text="Building graph...")
                    def update_progress(i, total):
                        progress.progress(i / total if total else 1., text=f"Processed {i}/{total} claims")
                    with st.spinner("Extracting and building KG..."):
                        result_graph, summaries = build_batch_graph(
                            normalized, model=model, threshold=ner_threshold,
                            approved_links=mapped_links, progress_callback=update_progress,
                        )
                    bundle = export_graph_bundle(result_graph, summaries)
                    bundle["rejected_rows"] = rejected
                    output_path = root / "outputs" / "batch_kg_v04.json"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
                    st.session_state["v04_bundle"] = bundle
                    st.success(f"Completed. Saved to {output_path.relative_to(root)}")
                except Exception as exc:
                    st.error(friendly_error(exc, "Pembuatan KG batch"))

    bundle = st.session_state.get("v04_bundle")
    if bundle:
        stats = bundle["stats"]
        a, b, c, d = st.columns(4)
        a.metric("Claims", stats["claims"])
        b.metric("Graph nodes", stats["nodes"])
        c.metric("Graph edges", stats["edges"])
        d.metric("NER errors", stats["ner_errors"])
        st.caption(f"Edge counts: {stats['edge_types']}")
        st.dataframe(bundle["runs"], use_container_width=True)
        if bundle.get("rejected_rows"):
            with st.expander("Skipped rows"):
                st.dataframe(bundle["rejected_rows"])
        st.download_button("Unduh KG batch (JSON)", json.dumps(bundle, ensure_ascii=False, indent=2),
                           file_name="verigraph_batch_kg_v04.json", mime="application/json")
        st.subheader("Graph preview — first six claims and their mentions")
        from networkx.readwrite import json_graph as _json_graph
        g4 = _json_graph.node_link_graph(bundle["graph"], edges="links", directed=True, multigraph=True)
        selected_claims = [n for n, a in g4.nodes(data=True) if a.get("node_type") == "unverified_claim"][:6]
        to_show = set(selected_claims)
        for node in selected_claims:
            to_show.update(g4.successors(node))
        if len(to_show) > 70:
            to_show = set(sorted(to_show)[:70])
        dot = ['digraph KG {', 'rankdir=LR;', 'node [shape=box, style="rounded"];']
        for node in to_show:
            data = g4.nodes[node]
            display = ("CLAIM " + str(data.get("external_id", ""))[:20]
                       if data.get("node_type") == "unverified_claim" else str(data.get("label", node))[:35])
            dot.append(f'{dot_escape(node)} [label={dot_escape(display)}];')
        for src, dst, data in g4.edges(data=True):
            if src in to_show and dst in to_show:
                dot.append(f'{dot_escape(src)} -> {dot_escape(dst)} [label={dot_escape(data.get("relation", ""))}];')
        dot.append("}")
        st.graphviz_chart("\n".join(dot), use_container_width=True)
        st.info("Shared nodes only arise from explicitly reviewed QID mappings. Graph display is a subset; export contains all nodes/edges.")


with quality_tab:
    render_quality_tab(st, Path(__file__).resolve().parent, dot_escape)


with reviewed_tab:
    render_reviewed_tab(st, Path(__file__).resolve().parent, dot_escape)


with classification_tab:
    render_classification_tab()


with annotation_tab:
    render_ner_annotation_tab(Path(__file__).resolve().parent)
