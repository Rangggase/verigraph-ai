"""Streamlit UI for opt-in post-processing of a saved v0.4 graph.

This module does not run GLiNER, and does not make network requests until
'Wikidata search' or 'Ollama extraction' is explicitly clicked.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from networkx.readwrite import json_graph

from src.errors import friendly_error
from src.entity_linking import search_wikidata_candidates
from src.kg_enrichment import load_v04_bundle, review_rows, enrich_bundle


def render_quality_tab(st, root: Path, dot_escape):
    st.subheader("v0.4.1 — Review NER, resolve Wikidata identities & enrich KG")
    st.warning("All candidate triples remain UNVERIFIED. A reviewed entity ID is NOT evidence that a claim is true.")
    st.caption("Reuse your existing v0.4 JSON: no GLiNER run or model download is needed.")
    source_option = st.radio("Batch graph source", ["Saved v0.4 output", "Upload v0.4 JSON", "Included 5-claim example"], horizontal=True, key="v041_source")
    raw = None
    try:
        if source_option == "Saved v0.4 output":
            output_path = root / "outputs" / "batch_kg_v04.json"
            if output_path.is_file():
                raw = output_path.read_bytes()
                st.caption(f"Found: {output_path.relative_to(root)}")
            elif st.session_state.get("v04_bundle"):
                raw = json.dumps(st.session_state["v04_bundle"], ensure_ascii=False).encode("utf-8")
                st.caption("Using previous batch result from this Streamlit session")
            else:
                st.info("No saved output found. Copy your v0.4 output to outputs/batch_kg_v04.json, or choose Upload / Example.")
        elif source_option == "Upload v0.4 JSON":
            upload = st.file_uploader("Select verigraph_batch_kg_v04.json", type=["json"], key="v041_upload")
            if upload is not None:
                raw = upload.getvalue()
        else:
            raw = (root / "data" / "examples" / "averitec_5_v04.json").read_bytes()
            st.caption("Included example uses the five AVeriTeC claims from the v0.4 pilot")
        if raw is None:
            return
        bundle, graph = load_v04_bundle(raw)
        mentions = review_rows(graph)
    except Exception as exc:
        st.error(friendly_error(exc, "Pemuatan graf batch"))
        return
    file_token = hashlib.sha256(raw).hexdigest()[:12]
    st.success(f"Loaded {len(mentions)} mention instances from {len(bundle.get('runs', []))} processed claims")
    suspicious = [r for r in mentions if r["warnings"]]
    st.write(f"Quality flags: **{len(suspicious)}** mention instances need closer review. Flags are heuristics, not confirmed errors.")
    if suspicious:
        st.dataframe([{"claim": r["external_id"], "surface": r["surface"], "model_type": r["ner_label"],
                       "score": round(r["score"], 3), "flags": r["warnings"]} for r in suspicious],
                     use_container_width=True, hide_index=True)

    st.subheader("Human review & canonical IDs")
    st.caption("For each row choose pending / accepted_mention / rejected_mention. Only accepted rows may have QIDs."
               " Edit 'qid' manually after checking Wikidata. Leave QID blank if uncertain.")
    # The input is a list of JSON-compatible dicts to avoid relying on pandas.
    edited = st.data_editor(
        mentions, key=f"v041_editor_{file_token}", use_container_width=True,
        hide_index=True, num_rows="fixed", disabled=["mention_key", "claim_id", "external_id", "language",
                    "target_id", "surface", "ner_label", "score", "start", "end", "warnings"],
        column_config={
            "decision": st.column_config.SelectboxColumn("decision", options=["pending", "accepted_mention", "rejected_mention"], required=True),
            "qid": st.column_config.TextColumn("qid (human checked)", help="Example: Q3630; accepted_mention required"),
            "canonical_label": st.column_config.TextColumn("canonical_label"),
        },
    )
    current_rows = edited.to_dict("records") if hasattr(edited, "to_dict") else list(edited)

    with st.expander("Search Wikidata candidate identity (manual approval required)"):
        if mentions:
            pick = st.selectbox("Mention", list(range(len(mentions))),
                                format_func=lambda i: f"{mentions[i]['external_id']} · {mentions[i]['surface']} ({mentions[i]['language']})",
                                key="v041_pick_mention")
            if st.button("Search Wikidata candidates", key="v041_search"):
                try:
                    st.session_state["v041_wikidata_hits"] = search_wikidata_candidates(
                        mentions[pick]["surface"], mentions[pick]["language"], limit=5)
                    st.session_state["v041_wikidata_query"] = mentions[pick]["mention_key"]
                except Exception as exc:
                    st.error(friendly_error(exc, "Permintaan Wikidata"))
            if st.session_state.get("v041_wikidata_query") == mentions[pick]["mention_key"]:
                hits = st.session_state.get("v041_wikidata_hits", [])
                if hits:
                    st.dataframe(hits, use_container_width=True, hide_index=True)
                    st.caption("Copy the correct QID into the corresponding editable row above and select accepted_mention. Search rank is not verification.")
                else:
                    st.info("No candidates retrieved yet (or no results).")

    st.subheader("Candidate relation extraction")
    method = st.radio("Extraction method", ["Auditable rules (offline)", "Local Ollama LLM (optional)", "None (review & QIDs only)"],
                      index=0, key="v041_method", horizontal=True)
    extractor = "rules" if method.startswith("Auditable") else "ollama" if method.startswith("Local") else "none"
    model_name = "qwen2.5:3b"
    if extractor == "ollama":
        st.info("Requires Ollama running locally with a downloaded model. On CPU this can be slow. Nothing is sent to a cloud API.")
        model_name = st.text_input("Ollama model name", value="qwen2.5:3b", key="v041_model_name")
    if st.button("Build reviewed candidate KG (v0.4.1)", type="primary", key="v041_enrich"):
        try:
            with st.spinner("Enriching existing graph..."):
                enriched = enrich_bundle(bundle, current_rows, extractor=extractor, model_name=model_name)
            path = root / "outputs" / "batch_kg_v041.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
            st.session_state["v041_enriched"] = enriched
            st.session_state["v041_enriched_token"] = file_token
            st.success(f"Saved to {path.relative_to(root)}")
        except Exception as exc:
            st.error(friendly_error(exc, "Pengayaan KG"))

    out = st.session_state.get("v041_enriched")
    if out and st.session_state.get("v041_enriched_token") == file_token:
        stats = out["quality_report"]
        a, b, c, d = st.columns(4)
        a.metric("Mentions", stats["mentions"])
        b.metric("Human-reviewed QIDs", stats["reviewed_qid_count"])
        c.metric("Semantic candidates", stats["candidate_relations"])
        d.metric("Graph edges", stats["edges"])
        st.json({"edge_types": stats["edge_types"], "candidate_relation_types": stats["candidate_relation_types"],
                 "quality_flags": stats["quality_flags"]}, expanded=True)
        if stats["errors"]:
            st.warning("Some claims could not be enriched; inspect errors below. The other claims were saved.")
            st.dataframe(stats["errors"])
        st.download_button("Download v0.4.1 reviewed KG JSON", data=json.dumps(out, indent=2, ensure_ascii=False),
                           file_name="verigraph_batch_kg_v041.json", mime="application/json")
        st.subheader("Graph preview — first 3 claims")
        g = json_graph.node_link_graph(out["graph"], edges="links", directed=True, multigraph=True)
        claims = [n for n, v in g.nodes(data=True) if v.get("node_type") == "unverified_claim"][:3]
        keep = set(claims)
        for claim in claims:
            keep.update(g.successors(claim))
        for mention in list(keep):
            if mention in g:
                keep.update(v for v in g.successors(mention) if g.nodes[v].get("node_type") == "reviewed_entity")
        dot = ['digraph KG {', 'rankdir=LR;', 'node [shape=box,style="rounded"];']
        for node in keep:
            attrs = g.nodes[node]
            title = ("CLAIM " + str(attrs.get("external_id", "")) if attrs.get("node_type") == "unverified_claim"
                     else str(attrs.get("label", node)))[:50]
            dot.append(f'{dot_escape(node)} [label={dot_escape(title)}];')
        for start, end, attrs in g.edges(data=True):
            if start in keep and end in keep:
                label = attrs.get("relation", "")
                style = "dashed" if attrs.get("edge_kind") == "claim_assertion" else "solid"
                dot.append(f'{dot_escape(start)} -> {dot_escape(end)} [label={dot_escape(label)},style={dot_escape(style)}];')
        dot.append("}")
        st.graphviz_chart("\n".join(dot), use_container_width=True)
        st.caption("Dashed edges are candidate assertions, NOT verified facts; RESOLVES_TO indicates reviewed identity only.")
