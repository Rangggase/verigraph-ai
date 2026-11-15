"""Streamlit v0.4.2 reviewed and canonical graph workflow."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from networkx.readwrite import json_graph

from src.errors import friendly_error
from src.entity_linking import search_wikidata_candidates
from src.reviewed_graph import (
    build_reviewed_graph, get_mention_rows, get_relation_rows, load_v041, resume_v042_as_v041,
)


def _as_rows(data):
    return data.to_dict("records") if hasattr(data, "to_dict") else list(data)


def render_reviewed_tab(st, root: Path, dot_escape):
    st.subheader("v0.4.2 — Reviewed & Canonical Knowledge Graph")
    st.warning("Accepting an entity or extracted relation does NOT verify the claim. All claim assertions stay UNVERIFIED.")
    st.caption("Start from your saved v0.4.1 graph; this tab never reruns NER or downloads a model.")

    source = st.radio("Review source", ["Saved v0.4.1 output", "Saved v0.4.2 output (continue review)", "Upload v0.4.1 JSON"],
                      horizontal=True, key="v042_source")
    incoming = None
    if source == "Saved v0.4.1 output":
        path = root / "outputs" / "batch_kg_v041.json"
        if path.is_file():
            incoming = path.read_bytes()
            st.caption(f"Source: {path.relative_to(root)}")
        else:
            st.info("Save your v0.4.1 result to outputs/batch_kg_v041.json or choose Upload.")
    elif source == "Saved v0.4.2 output (continue review)":
        path = root / "outputs" / "batch_kg_v042.json"
        if path.is_file():
            incoming = path.read_bytes()
            st.caption(f"Source: {path.relative_to(root)}")
        else:
            st.info("No saved v0.4.2 output found; build it from v0.4.1 first.")
    else:
        uploaded = st.file_uploader("Upload verigraph_batch_kg_v041.json", type=["json"], key="v042_upload")
        if uploaded is not None:
            incoming = uploaded.getvalue()
    if incoming is None:
        return
    try:
        bundle, graph = (resume_v042_as_v041(incoming)
                         if source == "Saved v0.4.2 output (continue review)"
                         else load_v041(incoming))
        mention_rows = get_mention_rows(bundle, graph)
        relation_rows = get_relation_rows(graph)
    except Exception as exc:
        st.error(friendly_error(exc, "Validasi input v0.4.1"))
        return

    token = hashlib.sha256(incoming).hexdigest()[:12]
    st.success(f"Source loaded: {len(mention_rows)} mentions, {len(relation_rows)} candidate assertions.")
    st.subheader("1 — Review entity mentions / Wikidata identity")
    st.caption("Only accepted_mention is eligible for the clean graph. QID is optional and must be human-checked; leave blank if unsure.")
    ment_edit = st.data_editor(
        mention_rows, hide_index=True, num_rows="fixed", use_container_width=True,
        key=f"v042_mentions_{token}",
        disabled=["mention_key", "claim_id", "external_id", "language", "target_id", "surface",
                  "ner_label", "score", "start", "end", "warnings"],
        column_config={
            "decision": st.column_config.SelectboxColumn("decision", options=["pending", "accepted_mention", "rejected_mention"], required=True),
            "qid": st.column_config.TextColumn("qid (manually verified)", help="Valid Q-number after human review; e.g. Q212"),
            "canonical_label": st.column_config.TextColumn("canonical_label")
        },
    )
    ment = _as_rows(ment_edit)
    with st.expander("Lookup Wikidata identity (suggestions only; do not auto-approve)"):
        if mention_rows:
            ix = st.selectbox("Mention to search", list(range(len(mention_rows))),
                              format_func=lambda i: f"{mention_rows[i]['external_id']} · {mention_rows[i]['surface']}",
                              key="v042_pick_mention")
            if st.button("Search candidate IDs", key="v042_lookup"):
                try:
                    m = mention_rows[ix]
                    hits = search_wikidata_candidates(m["surface"], m["language"], limit=5)
                    st.session_state["v042_lookup_results"] = hits
                    st.session_state["v042_lookup_key"] = m["mention_key"]
                except Exception as exc:
                    st.error(friendly_error(exc, "Pencarian Wikidata"))
            if st.session_state.get("v042_lookup_key") == mention_rows[ix]["mention_key"]:
                hits = st.session_state.get("v042_lookup_results", [])
                if hits:
                    st.dataframe([{k: h.get(k, "") for k in ("id", "label", "description", "url")} for h in hits],
                                 hide_index=True, use_container_width=True)
                    st.caption("Manually copy an approved candidate QID into the corresponding accepted_mention row above.")
                else:
                    st.info("No candidates found. Leave QID empty unless you can verify an identity independently.")

    st.subheader("2 — Review extracted relations")
    st.caption("accepted_extraction = relation is a faithful extraction of what the claim ASSERTS; NOT independently true.")
    if relation_rows:
        rel_edit = st.data_editor(
            relation_rows, hide_index=True, num_rows="fixed", use_container_width=True,
            key=f"v042_relations_{token}",
            disabled=["relation_key", "claim_id", "external_id", "subject", "predicate", "object", "evidence_text"],
            column_config={"decision": st.column_config.SelectboxColumn(
                "decision", options=["pending", "accepted_extraction", "rejected_extraction"], required=True)},
        )
        rels = _as_rows(rel_edit)
    else:
        st.info("No candidate semantic relations in the source. The clean graph can still contain accepted claim–entity mentions.")
        rels = []

    if st.button("Build clean & audit graphs (v0.4.2)", type="primary", key="v042_build"):
        try:
            result = build_reviewed_graph(bundle, ment, rels)
            saved = root / "outputs" / "batch_kg_v042.json"
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            st.session_state["v042_output"] = result
            st.session_state["v042_token"] = token
            st.success(f"Saved both graphs to {saved.relative_to(root)}")
        except Exception as exc:
            st.session_state.pop("v042_output", None)
            st.error(friendly_error(exc, "Pembangunan graf (ditolak validasi)"))

    result = st.session_state.get("v042_output")
    if not result or st.session_state.get("v042_token") != token:
        return
    report = result["report"]
    a, b, c, d = st.columns(4)
    a.metric("Audit edges", report["raw_edges"])
    b.metric("Clean edges", report["clean_edges"])
    c.metric("Canonical QIDs", report["canonical_qids"])
    d.metric("Accepted semantic edges", report["accepted_relation_edges_in_clean"])
    st.write("**Validation:**", "PASS" if result["validation"]["passed"] else "FAILED")
    st.json({"mention_decisions": report["mention_decisions"],
             "relation_decisions": report["relation_decisions"],
             "excluded_relations_due_to_endpoints": report["accepted_relations_excluded_due_to_endpoints"],
             "clean_edge_types": report["clean_edge_types"],
             "validation": result["validation"]})

    st.download_button("Download full reviewed bundle (audit + clean)",
                       json.dumps(result, ensure_ascii=False, indent=2),
                       file_name="verigraph_batch_kg_v042.json", mime="application/json", key="v042_export_full")
    st.download_button("Download clean graph only (node-link JSON)",
                       json.dumps(result["clean_graph"], ensure_ascii=False, indent=2),
                       file_name="verigraph_clean_graph_v042.json", mime="application/json", key="v042_export_clean")

    st.subheader("3 — Graph preview")
    preview = st.radio("Graph view", ["Clean (recommended)", "Audit (includes rejected/pending)"],
                       horizontal=True, key="v042_preview")
    blob = result["clean_graph"] if preview.startswith("Clean") else result["audit_graph"]
    g = json_graph.node_link_graph(blob, edges="links", directed=True, multigraph=True)
    claims = [n for n, a in g.nodes(data=True) if a.get("node_type") == "unverified_claim"][:3]
    show = set(claims)
    for c in claims:
        show.update(g.successors(c))
    if not preview.startswith("Clean"):
        show.update(n for n in g if g.nodes[n].get("node_type") == "reviewed_entity" and any(
            u in show for u in g.predecessors(n)))
    show = set(sorted(show)[:70])
    dot = ['digraph V {', 'rankdir=LR;', 'node [shape=box,style="rounded"];']
    for n in sorted(show):
        a = g.nodes[n]
        label = ("CLAIM " + str(a.get("external_id", "")) if a.get("node_type") == "unverified_claim"
                 else str(a.get("label", n)))[:65]
        dot.append(f'{dot_escape(n)} [label={dot_escape(label)}];')
    for s, t, e in g.edges(data=True):
        if s in show and t in show:
            kind = e.get("edge_kind")
            style = "dashed" if kind == "claim_assertion" else "solid"
            dot.append(f'{dot_escape(s)} -> {dot_escape(t)} [label={dot_escape(e.get("relation", ""))},style={dot_escape(style)}];')
    dot.append("}")
    st.graphviz_chart("\n".join(dot), use_container_width=True)
    st.caption("Claim direction is preserved. Dashed semantic assertions are never confirmed facts. Clean view hides unaccepted mentions and unreviewed relations.")
