"""Build reusable demonstration KG from curated CSV files."""
from __future__ import annotations

import json

from networkx.readwrite import json_graph
import networkx as nx

from src.core import ROOT, make_graph


def main():
    graph = make_graph()
    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, out / "demo_kg.graphml")
    (out / "demo_kg.json").write_text(
        json.dumps(json_graph.node_link_data(graph, edges="links"),
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Built demonstration KG: {graph.number_of_nodes()} entities, "
          f"{graph.number_of_edges()} relations")
    print(f"Saved {out / 'demo_kg.graphml'} and {out / 'demo_kg.json'}")


if __name__ == "__main__":
    main()
