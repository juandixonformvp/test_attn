#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_graph.common import add_common_arguments, connect, prepare_output, read, write_csv, write_json  # noqa: E402


TOTAL_QUERY = """
MATCH (d)-[:MENTIONS]->(:Entity)
WHERE d:Post OR d:Claim
RETURN count(DISTINCT d) AS documents
"""

MARGINS_QUERY = """
MATCH (d)-[:MENTIONS]->(e:Entity)
WHERE d:Post OR d:Claim
RETURN e.canonical AS entity, e.type AS entity_type, count(DISTINCT d) AS documents
"""

PAIRS_QUERY = """
MATCH (d)-[:MENTIONS]->(a:Entity), (d)-[:MENTIONS]->(b:Entity)
WHERE (d:Post OR d:Claim) AND a.canonical < b.canonical
WITH a.canonical AS entity_a, b.canonical AS entity_b,
     count(DISTINCT d) AS co_documents
WHERE co_documents >= $min_cooccurrence
RETURN entity_a, entity_b, co_documents
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a PMI-weighted entity co-attention network")
    add_common_arguments(parser)
    parser.add_argument("--min-cooccurrence", type=int, default=3)
    args = parser.parse_args()
    out = prepare_output(args.output_dir, "entity_coattention")
    with connect(args) as driver:
        total = read(driver, args.database, TOTAL_QUERY)[0]["documents"]
        margins_rows = read(driver, args.database, MARGINS_QUERY)
        pairs = read(driver, args.database, PAIRS_QUERY, min_cooccurrence=args.min_cooccurrence)

    margins = {row["entity"]: row for row in margins_rows}
    graph = nx.Graph()
    pair_rows = []
    for row in pairs:
        a, b, co = row["entity_a"], row["entity_b"], row["co_documents"]
        expected_denominator = margins[a]["documents"] * margins[b]["documents"]
        pmi = math.log2((co * total) / expected_denominator) if expected_denominator and total else 0.0
        ppmi = max(0.0, pmi)
        pair_rows.append({**row, "pmi": pmi, "ppmi": ppmi})
        if ppmi > 0:
            graph.add_edge(a, b, weight=ppmi * co, ppmi=ppmi, co_documents=co)
    pair_rows.sort(key=lambda row: (row["ppmi"], row["co_documents"]), reverse=True)

    communities = list(nx.community.louvain_communities(graph, weight="weight", seed=42)) if graph else []
    membership = {}
    for community_id, nodes in enumerate(sorted(communities, key=len, reverse=True), 1):
        for node in nodes:
            membership[node] = community_id
    pagerank = nx.pagerank(graph, weight="weight") if graph else {}
    entity_rows = [
        {
            **row,
            "community": membership.get(row["entity"]),
            "coattention_pagerank": pagerank.get(row["entity"], 0.0),
            "coattention_degree": graph.degree(row["entity"], weight="weight") if row["entity"] in graph else 0.0,
        }
        for row in margins_rows
    ]
    entity_rows.sort(key=lambda row: row["coattention_pagerank"], reverse=True)
    write_csv(out / "entity_pairs.csv", pair_rows)
    write_csv(out / "entities_ranked.csv", entity_rows)
    write_json(out / "summary.json", {"documents": total, "entities": len(margins), "positive_pmi_edges": graph.number_of_edges(), "communities": len(communities)})
    print(f"Wrote {graph.number_of_edges():,} positive-PMI entity pairs to {out}")


if __name__ == "__main__":
    main()
