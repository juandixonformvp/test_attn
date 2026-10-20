#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_graph.common import add_common_arguments, connect, prepare_output, read, write_csv, write_json  # noqa: E402


ENTITIES_QUERY = """
MATCH (e:Entity)
OPTIONAL MATCH (:Comment)-[r:REACTS]->(e)
WHERE coalesce(r.confidence, 1.0) >= $confidence
RETURN e.canonical AS entity, e.type AS entity_type,
       coalesce(e.mention_freq, 0) AS mention_freq, coalesce(e.doc_freq, 0) AS doc_freq,
       coalesce(e.endorsement_only, false) AS endorsement_only,
       count(r) AS reactions
"""

HALO_EDGES_QUERY = """
MATCH (source)-[r:PROMOTES]->(brand:Entity)
RETURN coalesce(source.canonical, source.handle) AS source, head(labels(source)) AS source_label,
       brand.canonical AS brand, 'PROMOTES' AS channel,
       toFloat(coalesce(r.count, 1)) AS weight
UNION ALL
MATCH (source:Entity)-[r:ENDORSES]->(brand:Entity)
RETURN source.canonical AS source, 'Entity' AS source_label, brand.canonical AS brand,
       'ENDORSES' AS channel, toFloat(coalesce(r.scale, 1.0)) AS weight
UNION ALL
MATCH (brand:Entity)-[r:SPONSORS_EVENT]->(source:Entity)
RETURN source.canonical AS source, 'Entity' AS source_label, brand.canonical AS brand,
       'SPONSORS_EVENT' AS channel, toFloat(coalesce(r.scale, 1.0)) AS weight
"""


def node_id(label: str, value: str) -> str:
    return ("a:" if label == "Account" else "e:") + value


def rank(graph: nx.DiGraph, personalization: dict[str, float]) -> dict[str, float]:
    if not graph:
        return {}
    usable = {node: value for node, value in personalization.items() if node in graph and value > 0}
    return nx.pagerank(graph, weight="weight", personalization=usable or None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose organic, promotional, and contractual brand halo")
    add_common_arguments(parser)
    args = parser.parse_args()
    out = prepare_output(args.output_dir, "brand_halo")
    with connect(args) as driver:
        entities = read(driver, args.database, ENTITIES_QUERY, confidence=args.confidence)
        edges = read(driver, args.database, HALO_EDGES_QUERY)

    graph_all = nx.DiGraph()
    graph_organic = nx.DiGraph()
    channel_totals: dict[str, Counter] = {}
    for row in edges:
        source = node_id(row["source_label"], row["source"])
        brand = "e:" + row["brand"]
        weight = float(row["weight"] or 0)
        graph_all.add_edge(source, brand, weight=graph_all.get_edge_data(source, brand, {}).get("weight", 0) + weight)
        if row["channel"] == "PROMOTES":
            graph_organic.add_edge(source, brand, weight=graph_organic.get_edge_data(source, brand, {}).get("weight", 0) + weight)
        channel_totals.setdefault(row["brand"], Counter())[row["channel"]] += weight
    personalization = {
        "e:" + row["entity"]: float(row["doc_freq"] or 0) + float(row["reactions"] or 0)
        for row in entities
    }
    all_rank = rank(graph_all, personalization)
    organic_rank = rank(graph_organic, personalization)
    brands = [row for row in entities if "brand" in (row["entity_type"] or "").lower() or row["entity"] in channel_totals]
    output = []
    for row in brands:
        key = "e:" + row["entity"]
        channels = channel_totals.get(row["entity"], Counter())
        output.append(
            {
                **row,
                "promotion_weight": channels["PROMOTES"],
                "endorsement_weight": channels["ENDORSES"],
                "event_sponsorship_weight": channels["SPONSORS_EVENT"],
                "all_channel_pagerank": all_rank.get(key, 0.0),
                "organic_network_pagerank": organic_rank.get(key, 0.0),
                "contractual_halo_lift": all_rank.get(key, 0.0) - organic_rank.get(key, 0.0),
            }
        )
    output.sort(key=lambda row: row["all_channel_pagerank"], reverse=True)
    write_csv(out / "brand_halo.csv", output)
    write_csv(out / "halo_edges.csv", edges)
    write_json(out / "summary.json", {"brands": len(output), "edges": len(edges), "confidence": args.confidence})
    print(f"Wrote halo decomposition for {len(output):,} brands to {out}")


if __name__ == "__main__":
    main()
