#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
import sys

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_graph.common import add_common_arguments, connect, prepare_output, read, write_csv, write_json  # noqa: E402


RELATIONS_QUERY = """
MATCH (a:Claim)-[r:SUPPORTS|ATTACKS|QUALIFIES|CONTRADICTS|RESTATES]->(b:Claim)
WHERE coalesce(r.confidence, 1.0) >= $confidence
RETURN a.claim_id AS source, a.claim_text AS source_text,
       b.claim_id AS target, b.claim_text AS target_text,
       type(r) AS relation,
       coalesce(r.weight, 1.0) * coalesce(r.confidence, 1.0) AS weight
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze signed cycles and contradiction components")
    add_common_arguments(parser)
    args = parser.parse_args()
    out = prepare_output(args.output_dir, "contradiction_structure")
    with connect(args) as driver:
        rows = read(driver, args.database, RELATIONS_QUERY, confidence=args.confidence)

    directed = nx.DiGraph()
    signed_weight: Counter = Counter()
    metadata = {}
    negative_degree: Counter = Counter()
    for row in rows:
        source, target = row["source"], row["target"]
        metadata[source], metadata[target] = row["source_text"], row["target_text"]
        if row["relation"] != "RESTATES":
            directed.add_edge(source, target, relation=row["relation"], weight=row["weight"])
        if row["relation"] in {"SUPPORTS", "ATTACKS", "CONTRADICTS"}:
            pair = tuple(sorted((source, target)))
            sign = 1 if row["relation"] == "SUPPORTS" else -1
            signed_weight[pair] += sign * float(row["weight"])
            if sign < 0:
                negative_degree[source] += float(row["weight"])
                negative_degree[target] += float(row["weight"])

    signed = nx.Graph()
    for (source, target), value in signed_weight.items():
        if value:
            signed.add_edge(source, target, sign=1 if value > 0 else -1, weight=abs(value))
    components = sorted(nx.strongly_connected_components(directed), key=len, reverse=True)
    component_id = {node: i for i, nodes in enumerate(components, 1) for node in nodes}

    cycle_rows = []
    for component in nx.connected_components(signed):
        subgraph = signed.subgraph(component)
        for cycle in nx.cycle_basis(subgraph):
            product = 1
            for index, source in enumerate(cycle):
                target = cycle[(index + 1) % len(cycle)]
                product *= signed[source][target]["sign"]
            cycle_rows.append({"length": len(cycle), "balanced": product > 0, "claims": ";".join(cycle)})

    triad_rows = []
    seen_triads = set()
    for node in signed:
        neighbors = sorted(signed.neighbors(node))
        for left, right in combinations(neighbors, 2):
            if not signed.has_edge(left, right):
                continue
            triad = tuple(sorted((node, left, right)))
            if triad in seen_triads:
                continue
            seen_triads.add(triad)
            sign_product = signed[node][left]["sign"] * signed[node][right]["sign"] * signed[left][right]["sign"]
            triad_rows.append({"claim_a": triad[0], "claim_b": triad[1], "claim_c": triad[2], "balanced": sign_product > 0})

    claim_rows = [
        {
            "claim_id": node,
            "claim_text": metadata.get(node),
            "strong_component": component_id.get(node),
            "strong_component_size": len(components[component_id[node] - 1]) if node in component_id else 0,
            "negative_weight": negative_degree[node],
            "in_degree": directed.in_degree(node),
            "out_degree": directed.out_degree(node),
        }
        for node in directed
    ]
    claim_rows.sort(key=lambda row: (row["negative_weight"], row["strong_component_size"]), reverse=True)
    write_csv(out / "claims_ranked.csv", claim_rows)
    write_csv(out / "signed_cycles.csv", cycle_rows)
    write_csv(out / "signed_triads.csv", triad_rows)
    write_json(
        out / "summary.json",
        {
            "claims": directed.number_of_nodes(),
            "relations": directed.number_of_edges(),
            "strong_components": len(components),
            "cycles": len(cycle_rows),
            "unbalanced_cycles": sum(not row["balanced"] for row in cycle_rows),
            "triads": len(triad_rows),
            "unbalanced_triads": sum(not row["balanced"] for row in triad_rows),
        },
    )
    print(f"Wrote signed-structure analysis to {out}")


if __name__ == "__main__":
    main()
