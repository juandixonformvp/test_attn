#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_graph.common import add_common_arguments, connect, prepare_output, read, write_csv, write_json  # noqa: E402


ACCOUNT_THEME_QUERY = """
MATCH (a:Account)-[:COMMENTED]->(:Comment)-[e:EXPRESSES]->(:Claim)-[:IN_THEME]->(t:Theme)
WHERE coalesce(e.confidence, 1.0) >= $confidence
RETURN a.handle AS account, a.type AS account_type, a.platform AS platform,
       t.cluster AS theme, count(*) AS weight
"""

CLAIM_GRAPH_QUERY = """
MATCH (a:Claim)-[r:SUPPORTS|ATTACKS|QUALIFIES|CONTRADICTS]->(b:Claim)
WHERE coalesce(r.confidence, 1.0) >= $confidence
OPTIONAL MATCH (a)-[:IN_THEME]->(ta:Theme)
OPTIONAL MATCH (b)-[:IN_THEME]->(tb:Theme)
RETURN a.claim_id AS source, a.claim_text AS source_text, ta.cluster AS source_theme,
       b.claim_id AS target, b.claim_text AS target_text, tb.cluster AS target_theme,
       type(r) AS relation, coalesce(r.weight, 1.0) * coalesce(r.confidence, 1.0) AS weight
"""


def participation(weights: Counter) -> float:
    total = sum(weights.values())
    return 0.0 if total == 0 else 1 - sum((value / total) ** 2 for value in weights.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Find accounts and claims that bridge narratives")
    add_common_arguments(parser)
    parser.add_argument("--betweenness-sample", type=int, default=500)
    args = parser.parse_args()
    out = prepare_output(args.output_dir, "bridges")
    with connect(args) as driver:
        account_theme = read(driver, args.database, ACCOUNT_THEME_QUERY, confidence=args.confidence)
        relations = read(driver, args.database, CLAIM_GRAPH_QUERY, confidence=args.confidence)

    by_account: dict[str, Counter] = defaultdict(Counter)
    account_meta: dict[str, dict] = {}
    for row in account_theme:
        by_account[row["account"]][row["theme"]] += row["weight"]
        account_meta[row["account"]] = {"account_type": row["account_type"], "platform": row["platform"]}
    account_rows = [
        {
            "account": account,
            **account_meta[account],
            "themes": len(weights),
            "activity": sum(weights.values()),
            "participation_coefficient": participation(weights),
            "theme_weights": ";".join(f"{k}:{v}" for k, v in weights.most_common()),
        }
        for account, weights in by_account.items()
    ]
    account_rows.sort(key=lambda r: (r["participation_coefficient"], r["activity"]), reverse=True)

    graph = nx.DiGraph()
    metadata: dict[str, dict] = {}
    cross_theme: Counter = Counter()
    for row in relations:
        source, target = row["source"], row["target"]
        graph.add_edge(source, target, weight=float(row["weight"]), relation=row["relation"])
        metadata[source] = {"claim_text": row["source_text"], "theme": row["source_theme"]}
        metadata[target] = {"claim_text": row["target_text"], "theme": row["target_theme"]}
        if row["source_theme"] != row["target_theme"]:
            cross_theme[source] += 1
            cross_theme[target] += 1
    k = min(args.betweenness_sample, graph.number_of_nodes())
    betweenness = nx.betweenness_centrality(graph, k=k, seed=42, weight=None) if k else {}
    articulation = set(nx.articulation_points(graph.to_undirected())) if graph.number_of_nodes() else set()
    claim_rows = [
        {
            "claim_id": node,
            **metadata.get(node, {}),
            "betweenness": score,
            "cross_theme_edges": cross_theme[node],
            "articulation_point": node in articulation,
            "degree": graph.degree(node),
        }
        for node, score in betweenness.items()
    ]
    claim_rows.sort(key=lambda r: (r["betweenness"], r["cross_theme_edges"]), reverse=True)
    write_csv(out / "account_bridges.csv", account_rows)
    write_csv(out / "claim_bridges.csv", claim_rows)
    write_json(out / "summary.json", {"accounts": len(account_rows), "claims": graph.number_of_nodes(), "relations": graph.number_of_edges(), "betweenness_sample": k})
    print(f"Wrote bridge rankings to {out}")


if __name__ == "__main__":
    main()
