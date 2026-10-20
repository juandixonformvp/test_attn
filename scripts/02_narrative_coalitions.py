#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_graph.common import add_common_arguments, connect, prepare_output, read, write_csv, write_json  # noqa: E402


MEMBERSHIPS_QUERY = """
MATCH (a:Account)-[:COMMENTED]->(:Comment)-[e:EXPRESSES]->(:Claim)-[:IN_THEME]->(t:Theme)
WHERE coalesce(e.confidence, 1.0) >= $confidence
RETURN a.handle AS account, t.cluster AS theme, count(*) AS weight, 'comment' AS source
UNION ALL
MATCH (a:Account)-[:POSTED]->(p:Post)-[:IN_NARRATIVE]->(t:Theme)
WHERE coalesce(p.is_stub, false) = false
RETURN a.handle AS account, t.cluster AS theme, count(*) AS weight, 'post' AS source
"""

ACCOUNT_QUERY = """
MATCH (a:Account)
RETURN a.handle AS account, a.type AS account_type, a.platform AS platform,
       a.followers AS followers, a.verified AS verified
"""

THEME_QUERY = """
MATCH (t:Theme)
RETURN t.cluster AS theme, t.name AS name, t.description AS description
"""


def build_coalitions(memberships: list[dict]) -> tuple[nx.Graph, list[set[str]]]:
    weights: Counter = Counter()
    for row in memberships:
        weights[(row["account"], row["theme"])] += int(row["weight"])
    graph = nx.Graph()
    for (account, theme), weight in weights.items():
        graph.add_node(f"a:{account}", kind="account", value=account)
        graph.add_node(f"t:{theme}", kind="theme", value=theme)
        graph.add_edge(f"a:{account}", f"t:{theme}", weight=weight)
    communities = list(nx.community.louvain_communities(graph, weight="weight", seed=42)) if graph else []
    return graph, communities


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect account–narrative coalitions")
    add_common_arguments(parser)
    args = parser.parse_args()
    out = prepare_output(args.output_dir, "narrative_coalitions")
    with connect(args) as driver:
        memberships = read(driver, args.database, MEMBERSHIPS_QUERY, confidence=args.confidence)
        accounts = {r["account"]: r for r in read(driver, args.database, ACCOUNT_QUERY)}
        themes = {r["theme"]: r for r in read(driver, args.database, THEME_QUERY)}

    graph, communities = build_coalitions(memberships)
    member_rows, community_rows = [], []
    for community_id, members in enumerate(sorted(communities, key=len, reverse=True), 1):
        account_ids = [graph.nodes[n]["value"] for n in members if graph.nodes[n]["kind"] == "account"]
        theme_ids = [graph.nodes[n]["value"] for n in members if graph.nodes[n]["kind"] == "theme"]
        types = Counter(accounts.get(a, {}).get("account_type") or "unknown" for a in account_ids)
        platforms = Counter(accounts.get(a, {}).get("platform") or "unknown" for a in account_ids)
        community_rows.append(
            {
                "community": community_id,
                "accounts": len(account_ids),
                "themes": len(theme_ids),
                "top_account_types": ";".join(f"{k}:{v}" for k, v in types.most_common(5)),
                "top_platforms": ";".join(f"{k}:{v}" for k, v in platforms.most_common(5)),
                "theme_names": ";".join(themes.get(t, {}).get("name") or str(t) for t in theme_ids),
            }
        )
        for account in account_ids:
            weighted_degree = graph.degree(f"a:{account}", weight="weight")
            member_rows.append({"community": community_id, **accounts.get(account, {"account": account}), "activity_weight": weighted_degree})

    write_csv(out / "coalitions.csv", community_rows)
    write_csv(out / "account_memberships.csv", sorted(member_rows, key=lambda r: r["activity_weight"], reverse=True))
    write_json(out / "summary.json", {"communities": len(communities), "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()})
    print(f"Wrote {len(communities):,} coalitions to {out}")


if __name__ == "__main__":
    main()
