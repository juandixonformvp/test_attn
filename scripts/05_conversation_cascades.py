#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_graph.common import add_common_arguments, connect, prepare_output, read, write_csv, write_json  # noqa: E402


CASCADES_QUERY = """
MATCH (c:Comment)-[:ON]->(p:Post)
OPTIONAL MATCH (c)-[:REPLY_TO]->(parent:Comment)
OPTIONAL MATCH (a:Account)-[:COMMENTED]->(c)
RETURN p.post_id AS post_id, p.platform AS platform, p.likes AS post_likes,
       p.views AS post_views, p.shares AS post_shares, p.created_at AS post_created_at,
       coalesce(p.is_stub, false) AS is_stub,
       c.comment_id AS comment_id, parent.comment_id AS parent_id,
       a.handle AS account, c.likes AS comment_likes, c.created_at AS comment_created_at
"""


def cascade_metrics(rows: list[dict]) -> dict:
    root = "__root__"
    parent_of = {row["comment_id"]: row["parent_id"] or root for row in rows}
    children: dict[str, list[str]] = defaultdict(list)
    for child, parent in parent_of.items():
        if parent not in parent_of and parent != root:
            parent = root
            parent_of[child] = root
        children[parent].append(child)

    depths = {root: 0}
    stack = [root]
    while stack:
        parent = stack.pop()
        for child in children.get(parent, []):
            if child not in depths:
                depths[child] = depths[parent] + 1
                stack.append(child)
    for node in parent_of:
        depths.setdefault(node, 1)
    breadth = Counter(depths.values())

    order = sorted(parent_of, key=lambda n: depths[n], reverse=True)
    subtree = Counter({node: 1 for node in parent_of})
    total_nodes = len(parent_of) + 1
    distance_sum = 0
    for node in order:
        size = subtree[node]
        distance_sum += size * (total_nodes - size)
        parent = parent_of[node]
        subtree[parent] += size
    virality = 0.0 if total_nodes < 2 else 2 * distance_sum / (total_nodes * (total_nodes - 1))
    comments = len(rows)
    unique_accounts = len({row["account"] for row in rows if row["account"]})
    max_depth = max(depths.values(), default=0)
    max_width = max((v for depth, v in breadth.items() if depth > 0), default=0)
    if comments < 3:
        shape = "low_volume"
    elif unique_accounts <= max(2, comments * 0.2):
        shape = "concentrated_back_and_forth"
    elif max_depth >= 5 and max_width < comments * 0.5:
        shape = "deep_debate"
    elif max_width >= comments * 0.6:
        shape = "shallow_pile_on"
    else:
        shape = "branching_discussion"
    return {
        "comments": comments,
        "unique_accounts": unique_accounts,
        "max_depth": max_depth,
        "max_width": max_width,
        "mean_branching": comments / max(1, sum(1 for node in children if children[node])),
        "structural_virality": virality,
        "shape": shape,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure reply-tree depth, breadth, and structural virality")
    add_common_arguments(parser)
    args = parser.parse_args()
    out = prepare_output(args.output_dir, "conversation_cascades")
    with connect(args) as driver:
        rows = read(driver, args.database, CASCADES_QUERY)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["post_id"]].append(row)
    results = []
    for post_id, items in grouped.items():
        first = items[0]
        results.append(
            {
                "post_id": post_id,
                "platform": first["platform"],
                "post_likes": first["post_likes"],
                "post_views": first["post_views"],
                "post_shares": first["post_shares"],
                "post_created_at": first["post_created_at"],
                "is_stub": first["is_stub"],
                **cascade_metrics(items),
            }
        )
    results.sort(key=lambda row: (row["structural_virality"], row["comments"]), reverse=True)
    shape_counts = Counter(row["shape"] for row in results)
    write_csv(out / "cascades.csv", results)
    write_json(out / "summary.json", {"posts": len(results), "comments": len(rows), "shape_counts": shape_counts})
    print(f"Wrote {len(results):,} cascade profiles to {out}")


if __name__ == "__main__":
    main()
