#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_graph.common import (  # noqa: E402
    add_common_arguments,
    connect,
    js_divergence,
    normalized_entropy,
    prepare_output,
    read,
    write_csv,
    write_json,
)


REACTIONS_QUERY = """
MATCH (a:Account)-[:COMMENTED]->(c:Comment)-[r:REACTS]->(target)
WHERE coalesce(r.confidence, 1.0) >= $confidence
RETURN coalesce(c.platform, a.platform, 'unknown') AS platform,
       coalesce(a.type, 'unknown') AS account_type,
       head(labels(target)) AS target_label,
       coalesce(r.reaction_type, 'unknown') AS reaction_type,
       coalesce(r.stance, 'unknown') AS stance,
       coalesce(r.register, 'unknown') AS register,
       count(*) AS count
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure reaction polarization across platforms and actor types")
    add_common_arguments(parser)
    args = parser.parse_args()
    out = prepare_output(args.output_dir, "reaction_polarization")
    with connect(args) as driver:
        rows = read(driver, args.database, REACTIONS_QUERY, confidence=args.confidence)

    totals: dict[tuple, Counter] = defaultdict(Counter)
    for row in rows:
        key = (row["target_label"], row["platform"], row["account_type"])
        totals[key][row["reaction_type"]] += row["count"]
    distributions = []
    for (target, platform, account_type), counts in totals.items():
        total = sum(counts.values())
        for reaction, count in counts.items():
            distributions.append(
                {
                    "target_label": target,
                    "platform": platform,
                    "account_type": account_type,
                    "reaction_type": reaction,
                    "count": count,
                    "share": count / total,
                    "reaction_entropy": normalized_entropy(counts.values()),
                }
            )

    platform_vectors: dict[tuple, Counter] = defaultdict(Counter)
    for row in rows:
        platform_vectors[(row["target_label"], row["platform"])][row["reaction_type"]] += row["count"]
    divergence = []
    for target in sorted({key[0] for key in platform_vectors}):
        platforms = sorted(key[1] for key in platform_vectors if key[0] == target)
        for left, right in combinations(platforms, 2):
            divergence.append(
                {
                    "target_label": target,
                    "platform_a": left,
                    "platform_b": right,
                    "js_divergence": js_divergence(platform_vectors[(target, left)], platform_vectors[(target, right)]),
                    "reactions_a": sum(platform_vectors[(target, left)].values()),
                    "reactions_b": sum(platform_vectors[(target, right)].values()),
                }
            )
    divergence.sort(key=lambda row: row["js_divergence"], reverse=True)
    write_csv(out / "reaction_distributions.csv", distributions)
    write_csv(out / "platform_divergence.csv", divergence)
    write_csv(out / "raw_aggregates.csv", rows)
    write_json(out / "summary.json", {"aggregated_rows": len(rows), "comparisons": len(divergence), "confidence": args.confidence})
    print(f"Wrote reaction distributions and {len(divergence):,} comparisons to {out}")


if __name__ == "__main__":
    main()
