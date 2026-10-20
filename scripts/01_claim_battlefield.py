#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
import sys

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_graph.common import (  # noqa: E402
    add_common_arguments,
    connect,
    normalized_entropy,
    prepare_output,
    read,
    write_csv,
    write_json,
)


CLAIMS_QUERY = """
MATCH (c:Claim)
CALL {
  WITH c
  OPTIONAL MATCH (a:Account)-[:COMMENTED]->(:Comment)-[e:EXPRESSES]->(c)
  WHERE coalesce(e.confidence, 1.0) >= $confidence
  RETURN count(e) AS expressions, count(DISTINCT a) AS distinct_accounts,
         collect(toLower(coalesce(e.stance, 'unknown'))) AS stances,
         avg(coalesce(e.certainty, 0.0)) AS avg_certainty,
         avg(coalesce(e.sarcasm_prob, 0.0)) AS avg_sarcasm
}
RETURN c.claim_id AS claim_id, c.claim_text AS claim_text,
       c.claim_type AS claim_type, c.post_count AS post_count,
       c.account_count AS source_account_count,
       expressions, distinct_accounts, stances, avg_certainty, avg_sarcasm
"""

RELATIONS_QUERY = """
MATCH (a:Claim)-[r:SUPPORTS|ATTACKS|CONTRADICTS]->(b:Claim)
WHERE coalesce(r.confidence, 1.0) >= $confidence
RETURN a.claim_id AS source, b.claim_id AS target, type(r) AS relation,
       coalesce(r.weight, 1.0) * coalesce(r.confidence, 1.0) AS weight
"""


def pagerank_by_sign(relations: list[dict]) -> tuple[dict, dict, Counter, Counter]:
    positive = nx.DiGraph()
    negative = nx.DiGraph()
    positive_in: Counter = Counter()
    negative_in: Counter = Counter()
    for row in relations:
        graph = positive if row["relation"] == "SUPPORTS" else negative
        weight = float(row["weight"] or 0)
        source, target = row["source"], row["target"]
        graph.add_edge(source, target, weight=graph.get_edge_data(source, target, {}).get("weight", 0) + weight)
        (positive_in if graph is positive else negative_in)[target] += weight
    return (
        nx.pagerank(positive, weight="weight") if positive else {},
        nx.pagerank(negative, weight="weight") if negative else {},
        positive_in,
        negative_in,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank claims by contestation and signed influence")
    add_common_arguments(parser)
    args = parser.parse_args()
    out = prepare_output(args.output_dir, "claim_battlefield")

    with connect(args) as driver:
        claims = read(driver, args.database, CLAIMS_QUERY, confidence=args.confidence)
        relations = read(driver, args.database, RELATIONS_QUERY, confidence=args.confidence)

    positive_pr, negative_pr, positive_in, negative_in = pagerank_by_sign(relations)
    output = []
    for claim in claims:
        stance_counts = Counter(claim.pop("stances"))
        entropy = normalized_entropy(stance_counts.values())
        accounts = claim.get("distinct_accounts") or claim.get("source_account_count") or 0
        incoming_negative = negative_in[claim["claim_id"]]
        contestation = math.log1p(accounts) * entropy * math.log1p(incoming_negative)
        output.append(
            {
                **claim,
                "stance_entropy": round(entropy, 6),
                "stance_counts": ";".join(f"{k}:{v}" for k, v in sorted(stance_counts.items())),
                "support_weight_in": round(positive_in[claim["claim_id"]], 6),
                "attack_weight_in": round(incoming_negative, 6),
                "support_pagerank": positive_pr.get(claim["claim_id"], 0.0),
                "attack_pagerank": negative_pr.get(claim["claim_id"], 0.0),
                "contestation_score": round(contestation, 6),
            }
        )
    output.sort(key=lambda row: row["contestation_score"], reverse=True)
    write_csv(out / "claims_ranked.csv", output)
    write_json(out / "summary.json", {"claims": len(output), "relations": len(relations), "confidence": args.confidence})
    print(f"Wrote {len(output):,} claims to {out}")


if __name__ == "__main__":
    main()
