#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_graph.common import add_common_arguments, connect, gini, prepare_output, read, write_csv, write_json  # noqa: E402


EXPRESSIONS_QUERY = """
MATCH (a:Account)-[:COMMENTED]->(c:Comment)-[e:EXPRESSES]->(claim:Claim)
WHERE coalesce(e.confidence, 1.0) >= $confidence
RETURN a.handle AS account, a.type AS account_type, a.platform AS platform,
       a.followers AS followers, a.verified AS verified,
       claim.claim_id AS claim, count(*) AS expressions,
       sum(coalesce(c.likes, 0)) AS engagement
"""


def weighted_hits(edges: list[dict], iterations: int = 100, tolerance: float = 1e-10):
    accounts = sorted({row["account"] for row in edges})
    claims = sorted({row["claim"] for row in edges})
    hubs = {node: 1 / math.sqrt(max(1, len(accounts))) for node in accounts}
    authorities = {node: 1 / math.sqrt(max(1, len(claims))) for node in claims}
    by_account: dict[str, list[tuple[str, float]]] = defaultdict(list)
    by_claim: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in edges:
        weight = float(row["expressions"])
        by_account[row["account"]].append((row["claim"], weight))
        by_claim[row["claim"]].append((row["account"], weight))
    for _ in range(iterations):
        new_authorities = {claim: sum(hubs[a] * w for a, w in by_claim[claim]) for claim in claims}
        norm = math.sqrt(sum(v * v for v in new_authorities.values())) or 1
        new_authorities = {k: v / norm for k, v in new_authorities.items()}
        new_hubs = {account: sum(new_authorities[c] * w for c, w in by_account[account]) for account in accounts}
        norm = math.sqrt(sum(v * v for v in new_hubs.values())) or 1
        new_hubs = {k: v / norm for k, v in new_hubs.items()}
        delta = sum(abs(new_hubs[k] - hubs[k]) for k in hubs)
        hubs, authorities = new_hubs, new_authorities
        if delta < tolerance:
            break
    return hubs, authorities


def residuals(account_totals: dict[str, dict]) -> dict[str, float]:
    points = [(math.log1p(v["followers"] or 0), math.log1p(v["engagement"])) for v in account_totals.values()]
    if len(points) < 2:
        return {account: 0.0 for account in account_totals}
    xbar = sum(x for x, _ in points) / len(points)
    ybar = sum(y for _, y in points) / len(points)
    denominator = sum((x - xbar) ** 2 for x, _ in points)
    slope = sum((x - xbar) * (y - ybar) for x, y in points) / denominator if denominator else 0.0
    intercept = ybar - slope * xbar
    return {
        account: math.log1p(values["engagement"]) - (intercept + slope * math.log1p(values["followers"] or 0))
        for account, values in account_totals.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank amplifiers and measure claim-level concentration")
    add_common_arguments(parser)
    args = parser.parse_args()
    out = prepare_output(args.output_dir, "amplifier_analysis")
    with connect(args) as driver:
        rows = read(driver, args.database, EXPRESSIONS_QUERY, confidence=args.confidence)
    hubs, authorities = weighted_hits(rows)
    account_totals: dict[str, dict] = {}
    claim_accounts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        values = account_totals.setdefault(
            row["account"],
            {"account": row["account"], "account_type": row["account_type"], "platform": row["platform"],
             "followers": row["followers"] or 0, "verified": row["verified"], "expressions": 0, "engagement": 0, "claims": 0},
        )
        values["expressions"] += row["expressions"]
        values["engagement"] += row["engagement"] or 0
        values["claims"] += 1
        claim_accounts[row["claim"]][row["account"]] += row["expressions"]
    engagement_residuals = residuals(account_totals)
    account_rows = [
        {**values, "hits_hub_score": hubs.get(account, 0.0), "follower_adjusted_engagement": engagement_residuals[account]}
        for account, values in account_totals.items()
    ]
    account_rows.sort(key=lambda row: (row["hits_hub_score"], row["follower_adjusted_engagement"]), reverse=True)
    claim_rows = []
    for claim, counts in claim_accounts.items():
        total = sum(counts.values())
        shares = [value / total for value in counts.values()]
        claim_rows.append(
            {
                "claim_id": claim,
                "authority_score": authorities.get(claim, 0.0),
                "accounts": len(counts),
                "expressions": total,
                "hhi": sum(share * share for share in shares),
                "gini": gini(counts.values()),
                "top_account_share": max(shares),
            }
        )
    claim_rows.sort(key=lambda row: (row["hhi"], row["expressions"]), reverse=True)
    write_csv(out / "accounts_ranked.csv", account_rows)
    write_csv(out / "claim_concentration.csv", claim_rows)
    write_json(out / "summary.json", {"accounts": len(account_rows), "claims": len(claim_rows), "expression_pairs": len(rows)})
    print(f"Wrote amplifier rankings for {len(account_rows):,} accounts to {out}")


if __name__ == "__main__":
    main()
