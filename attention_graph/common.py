from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from neo4j import GraphDatabase, READ_ACCESS


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI"))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.75,
        help="Minimum inferred-edge confidence (default: 0.75)",
    )


def connect(args: argparse.Namespace):
    if not args.uri:
        raise SystemExit("Set NEO4J_URI or pass --uri")
    password = os.getenv("NEO4J_PASSWORD")
    auth = (args.user, password) if password else None
    driver = GraphDatabase.driver(args.uri, auth=auth, connection_timeout=15)
    driver.verify_connectivity()
    return driver


def read(driver, database: str, query: str, **parameters: Any) -> list[dict[str, Any]]:
    with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
        return [record.data() for record in session.run(query, **parameters)]


def prepare_output(base: Path, analysis: str) -> Path:
    path = base / analysis
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in materialized:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def normalized_entropy(counts: Iterable[float]) -> float:
    import math

    values = [float(v) for v in counts if v and v > 0]
    if len(values) <= 1:
        return 0.0
    total = sum(values)
    entropy = -sum((v / total) * math.log(v / total) for v in values)
    return entropy / math.log(len(values))


def gini(values: Iterable[float]) -> float:
    ordered = sorted(float(v) for v in values if v >= 0)
    if not ordered or sum(ordered) == 0:
        return 0.0
    n = len(ordered)
    weighted = sum((i + 1) * value for i, value in enumerate(ordered))
    return (2 * weighted) / (n * sum(ordered)) - (n + 1) / n


def js_divergence(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    import math

    keys = set(left) | set(right)
    lsum, rsum = sum(left.values()), sum(right.values())
    if not lsum or not rsum:
        return 0.0
    p = {k: left.get(k, 0.0) / lsum for k in keys}
    q = {k: right.get(k, 0.0) / rsum for k in keys}
    m = {k: (p[k] + q[k]) / 2 for k in keys}

    def kl(a: Mapping[str, float], b: Mapping[str, float]) -> float:
        return sum(a[k] * math.log2(a[k] / b[k]) for k in keys if a[k] > 0)

    return (kl(p, m) + kl(q, m)) / 2
