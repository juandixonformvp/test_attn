#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_graph.common import add_common_arguments, connect, prepare_output, read, write_csv, write_json  # noqa: E402


TIMELINE_QUERY = """
MATCH (p:Post)-[:IN_NARRATIVE]->(t:Theme)
WHERE p.created_at IS NOT NULL AND coalesce(p.is_stub, false) = false
RETURN t.cluster AS theme, t.name AS theme_name, p.platform AS platform,
       substring(toString(p.created_at), 0, 10) AS day, count(DISTINCT p) AS activity, 'post' AS source
UNION ALL
MATCH (c:Comment)-[e:EXPRESSES]->(:Claim)-[:IN_THEME]->(t:Theme)
WHERE c.created_at IS NOT NULL AND coalesce(e.confidence, 1.0) >= $confidence
RETURN t.cluster AS theme, t.name AS theme_name, c.platform AS platform,
       substring(toString(c.created_at), 0, 10) AS day, count(DISTINCT c) AS activity, 'comment' AS source
"""


def parse_day(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate narrative timing and platform lead/lag")
    add_common_arguments(parser)
    args = parser.parse_args()
    out = prepare_output(args.output_dir, "cross_platform_migration")
    with connect(args) as driver:
        raw = read(driver, args.database, TIMELINE_QUERY, confidence=args.confidence)

    series: dict[tuple, dict[date, int]] = defaultdict(lambda: defaultdict(int))
    theme_names = {}
    for row in raw:
        parsed = parse_day(row["day"])
        if parsed and row["platform"]:
            series[(row["theme"], row["platform"])][parsed] += row["activity"]
            theme_names[row["theme"]] = row["theme_name"]
    timeline_rows, summary_rows = [], []
    starts_by_theme: dict[object, list[date]] = defaultdict(list)
    interim = []
    for (theme, platform), days in series.items():
        values = list(days.values())
        threshold = max(3.0, statistics.mean(values) + (2 * statistics.pstdev(values) if len(values) > 1 else 0))
        active_days = sorted(day for day, count in days.items() if count >= threshold)
        start = active_days[0] if active_days else min(days)
        peak_day, peak_count = max(days.items(), key=lambda item: item[1])
        starts_by_theme[theme].append(start)
        interim.append((theme, platform, start, peak_day, peak_count, sum(values), threshold))
        for day, count in sorted(days.items()):
            timeline_rows.append({"theme": theme, "theme_name": theme_names.get(theme), "platform": platform, "day": day, "activity": count})
    for theme, platform, start, peak_day, peak_count, total, threshold in interim:
        earliest = min(starts_by_theme[theme])
        summary_rows.append(
            {
                "theme": theme,
                "theme_name": theme_names.get(theme),
                "platform": platform,
                "burst_start": start,
                "lag_days_from_first_platform": (start - earliest).days,
                "peak_day": peak_day,
                "peak_activity": peak_count,
                "total_activity": total,
                "burst_threshold": round(threshold, 3),
            }
        )
    summary_rows.sort(key=lambda row: (str(row["theme"]), row["lag_days_from_first_platform"], -row["total_activity"]))
    write_csv(out / "daily_activity.csv", timeline_rows)
    write_csv(out / "platform_lags.csv", summary_rows)
    write_json(out / "summary.json", {"series": len(series), "themes": len(starts_by_theme), "confidence": args.confidence})
    print(f"Wrote {len(series):,} platform–narrative series to {out}")


if __name__ == "__main__":
    main()
