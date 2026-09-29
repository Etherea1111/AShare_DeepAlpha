#!/usr/bin/env python3
"""Filter a raw price panel with dated industry membership snapshots."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAILY_INPUT = PROJECT_ROOT / "data/j66_money_finance_daily_2018_2024.csv"
DEFAULT_MEMBERSHIP_INPUT = (
    PROJECT_ROOT / "data/j66_money_finance_constituent_snapshots_2018_2024.csv"
)
DEFAULT_DAILY_OUTPUT = (
    PROJECT_ROOT / "data/point_in_time/j66_money_finance_daily_2018_2024.csv"
)
DEFAULT_REPORT_OUTPUT = (
    PROJECT_ROOT / "data/point_in_time/point_in_time_panel_report.json"
)
MEMBERSHIP_FIELDS = {
    "snapshot_date",
    "effective_date",
    "code",
    "industry_code",
}
MEMBERSHIP_OUTPUT_FIELDS = [
    "membership_asof_date",
    "membership_snapshot_date",
    "membership_effective_date",
    "membership_source",
    "point_in_time_member",
]


# 解析行情、点时成员快照和输出路径。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply dated industry membership snapshots without using future "
            "classification information."
        )
    )
    parser.add_argument("--daily-input", type=Path, default=DEFAULT_DAILY_INPUT)
    parser.add_argument(
        "--membership-input",
        type=Path,
        default=DEFAULT_MEMBERSHIP_INPUT,
    )
    parser.add_argument("--daily-output", type=Path, default=DEFAULT_DAILY_OUTPUT)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=DEFAULT_REPORT_OUTPUT,
    )
    parser.add_argument("--industry-code", default="J66")
    return parser.parse_args()


# 读取 CSV 并校验表头。
def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames or []
        return fields, list(reader)


def parse_iso_date(value: object, context: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"Invalid date in {context}: {value!r}")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO date in {context}: {value!r}") from exc
    if parsed.isoformat() != value:
        raise ValueError(
            f"Date must use YYYY-MM-DD in {context}: {value!r}"
        )
    return parsed


# 将点时成员快照索引为 code -> 按生效日期排序的记录。
def load_membership(
    path: Path,
    industry_code: str,
) -> dict[str, list[dict[str, str]]]:
    fields, rows = read_csv(path)
    missing = MEMBERSHIP_FIELDS - set(fields)
    if missing:
        raise ValueError(f"{path} is missing membership fields: {sorted(missing)}")
    membership: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if row["industry_code"] != industry_code:
            continue
        code = row.get("code", "").strip()
        if not code:
            raise ValueError(f"Empty membership code in {path}.")
        snapshot_date = parse_iso_date(
            row["snapshot_date"],
            f"{path} membership snapshot",
        )
        effective_date = parse_iso_date(
            row["effective_date"],
            f"{path} membership event",
        )
        if effective_date > snapshot_date:
            raise ValueError(
                "Membership effective_date cannot be after snapshot_date: "
                f"{code} {effective_date} {snapshot_date}"
            )
        key = (code, row["effective_date"])
        if key in seen:
            raise ValueError(f"Duplicate membership snapshot: {key}")
        seen.add(key)
        normalized = dict(row)
        normalized["code"] = code
        normalized["_snapshot_date"] = snapshot_date.isoformat()
        normalized["_effective_date"] = effective_date.isoformat()
        membership[code].append(normalized)
    for code in membership:
        membership[code].sort(
            key=lambda row: (row["_effective_date"], row["_snapshot_date"])
        )
    if not membership:
        raise ValueError(f"No {industry_code} membership rows found in {path}.")
    return membership


# 返回某只股票在指定日期可见的最新成员快照。
def membership_asof(
    events: list[dict[str, str]] | None,
    asof_date: str,
) -> dict[str, str] | None:
    if not events:
        return None
    parse_iso_date(asof_date, "daily row")
    effective_dates = [event["_effective_date"] for event in events]
    position = bisect.bisect_right(effective_dates, asof_date) - 1
    while position >= 0:
        event = events[position]
        # Do not use an old effective date before the snapshot that revealed it.
        if event["_snapshot_date"] <= asof_date:
            return event
        position -= 1
    return None


# 生成过滤后的点时行情面板和审计报告。
def build_panel(
    daily_input: Path,
    membership_input: Path,
    daily_output: Path,
    report_output: Path,
    industry_code: str,
) -> dict[str, Any]:
    daily_fields, daily_rows = read_csv(daily_input)
    if not {"date", "code"}.issubset(daily_fields):
        raise ValueError(f"{daily_input} must contain date and code.")
    membership = load_membership(membership_input, industry_code)
    output_fields = daily_fields + [
        field for field in MEMBERSHIP_OUTPUT_FIELDS if field not in daily_fields
    ]
    output_rows: list[dict[str, str]] = []
    counts = Counter()
    for row in daily_rows:
        counts["input_rows"] += 1
        code = row.get("code", "").strip()
        if not code:
            counts["dropped_invalid_identity"] += 1
            continue
        parse_iso_date(row.get("date"), f"{daily_input} daily row")
        selected = membership_asof(membership.get(code), row["date"])
        if selected is None:
            counts["dropped_without_point_in_time_membership"] += 1
            continue
        output = dict(row)
        output.update(
            {
                "membership_asof_date": row["date"],
                "membership_snapshot_date": selected["_snapshot_date"],
                "membership_effective_date": selected["_effective_date"],
                "membership_source": selected.get("source", "unknown"),
                "point_in_time_member": "1",
            }
        )
        output_rows.append(output)
        counts["output_rows"] += 1

    daily_output.parent.mkdir(parents=True, exist_ok=True)
    with daily_output.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=output_fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(output_rows)
    report = {
        "source_daily_file": str(daily_input),
        "membership_file": str(membership_input),
        "industry_code": industry_code,
        "membership_rule": (
            "For each row, use the latest membership event whose effective "
            "date and source snapshot date are both not after the row date."
        ),
        "counts": {key: int(value) for key, value in sorted(counts.items())},
        "unique_input_codes": len({row["code"] for row in daily_rows}),
        "unique_output_codes": len({row["code"] for row in output_rows}),
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


# 执行点时面板构建。
def main() -> int:
    args = parse_args()
    report = build_panel(
        args.daily_input,
        args.membership_input,
        args.daily_output,
        args.report_output,
        args.industry_code,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
