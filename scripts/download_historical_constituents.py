#!/usr/bin/env python3
"""Download historical point-in-time industry membership snapshots from BaoStock."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = "2024-12-31"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/j66_money_finance_constituent_snapshots_2018_2024.csv"
)
INDUSTRY_CODE = "J66"
INDUSTRY_NAME = "货币金融服务"
OUTPUT_FIELDS = [
    "snapshot_date",
    "effective_date",
    "code",
    "code_name",
    "industry_code",
    "industry_name",
    "classification",
    "source_update_date",
    "source",
]
CODE_PATTERN = re.compile(r"^(sh|sz|bj)\.\d{6}$")
INDUSTRY_FIELD_ALIASES = ("industry", "industry_code", "industryCode")
CLASSIFICATION_FIELD_ALIASES = (
    "industryClassification",
    "industry_classification",
    "classification",
)
UPDATE_DATE_FIELD_ALIASES = ("updateDate", "update_date")


# 解析快照日期范围和输出路径。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download weekly historical J66 industry snapshots. The resulting "
            "file is consumed by the point-in-time panel builder."
        )
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    return parser.parse_args()


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


# 返回范围内每个周一，避免用当前分类快照回填整个历史区间。
def weekly_mondays(start_date: str, end_date: str) -> list[str]:
    start = parse_iso_date(start_date, "start-date")
    end = parse_iso_date(end_date, "end-date")
    if start > end:
        raise ValueError("start-date must not be later than end-date")
    first_monday = start + timedelta(days=(7 - start.weekday()) % 7)
    dates: list[str] = []
    current = first_monday
    while current <= end:
        dates.append(current.isoformat())
        current += timedelta(days=7)
    return dates


def first_value(row: dict[str, str], aliases: tuple[str, ...]) -> str:
    for field in aliases:
        value = row.get(field)
        if value:
            return value.strip()
    return ""


# 将 BaoStock 查询结果读取为字典记录。
def read_result(result: object) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    while result.next():
        rows.append(dict(zip(result.fields, result.get_row_data())))
    return rows


# 将某日行业查询结果规范化为可用于点时过滤的成员快照。
def normalize_snapshot_rows(
    snapshot_date: str,
    raw_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    parse_iso_date(snapshot_date, "snapshot date")
    output: list[dict[str, str]] = []
    seen_codes: set[str] = set()
    for row in raw_rows:
        industry = first_value(row, INDUSTRY_FIELD_ALIASES)
        if not industry.upper().startswith(INDUSTRY_CODE):
            continue
        code = first_value(row, ("code",)).lower()
        if not CODE_PATTERN.fullmatch(code):
            raise ValueError(
                f"Invalid BaoStock code in {snapshot_date}: {code!r}"
            )
        if code in seen_codes:
            raise ValueError(
                f"Duplicate code in {snapshot_date} industry snapshot: {code}"
            )
        seen_codes.add(code)
        source_update_date = first_value(row, UPDATE_DATE_FIELD_ALIASES)
        if source_update_date:
            parse_iso_date(source_update_date, "BaoStock updateDate")
        output.append(
            {
                "snapshot_date": snapshot_date,
                # Use the query date as the usable effective date. This is
                # conservative: updateDate may describe an older event that
                # was only revealed by a later snapshot.
                "effective_date": snapshot_date,
                "code": code,
                "code_name": first_value(row, ("code_name", "codeName")),
                "industry_code": INDUSTRY_CODE,
                "industry_name": INDUSTRY_NAME,
                "classification": first_value(
                    row,
                    CLASSIFICATION_FIELD_ALIASES,
                )
                or "BaoStock",
                "source_update_date": source_update_date,
                "source": "BaoStock.query_stock_industry(date=...)",
            }
        )
    return output


# 查询指定日期的行业分类并筛选 J66。
def query_snapshot(
    snapshot_date: str,
    client: Any | None = None,
) -> list[dict[str, str]]:
    if client is None:
        try:
            import baostock as client
        except ImportError as exc:
            raise RuntimeError(
                "BaoStock is required for historical constituent downloads."
            ) from exc
    parameters = inspect.signature(client.query_stock_industry).parameters
    if "date" not in parameters:
        raise RuntimeError(
            "Installed BaoStock does not expose dated query_stock_industry(); "
            "a point-in-time membership source is required."
        )
    result = client.query_stock_industry(date=snapshot_date)
    if result.error_code != "0":
        raise RuntimeError(
            f"Industry snapshot failed for {snapshot_date}: "
            f"{result.error_code} {result.error_msg}"
        )
    return normalize_snapshot_rows(snapshot_date, read_result(result))


# 登录后下载所有周度行业成员快照。
def download_snapshots(
    args: argparse.Namespace,
    query: Any = query_snapshot,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for snapshot_date in weekly_mondays(args.start_date, args.end_date):
        last_error: Exception | None = None
        for attempt in range(1, args.retries + 1):
            try:
                snapshot_rows = query(snapshot_date)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < args.retries:
                    time.sleep(attempt * args.retry_delay)
        if last_error is not None:
            raise RuntimeError(
                f"Failed to download membership snapshot {snapshot_date}"
            ) from last_error
        rows.extend(snapshot_rows)
        print(f"{snapshot_date}: {len(snapshot_rows)} J66 members", flush=True)
    return rows


# 写出可用于 as-of 过滤的成员快照文件及元数据。
def main() -> int:
    args = parse_args()
    try:
        import baostock as bs
    except ImportError as exc:
        raise RuntimeError(
            "BaoStock is required for historical constituent downloads."
        ) from exc

    login_result = bs.login()
    if login_result.error_code != "0":
        raise RuntimeError(
            f"BaoStock login failed: {login_result.error_code} "
            f"{login_result.error_msg}"
        )
    try:
        rows = download_snapshots(
            args,
            query=lambda snapshot_date: query_snapshot(
                snapshot_date,
                client=bs,
            ),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(
                target,
                fieldnames=OUTPUT_FIELDS,
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
        metadata = {
            "source": "BaoStock",
            "industry_code": INDUSTRY_CODE,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "snapshot_frequency": "weekly_monday",
            "snapshot_count": len(set(row["snapshot_date"] for row in rows)),
            "row_count": len(rows),
            "output": str(args.output),
        }
        args.output.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0
    finally:
        bs.logout()


if __name__ == "__main__":
    raise SystemExit(main())
