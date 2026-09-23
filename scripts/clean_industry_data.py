#!/usr/bin/env python3
"""Clean and normalize the J66 industry CSV datasets without altering raw files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Callable


RAW_DAILY_PATH = Path("data/j66_money_finance_daily_2018_2024.csv")
RAW_CONSTITUENTS_PATH = Path("data/j66_money_finance_constituents.csv")
CLEAN_DAILY_PATH = Path("data/cleaned/j66_money_finance_daily_2018_2024.csv")
CLEAN_CONSTITUENTS_PATH = Path("data/cleaned/j66_money_finance_constituents.csv")
CLEAN_REPORT_PATH = Path("data/cleaned/cleaning_report.json")

DAILY_FIELDS = [
    "date",
    "code",
    "code_name",
    "industry_code",
    "industry_name",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "turn",
    "pctChg",
    "tradestatus",
    "isST",
    "adjustflag",
    "adjustment",
    "data_source",
]
CONSTITUENTS_FIELDS = [
    "code",
    "code_name",
    "industry_code",
    "industry_name",
    "classification",
    "updateDate",
]
MISSING_VALUES = {"", "na", "n/a", "nan", "null"}
CODE_PATTERN = re.compile(r"^(sh|sz|bj)\.?(?P<number>\d{6})$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove incomplete records and normalize J66 CSV field formats."
    )
    parser.add_argument("--daily-input", type=Path, default=RAW_DAILY_PATH)
    parser.add_argument(
        "--constituents-input", type=Path, default=RAW_CONSTITUENTS_PATH
    )
    parser.add_argument("--daily-output", type=Path, default=CLEAN_DAILY_PATH)
    parser.add_argument(
        "--constituents-output", type=Path, default=CLEAN_CONSTITUENTS_PATH
    )
    parser.add_argument("--report-output", type=Path, default=CLEAN_REPORT_PATH)
    return parser.parse_args()


def is_missing(value: str | None) -> bool:
    return value is None or value.strip().lower() in MISSING_VALUES


def normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def normalize_date(value: str) -> str:
    raw_value = normalize_text(value)
    for date_format in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d"):
        try:
            return datetime.strptime(raw_value, date_format).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"Invalid date: {value!r}")


def normalize_code(value: str) -> str:
    match = CODE_PATTERN.fullmatch(normalize_text(value))
    if not match:
        raise ValueError(f"Invalid stock code: {value!r}")
    return f"{match.group(1).lower()}.{match.group('number')}"


def parse_decimal(value: str) -> Decimal:
    try:
        number = Decimal(normalize_text(value).replace(",", ""))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid number: {value!r}") from exc
    if not number.is_finite():
        raise ValueError(f"Non-finite number: {value!r}")
    return number


def normalize_decimal(value: str, places: int) -> str:
    scale = Decimal(1).scaleb(-places)
    number = parse_decimal(value).quantize(scale, rounding=ROUND_HALF_UP)
    return format(number, f".{places}f")


def normalize_integer(value: str) -> str:
    number = parse_decimal(value)
    if number != number.to_integral_value():
        raise ValueError(f"Expected integer: {value!r}")
    return str(int(number))


def normalize_daily_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "date": normalize_date(row["date"]),
        "code": normalize_code(row["code"]),
        "code_name": normalize_text(row["code_name"]),
        "industry_code": normalize_text(row["industry_code"]).upper(),
        "industry_name": normalize_text(row["industry_name"]),
        "open": normalize_decimal(row["open"], 4),
        "high": normalize_decimal(row["high"], 4),
        "low": normalize_decimal(row["low"], 4),
        "close": normalize_decimal(row["close"], 4),
        "preclose": normalize_decimal(row["preclose"], 4),
        "volume": normalize_integer(row["volume"]),
        "amount": normalize_decimal(row["amount"], 2),
        "turn": normalize_decimal(row["turn"], 6),
        "pctChg": normalize_decimal(row["pctChg"], 6),
        "tradestatus": normalize_integer(row["tradestatus"]),
        "isST": normalize_integer(row["isST"]),
        "adjustflag": normalize_integer(row["adjustflag"]),
        "adjustment": normalize_text(row["adjustment"]).lower(),
        "data_source": normalize_text(row["data_source"]),
    }


def normalize_constituents_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "code": normalize_code(row["code"]),
        "code_name": normalize_text(row["code_name"]),
        "industry_code": normalize_text(row["industry_code"]).upper(),
        "industry_name": normalize_text(row["industry_name"]),
        "classification": normalize_text(row["classification"]),
        "updateDate": normalize_date(row["updateDate"]),
    }


def clean_csv(
    input_path: Path,
    output_path: Path,
    fields: list[str],
    normalizer: Callable[[dict[str, str]], dict[str, str]],
) -> dict[str, int]:
    with input_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != fields:
            raise ValueError(
                f"{input_path} has unexpected fields: {reader.fieldnames!r}"
            )

        cleaned_rows: list[dict[str, str]] = []
        dropped_rows = Counter()
        total_rows = 0
        for row in reader:
            total_rows += 1
            if any(is_missing(row[field]) for field in fields):
                dropped_rows["missing_value"] += 1
                continue
            try:
                cleaned_rows.append(normalizer(row))
            except ValueError:
                dropped_rows["invalid_format"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    return {
        "input_rows": total_rows,
        "output_rows": len(cleaned_rows),
        "dropped_missing_value_rows": dropped_rows["missing_value"],
        "dropped_invalid_format_rows": dropped_rows["invalid_format"],
    }


def main() -> int:
    args = parse_args()
    daily_result = clean_csv(
        args.daily_input,
        args.daily_output,
        DAILY_FIELDS,
        normalize_daily_row,
    )
    constituents_result = clean_csv(
        args.constituents_input,
        args.constituents_output,
        CONSTITUENTS_FIELDS,
        normalize_constituents_row,
    )

    report = {
        "cleaned_on": date.today().isoformat(),
        "rules": {
            "missing_values": "Rows containing blank, NA, N/A, NaN, or null values are removed.",
            "date_format": "YYYY-MM-DD",
            "stock_code_format": "lowercase exchange prefix plus a period and six digits",
            "price_format": "four decimal places",
            "volume_format": "integer",
            "amount_format": "two decimal places",
            "turn_and_pctChg_format": "six decimal places",
            "status_flag_format": "integer",
        },
        "daily_data": daily_result,
        "constituents_data": constituents_result,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Daily data: {daily_result['input_rows']} -> "
        f"{daily_result['output_rows']} rows"
    )
    print(
        f"Constituents data: {constituents_result['input_rows']} -> "
        f"{constituents_result['output_rows']} rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
