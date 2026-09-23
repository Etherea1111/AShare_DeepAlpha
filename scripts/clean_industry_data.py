#!/usr/bin/env python3
"""Clean the J66 daily data and build point-in-time status information.

The raw CSV files are kept unchanged. The cleaned daily file contains only
complete rows for securities that have been listed for at least the configured
number of market sessions. A separate status ledger keeps suspension,
pre-listing, limit-up, and limit-down information instead of confusing those
states with ordinary missing data.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DAILY_PATH = PROJECT_ROOT / "data/j66_money_finance_daily_2018_2024.csv"
RAW_CONSTITUENTS_PATH = PROJECT_ROOT / "data/j66_money_finance_constituents.csv"
CLEAN_DAILY_PATH = PROJECT_ROOT / "data/cleaned/j66_money_finance_daily_2018_2024.csv"
CLEAN_STATUS_PATH = PROJECT_ROOT / "data/cleaned/j66_money_finance_status_2018_2024.csv"
CLEAN_CONSTITUENTS_PATH = (
    PROJECT_ROOT / "data/cleaned/j66_money_finance_constituents.csv"
)
CLEAN_REPORT_PATH = PROJECT_ROOT / "data/cleaned/cleaning_report.json"

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
DAILY_OUTPUT_FIELDS = DAILY_FIELDS + [
    "listing_date",
    "listing_age_trading_days",
    "listing_status",
    "listing_eligible",
    "trading_status",
    "price_limit_status",
    "security_status",
]
STATUS_FIELDS = [
    "date",
    "code",
    "code_name",
    "listing_date",
    "listing_age_trading_days",
    "listing_status",
    "listing_eligible",
    "trading_status",
    "price_limit_status",
    "security_status",
]
CONSTITUENT_FIELDS = [
    "code",
    "code_name",
    "industry_code",
    "industry_name",
    "classification",
    "updateDate",
]

MISSING_VALUES = {"", "na", "n/a", "nan", "null", "nat"}
CODE_PATTERN = re.compile(r"^(sh|sz|bj)\.?(?P<number>\d{6})$", re.IGNORECASE)
PRICE_TICK = Decimal("0.01")
PRICE_TOLERANCE = Decimal("0.0001")


@dataclass(frozen=True)
class Presence:
    """The part of a raw row needed to classify incomplete observations."""

    row_number: int
    code_name: str
    tradestatus: int | None
    quality: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean the J66 daily data, classify point-in-time statuses, and "
            "remove securities with fewer than one year of trading history."
        )
    )
    parser.add_argument("--daily-input", type=Path, default=RAW_DAILY_PATH)
    parser.add_argument(
        "--constituents-input", type=Path, default=RAW_CONSTITUENTS_PATH
    )
    parser.add_argument("--daily-output", type=Path, default=CLEAN_DAILY_PATH)
    parser.add_argument("--status-output", type=Path, default=CLEAN_STATUS_PATH)
    parser.add_argument(
        "--constituents-output",
        type=Path,
        default=CLEAN_CONSTITUENTS_PATH,
    )
    parser.add_argument("--report-output", type=Path, default=CLEAN_REPORT_PATH)
    parser.add_argument(
        "--min-listing-trading-days",
        type=int,
        default=250,
        help="Minimum number of market sessions since listing, inclusive.",
    )
    return parser.parse_args()


def is_missing(value: str | None) -> bool:
    return value is None or value.strip().casefold() in MISSING_VALUES


def normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def normalize_date(value: str) -> date:
    raw_value = normalize_text(value)
    for date_format in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d"):
        try:
            return datetime.strptime(raw_value, date_format).date()
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


def normalize_integer(value: str) -> int:
    number = parse_decimal(value)
    if number != number.to_integral_value():
        raise ValueError(f"Expected integer: {value!r}")
    return int(number)


def normalize_daily_row(row: dict[str, str]) -> dict[str, str]:
    normalized_date = normalize_date(row["date"])
    normalized_code = normalize_code(row["code"])
    normalized = {
        "date": normalized_date.isoformat(),
        "code": normalized_code,
        "code_name": normalize_text(row["code_name"]),
        "industry_code": normalize_text(row["industry_code"]).upper(),
        "industry_name": normalize_text(row["industry_name"]),
        "open": normalize_decimal(row["open"], 4),
        "high": normalize_decimal(row["high"], 4),
        "low": normalize_decimal(row["low"], 4),
        "close": normalize_decimal(row["close"], 4),
        "preclose": normalize_decimal(row["preclose"], 4),
        "volume": str(normalize_integer(row["volume"])),
        "amount": normalize_decimal(row["amount"], 2),
        "turn": normalize_decimal(row["turn"], 6),
        "pctChg": normalize_decimal(row["pctChg"], 6),
        "tradestatus": str(normalize_integer(row["tradestatus"])),
        "isST": str(normalize_integer(row["isST"])),
        "adjustflag": str(normalize_integer(row["adjustflag"])),
        "adjustment": normalize_text(row["adjustment"]).lower(),
        "data_source": normalize_text(row["data_source"]),
    }
    return normalized


def normalize_constituent_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "code": normalize_code(row["code"]),
        "code_name": normalize_text(row["code_name"]),
        "industry_code": normalize_text(row["industry_code"]).upper(),
        "industry_name": normalize_text(row["industry_name"]),
        "classification": normalize_text(row["classification"]),
        "updateDate": normalize_date(row["updateDate"]).isoformat(),
    }


def read_constituents(
    input_path: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows: list[dict[str, str]] = []
    dropped = Counter()
    total_rows = 0

    with input_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != CONSTITUENT_FIELDS:
            raise ValueError(
                f"{input_path} has unexpected fields: {reader.fieldnames!r}"
            )

        seen_codes: set[str] = set()
        for row in reader:
            total_rows += 1
            if any(is_missing(row.get(field)) for field in CONSTITUENT_FIELDS):
                dropped["missing_value"] += 1
                continue
            try:
                normalized = normalize_constituent_row(row)
            except ValueError:
                dropped["invalid_format"] += 1
                continue
            if normalized["code"] in seen_codes:
                dropped["duplicate"] += 1
                continue
            seen_codes.add(normalized["code"])
            rows.append(normalized)

    result = {
        "input_rows": total_rows,
        "output_rows": len(rows),
        "dropped_missing_value_rows": dropped["missing_value"],
        "dropped_invalid_format_rows": dropped["invalid_format"],
        "dropped_duplicate_rows": dropped["duplicate"],
    }
    return rows, result


def read_daily(
    input_path: Path,
) -> tuple[
    dict[tuple[date, str], dict[str, str]],
    dict[tuple[date, str], Presence],
    dict[str, date],
    set[date],
    dict[str, str],
    dict[str, Any],
]:
    normalized_rows: dict[tuple[date, str], dict[str, str]] = {}
    presence: dict[tuple[date, str], Presence] = {}
    listing_dates: dict[str, date] = {}
    trading_dates: set[date] = set()
    code_names: dict[str, str] = {}
    dropped = Counter()
    total_rows = 0
    identity_valid_rows = 0

    with input_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != DAILY_FIELDS:
            raise ValueError(
                f"{input_path} has unexpected fields: {reader.fieldnames!r}"
            )

        for row_number, row in enumerate(reader, start=2):
            total_rows += 1
            try:
                row_date = normalize_date(row["date"])
                code = normalize_code(row["code"])
            except ValueError:
                dropped["invalid_identity"] += 1
                continue

            identity_valid_rows += 1
            trading_dates.add(row_date)
            listing_dates[code] = min(row_date, listing_dates.get(code, row_date))
            raw_code_name = row.get("code_name")
            if not is_missing(raw_code_name):
                code_names[code] = normalize_text(raw_code_name or "")

            key = (row_date, code)
            if key in presence:
                dropped["duplicate"] += 1
                continue

            raw_tradestatus: int | None = None
            if not is_missing(row.get("tradestatus")):
                try:
                    raw_tradestatus = normalize_integer(row["tradestatus"])
                except ValueError:
                    raw_tradestatus = None

            if any(is_missing(row.get(field)) for field in DAILY_FIELDS):
                presence[key] = Presence(
                    row_number=row_number,
                    code_name=code_names.get(code, ""),
                    tradestatus=raw_tradestatus,
                    quality="missing",
                )
                dropped["missing_value"] += 1
                continue

            try:
                normalized = normalize_daily_row(row)
            except ValueError:
                presence[key] = Presence(
                    row_number=row_number,
                    code_name=code_names.get(code, ""),
                    tradestatus=raw_tradestatus,
                    quality="invalid_format",
                )
                dropped["invalid_format"] += 1
                continue

            normalized_date = normalize_date(normalized["date"])
            normalized_code = normalized["code"]
            presence[key] = Presence(
                row_number=row_number,
                code_name=normalized["code_name"],
                tradestatus=int(normalized["tradestatus"]),
                quality="complete",
            )
            normalized_rows[(normalized_date, normalized_code)] = normalized
            code_names[normalized_code] = normalized["code_name"]

    result = {
        "input_rows": total_rows,
        "identity_valid_rows": identity_valid_rows,
        "complete_valid_rows": len(normalized_rows),
        "dropped_missing_value_rows": dropped["missing_value"],
        "dropped_invalid_identity_rows": dropped["invalid_identity"],
        "dropped_invalid_format_rows": dropped["invalid_format"],
        "dropped_duplicate_rows": dropped["duplicate"],
    }
    return (
        normalized_rows,
        presence,
        listing_dates,
        trading_dates,
        code_names,
        result,
    )


def parse_numeric_fields(row: dict[str, str]) -> dict[str, Decimal]:
    return {
        field: parse_decimal(row[field])
        for field in (
            "high",
            "low",
            "close",
            "preclose",
        )
    }


def limit_rate(code: str, is_st: int) -> Decimal:
    if is_st == 1:
        return Decimal("0.05")
    if code.startswith(("sh.688", "sz.300")):
        return Decimal("0.20")
    if code.startswith("bj."):
        return Decimal("0.30")
    return Decimal("0.10")


def price_limit_status(
    row: dict[str, str],
    listing_age_trading_days: int,
) -> str:
    """Classify whether the daily high/low touched a normal price limit.

    The first listing session is excluded because IPO pricing rules do not
    follow the ordinary daily limit calculation.
    """

    if int(row["tradestatus"]) != 1 or listing_age_trading_days <= 1:
        return "not_applicable" if int(row["tradestatus"]) != 1 else "none"

    numeric = parse_numeric_fields(row)
    rate = limit_rate(row["code"], int(row["isST"]))
    upper_limit = (numeric["preclose"] * (Decimal(1) + rate)).quantize(
        PRICE_TICK,
        rounding=ROUND_HALF_UP,
    )
    lower_limit = (numeric["preclose"] * (Decimal(1) - rate)).quantize(
        PRICE_TICK,
        rounding=ROUND_HALF_UP,
    )
    touched_upper = numeric["high"] + PRICE_TOLERANCE >= upper_limit
    touched_lower = numeric["low"] - PRICE_TOLERANCE <= lower_limit

    if touched_upper and touched_lower:
        return "limit_up_and_down"
    if touched_upper:
        return "limit_up"
    if touched_lower:
        return "limit_down"
    return "none"


def listing_age(
    row_date: date,
    listing_date: date,
    date_position: dict[date, int],
) -> int:
    if row_date < listing_date:
        return 0
    return date_position[row_date] - date_position[listing_date] + 1


def listing_status(age: int, min_listing_trading_days: int) -> str:
    if age == 0:
        return "not_listed"
    if age < min_listing_trading_days:
        return "listed_less_than_one_year"
    return "listed_at_least_one_year"


def trading_status_from_presence(
    row_presence: Presence | None,
    normalized_row: dict[str, str] | None,
    is_pre_listing: bool,
) -> str:
    if is_pre_listing:
        return "not_listed"
    if row_presence is None:
        return "no_record"
    if row_presence.quality == "missing":
        if row_presence.tradestatus == 0:
            return "suspended"
        return "missing_data"
    if row_presence.quality == "invalid_format":
        return "invalid_data"
    if normalized_row is None:
        return "no_record"
    tradestatus = int(normalized_row["tradestatus"])
    if tradestatus == 0:
        return "suspended"
    if tradestatus == 1:
        return "trading"
    return "unknown"


def security_status(
    *,
    age: int,
    listing_eligible: int,
    trading_status: str,
    price_status: str,
) -> str:
    if age == 0:
        return "not_listed"
    if trading_status in {"suspended", "missing_data", "invalid_data", "no_record"}:
        return trading_status
    if trading_status == "unknown":
        return "unknown"
    if not listing_eligible:
        return "listed_less_than_one_year"
    if price_status != "none":
        return price_status
    return "normal"


def build_status_ledger(
    *,
    normalized_rows: dict[tuple[date, str], dict[str, str]],
    presence: dict[tuple[date, str], Presence],
    listing_dates: dict[str, date],
    trading_dates: list[date],
    code_names: dict[str, str],
    min_listing_trading_days: int,
) -> tuple[list[dict[str, str]], Counter, Counter, Counter]:
    date_position = {trading_date: index for index, trading_date in enumerate(trading_dates)}
    status_rows: list[dict[str, str]] = []
    security_counts = Counter()
    trading_counts = Counter()
    price_limit_counts = Counter()

    for trading_date in trading_dates:
        for code in sorted(listing_dates):
            listing_date = listing_dates[code]
            age = listing_age(trading_date, listing_date, date_position)
            eligible = int(age >= min_listing_trading_days)
            key = (trading_date, code)
            normalized_row = normalized_rows.get(key)
            row_presence = presence.get(key)
            pre_listing = age == 0
            trading_status = trading_status_from_presence(
                row_presence,
                normalized_row,
                pre_listing,
            )
            if normalized_row is None or trading_status != "trading":
                price_status = "not_applicable"
            else:
                price_status = price_limit_status(normalized_row, age)
            row_security_status = security_status(
                age=age,
                listing_eligible=eligible,
                trading_status=trading_status,
                price_status=price_status,
            )
            status_row = {
                "date": trading_date.isoformat(),
                "code": code,
                "code_name": code_names.get(code, ""),
                "listing_date": listing_date.isoformat(),
                "listing_age_trading_days": str(age),
                "listing_status": listing_status(age, min_listing_trading_days),
                "listing_eligible": str(eligible),
                "trading_status": trading_status,
                "price_limit_status": price_status,
                "security_status": row_security_status,
            }
            status_rows.append(status_row)
            security_counts[row_security_status] += 1
            trading_counts[trading_status] += 1
            price_limit_counts[price_status] += 1

    return status_rows, security_counts, trading_counts, price_limit_counts


def build_clean_daily_rows(
    *,
    normalized_rows: dict[tuple[date, str], dict[str, str]],
    listing_dates: dict[str, date],
    trading_dates: list[date],
    min_listing_trading_days: int,
) -> tuple[list[dict[str, str]], Counter, Counter]:
    date_position = {trading_date: index for index, trading_date in enumerate(trading_dates)}
    clean_rows: list[dict[str, str]] = []
    dropped = Counter()
    security_counts = Counter()

    for key in sorted(normalized_rows):
        row_date, code = key
        row = normalized_rows[key]
        listing_date = listing_dates[code]
        age = listing_age(row_date, listing_date, date_position)
        if age < min_listing_trading_days:
            dropped["listed_less_than_one_year"] += 1
            continue

        trading_status = (
            "suspended"
            if int(row["tradestatus"]) == 0
            else "trading"
            if int(row["tradestatus"]) == 1
            else "unknown"
        )
        price_status = price_limit_status(row, age)
        row_security_status = security_status(
            age=age,
            listing_eligible=1,
            trading_status=trading_status,
            price_status=price_status,
        )
        output_row = dict(row)
        output_row.update(
            {
                "listing_date": listing_date.isoformat(),
                "listing_age_trading_days": str(age),
                "listing_status": "listed_at_least_one_year",
                "listing_eligible": "1",
                "trading_status": trading_status,
                "price_limit_status": price_status,
                "security_status": row_security_status,
            }
        )
        clean_rows.append(output_row)
        security_counts[row_security_status] += 1

    return clean_rows, dropped, security_counts


def write_csv(
    output_path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def counter_as_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def main() -> int:
    args = parse_args()
    if args.min_listing_trading_days < 1:
        raise ValueError("min-listing-trading-days must be at least 1")

    constituents, constituent_result = read_constituents(
        args.constituents_input
    )
    (
        normalized_rows,
        presence,
        listing_dates,
        trading_date_set,
        daily_code_names,
        daily_result,
    ) = read_daily(args.daily_input)

    constituent_code_names = {
        row["code"]: row["code_name"] for row in constituents
    }
    code_names = {**constituent_code_names, **daily_code_names}
    trading_dates = sorted(trading_date_set)
    if not trading_dates:
        raise ValueError("No valid trading dates were found in the daily input.")

    status_rows, status_security_counts, status_trading_counts, status_price_counts = (
        build_status_ledger(
            normalized_rows=normalized_rows,
            presence=presence,
            listing_dates=listing_dates,
            trading_dates=trading_dates,
            code_names=code_names,
            min_listing_trading_days=args.min_listing_trading_days,
        )
    )
    clean_rows, listing_filter_drops, clean_security_counts = build_clean_daily_rows(
        normalized_rows=normalized_rows,
        listing_dates=listing_dates,
        trading_dates=trading_dates,
        min_listing_trading_days=args.min_listing_trading_days,
    )

    write_csv(args.daily_output, DAILY_OUTPUT_FIELDS, clean_rows)
    write_csv(args.status_output, STATUS_FIELDS, status_rows)
    write_csv(
        args.constituents_output,
        CONSTITUENT_FIELDS,
        constituents,
    )

    report = {
        "cleaned_on": date.today().isoformat(),
        "source": {
            "daily_input": str(args.daily_input),
            "constituents_input": str(args.constituents_input),
            "trading_date_start": trading_dates[0].isoformat(),
            "trading_date_end": trading_dates[-1].isoformat(),
            "trading_date_count": len(trading_dates),
            "security_count": len(listing_dates),
        },
        "parameters": {
            "min_listing_trading_days": args.min_listing_trading_days,
            "price_limit_detection": (
                "Daily high/low reached the exchange price-limit price; "
                "the first listing session is excluded."
            ),
        },
        "rules": {
            "missing_values": (
                "Rows with a missing value in any expected source column are "
                "removed from the cleaned daily table."
            ),
            "suspension": (
                "tradestatus=0 is classified as suspended. Suspended rows are "
                "kept in the status ledger and are removed from the clean table "
                "when they also contain missing fields."
            ),
            "not_listed": (
                "A date before the first observed date for a code is classified "
                "as not_listed in the full date-code status ledger."
            ),
            "listing_age": (
                "Listing age is counted dynamically on the global trading "
                "calendar, inclusive of the listing session."
            ),
            "data_types": {
                "date": "YYYY-MM-DD",
                "code": "exchange prefix plus period and six digits",
                "price": "four decimal places",
                "volume": "integer",
                "amount": "two decimal places",
                "turn_and_pctChg": "six decimal places",
                "flags_and_listing_age": "integer",
            },
        },
        "daily_data": {
            **daily_result,
            "dropped_listed_less_than_one_year_rows": int(
                listing_filter_drops["listed_less_than_one_year"]
            ),
            "output_rows": len(clean_rows),
            "output_security_status": counter_as_dict(clean_security_counts),
        },
        "status_ledger": {
            "output_rows": len(status_rows),
            "security_status": counter_as_dict(status_security_counts),
            "trading_status": counter_as_dict(status_trading_counts),
            "price_limit_status": counter_as_dict(status_price_counts),
        },
        "constituents_data": constituent_result,
        "listing_dates": {
            code: listing_dates[code].isoformat()
            for code in sorted(listing_dates)
        },
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Daily data: {daily_result['input_rows']} input rows -> "
        f"{len(clean_rows)} cleaned rows"
    )
    print(
        f"Status ledger: {len(status_rows)} rows "
        f"({len(trading_dates)} trading dates x {len(listing_dates)} securities)"
    )
    print(
        f"Removed missing={daily_result['dropped_missing_value_rows']}, "
        f"invalid={daily_result['dropped_invalid_format_rows']}, "
        f"under_one_year={listing_filter_drops['listed_less_than_one_year']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
