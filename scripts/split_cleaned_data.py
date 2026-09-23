#!/usr/bin/env python3
"""Split cleaned daily data into chronological train/validation/test/OOS sets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT / "data/cleaned/j66_money_finance_daily_2018_2024.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/cleaned/splits"
SPLIT_NAMES = ("train", "validation", "test", "final_oos")
DEFAULT_RATIOS = {
    "train": Decimal("0.70"),
    "validation": Decimal("0.10"),
    "test": Decimal("0.10"),
    "final_oos": Decimal("0.10"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split cleaned panel data chronologically by trading date. All "
            "securities from a date stay in the same dataset."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    for split_name in SPLIT_NAMES:
        parser.add_argument(
            f"--{split_name.replace('_', '-')}-ratio",
            type=Decimal,
            default=DEFAULT_RATIOS[split_name],
            help=f"Fraction of trading dates assigned to {split_name}.",
        )
    return parser.parse_args()


def validate_ratios(args: argparse.Namespace) -> dict[str, Decimal]:
    ratios = {
        split_name: getattr(args, f"{split_name}_ratio")
        for split_name in SPLIT_NAMES
    }
    if any(not ratio.is_finite() or ratio <= 0 for ratio in ratios.values()):
        raise ValueError("All split ratios must be finite and greater than zero.")
    if sum(ratios.values()) != Decimal("1"):
        raise ValueError(
            "Split ratios must sum exactly to 1.0 "
            f"(received {sum(ratios.values())})."
        )
    return ratios


def allocate_date_counts(
    total_dates: int,
    ratios: dict[str, Decimal],
) -> dict[str, int]:
    """Allocate whole trading dates with largest-remainder rounding."""

    raw_counts = {
        name: Decimal(total_dates) * ratios[name] for name in SPLIT_NAMES
    }
    counts = {name: int(raw_counts[name]) for name in SPLIT_NAMES}
    remaining = total_dates - sum(counts.values())
    by_remainder = sorted(
        SPLIT_NAMES,
        key=lambda name: (raw_counts[name] - counts[name], -SPLIT_NAMES.index(name)),
        reverse=True,
    )
    for name in by_remainder[:remaining]:
        counts[name] += 1

    if any(counts[name] == 0 for name in SPLIT_NAMES):
        raise ValueError(
            "Not enough unique trading dates to create four non-empty splits."
        )
    return counts


def load_rows(
    input_path: Path,
) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    rows_by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    with input_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        if not fields or "date" not in fields:
            raise ValueError(f"{input_path} must contain a date column.")
        if len(fields) != len(set(fields)):
            raise ValueError(f"{input_path} contains duplicate column names.")

        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    f"Malformed CSV row at line {row_number}: field count mismatch."
                )
            raw_date = row["date"].strip()
            try:
                parsed_date = date.fromisoformat(raw_date)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid ISO date at line {row_number}: {raw_date!r}"
                ) from exc
            if parsed_date.isoformat() != raw_date:
                raise ValueError(
                    f"Date at line {row_number} is not YYYY-MM-DD: {raw_date!r}"
                )
            rows_by_date[raw_date].append(row)

    if not rows_by_date:
        raise ValueError(f"{input_path} contains no data rows.")
    return fields, rows_by_date


def relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def write_split(
    output_path: Path,
    fields: list[str],
    date_keys: list[str],
    rows_by_date: dict[str, list[dict[str, str]]],
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output_path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for trading_date in date_keys:
            for row in rows_by_date[trading_date]:
                writer.writerow(row)
                row_count += 1
    return row_count


def build_report(
    *,
    input_path: Path,
    output_dir: Path,
    ratios: dict[str, Decimal],
    split_dates: dict[str, list[str]],
    split_rows: dict[str, int],
) -> dict[str, Any]:
    return {
        "source_file": relative_or_absolute(input_path),
        "split_method": "chronological_by_unique_trading_date",
        "date_grouping": (
            "All records sharing one trading date are assigned to the same split."
        ),
        "ratios": {name: str(ratios[name]) for name in SPLIT_NAMES},
        "label_leakage_note": (
            "These files split feature rows by date. When forward-looking labels "
            "are added, apply a purge/embargo around split boundaries according "
            "to the label horizon."
        ),
        "splits": {
            name: {
                "file": relative_or_absolute(output_dir / f"{name}.csv"),
                "trading_dates": len(split_dates[name]),
                "start_date": split_dates[name][0],
                "end_date": split_dates[name][-1],
                "rows": split_rows[name],
            }
            for name in SPLIT_NAMES
        },
        "final_oos_usage": (
            "Keep final_oos untouched during feature/model selection and evaluate "
            "it once after the research choices are frozen."
        ),
    }


def main() -> int:
    args = parse_args()
    ratios = validate_ratios(args)
    fields, rows_by_date = load_rows(args.input)
    dates = sorted(rows_by_date)
    date_counts = allocate_date_counts(len(dates), ratios)

    split_dates: dict[str, list[str]] = {}
    offset = 0
    for split_name in SPLIT_NAMES:
        next_offset = offset + date_counts[split_name]
        split_dates[split_name] = dates[offset:next_offset]
        offset = next_offset
    if offset != len(dates):
        raise AssertionError("Split allocation did not cover every trading date.")

    split_rows: dict[str, int] = {}
    for split_name in SPLIT_NAMES:
        split_rows[split_name] = write_split(
            args.output_dir / f"{split_name}.csv",
            fields,
            split_dates[split_name],
            rows_by_date,
        )

    report = build_report(
        input_path=args.input,
        output_dir=args.output_dir,
        ratios=ratios,
        split_dates=split_dates,
        split_rows=split_rows,
    )
    report_path = args.output_dir / "split_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for split_name in SPLIT_NAMES:
        print(
            f"{split_name}: {split_rows[split_name]} rows, "
            f"{split_dates[split_name][0]} - {split_dates[split_name][-1]}"
        )
    print(f"Report: {relative_or_absolute(report_path)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (InvalidOperation, OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
