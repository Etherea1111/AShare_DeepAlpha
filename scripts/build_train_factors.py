#!/usr/bin/env python3
"""Build exploratory price-volume factors and split-safe forward labels."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_INPUT = PROJECT_ROOT / "data/cleaned/splits/train.csv"
DEFAULT_HISTORY_INPUT = PROJECT_ROOT / "data/j66_money_finance_daily_2018_2024.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/train_factor_labels.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data/processed/train_factor_label_report.json"
LABEL_HORIZONS = (5, 20)
FEATURE_FIELDS = [
    "ret_1",
    "mom_5",
    "mom_20",
    "mom_60",
    "volatility_5",
    "volatility_20",
    "volatility_60",
    "sma_ratio_5_20",
    "sma_ratio_20_60",
    "volume_ratio_5_20",
    "amount_ratio_5_20",
    "turnover_mean_20",
    "intraday_range_pct",
    "close_location",
    "open_gap_pct",
    "amihud_20",
    "rsi_14",
]
RAW_FIELDS = [
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "turn",
    "tradestatus",
]


@dataclass(frozen=True)
class Bar:
    trading_date: date
    open: float
    high: float
    low: float
    close: float
    preclose: float
    volume: float
    amount: float
    turnover: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create past-only price-volume factors and 5/20-session forward "
            "close returns for the training split."
        )
    )
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT)
    parser.add_argument(
        "--history-input",
        type=Path,
        default=DEFAULT_HISTORY_INPUT,
        help="Raw history used only for factor warm-up, never after train end.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def parse_day(value: str, context: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid date {value!r} in {context}") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"Date must use YYYY-MM-DD in {context}: {value!r}")
    return parsed


def parse_number(value: str, field: str, context: str) -> float:
    try:
        number = float(value.replace(",", "").strip())
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} in {context}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {field} in {context}: {value!r}")
    return number


def read_training(
    input_path: Path,
) -> tuple[list[str], list[dict[str, str]], list[date], dict[date, dict[str, dict[str, str]]]]:
    rows: list[dict[str, str]] = []
    rows_by_date: dict[date, dict[str, dict[str, str]]] = defaultdict(dict)
    with input_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        if not fields or not {"date", "code", "close"}.issubset(fields):
            raise ValueError(f"{input_path} must contain date, code, and close.")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed training row at line {line_number}.")
            trading_date = parse_day(row["date"], f"{input_path}:{line_number}")
            code = row["code"]
            if not code:
                raise ValueError(f"Empty code at line {line_number}.")
            if code in rows_by_date[trading_date]:
                raise ValueError(
                    f"Duplicate date/code pair in training input: "
                    f"{trading_date} {code}"
                )
            if "tradestatus" in row and row["tradestatus"] != "1":
                raise ValueError(
                    f"Training row is not tradable at line {line_number}: {code}"
                )
            parse_number(row["close"], "close", f"{input_path}:{line_number}")
            rows.append(row)
            rows_by_date[trading_date][code] = row
    if not rows:
        raise ValueError(f"No training rows found in {input_path}.")
    dates = sorted(rows_by_date)
    return list(fields), rows, dates, rows_by_date


def read_history(
    input_path: Path,
    train_end: date,
) -> tuple[dict[str, list[Bar]], dict[str, Any]]:
    histories: dict[str, list[Bar]] = defaultdict(list)
    counts = Counter()
    with input_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not set(RAW_FIELDS).issubset(reader.fieldnames):
            raise ValueError(
                f"{input_path} is missing required fields: "
                f"{sorted(set(RAW_FIELDS) - set(reader.fieldnames or []))!r}"
            )
        seen: set[tuple[date, str]] = set()
        for line_number, row in enumerate(reader, start=2):
            counts["input_rows"] += 1
            trading_date = parse_day(row["date"], f"{input_path}:{line_number}")
            if trading_date > train_end:
                counts["after_training_end_excluded"] += 1
                continue
            code = row["code"]
            key = (trading_date, code)
            if key in seen:
                raise ValueError(
                    f"Duplicate raw date/code pair: {trading_date} {code}"
                )
            seen.add(key)
            if row["tradestatus"] != "1":
                counts["non_trading_rows_excluded"] += 1
                continue
            try:
                bar = Bar(
                    trading_date=trading_date,
                    open=parse_number(row["open"], "open", f"line {line_number}"),
                    high=parse_number(row["high"], "high", f"line {line_number}"),
                    low=parse_number(row["low"], "low", f"line {line_number}"),
                    close=parse_number(row["close"], "close", f"line {line_number}"),
                    preclose=parse_number(
                        row["preclose"], "preclose", f"line {line_number}"
                    ),
                    volume=parse_number(
                        row["volume"], "volume", f"line {line_number}"
                    ),
                    amount=parse_number(
                        row["amount"], "amount", f"line {line_number}"
                    ),
                    turnover=parse_number(row["turn"], "turn", f"line {line_number}"),
                )
            except ValueError:
                counts["incomplete_or_invalid_rows_excluded"] += 1
                continue
            histories[code].append(bar)
            counts["usable_rows"] += 1

    for code, bars in histories.items():
        bars.sort(key=lambda bar: bar.trading_date)
        dates = [bar.trading_date for bar in bars]
        if len(dates) != len(set(dates)):
            raise ValueError(f"Duplicate usable history date for {code}.")
    if not histories:
        raise ValueError(f"No usable historical bars found in {input_path}.")

    result = {
        "input_rows": counts["input_rows"],
        "usable_trading_rows_at_or_before_train_end": counts["usable_rows"],
        "non_trading_rows_excluded": counts["non_trading_rows_excluded"],
        "incomplete_or_invalid_rows_excluded": counts[
            "incomplete_or_invalid_rows_excluded"
        ],
        "rows_after_training_end_excluded": counts[
            "after_training_end_excluded"
        ],
        "history_start_date": min(
            bar.trading_date for bars in histories.values() for bar in bars
        ).isoformat(),
        "history_end_date": max(
            bar.trading_date for bars in histories.values() for bar in bars
        ).isoformat(),
    }
    return histories, result


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def rolling_mean(values: list[float], end: int, window: int) -> float | None:
    start = end - window + 1
    if start < 0:
        return None
    return mean(values[start : end + 1])


def ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator - 1.0


def rsi(returns: list[float], end: int, window: int = 14) -> float | None:
    start = end - window + 1
    if start < 1:
        return None
    sample = returns[start : end + 1]
    gains = [max(value, 0.0) for value in sample]
    losses = [max(-value, 0.0) for value in sample]
    average_gain = mean(gains)
    average_loss = mean(losses)
    if average_gain == 0 and average_loss == 0:
        return 50.0
    if average_loss == 0:
        return 100.0
    relative_strength = average_gain / average_loss
    return 100.0 - 100.0 / (1.0 + relative_strength)


def calculate_features(bars: list[Bar], index: int) -> dict[str, float] | None:
    if index < 1:
        return None
    visible_bars = bars[: index + 1]
    closes = [bar.close for bar in visible_bars]
    returns: list[float | None] = [None]
    for previous, current in zip(closes, closes[1:]):
        returns.append(current / previous - 1.0 if previous else None)

    current = visible_bars[index]
    previous_close = visible_bars[index - 1].close
    features: dict[str, float | None] = {
        "ret_1": returns[index],
        "mom_5": ratio(current.close, closes[index - 5]) if index >= 5 else None,
        "mom_20": ratio(current.close, closes[index - 20]) if index >= 20 else None,
        "mom_60": ratio(current.close, closes[index - 60]) if index >= 60 else None,
        "sma_ratio_5_20": None,
        "sma_ratio_20_60": None,
        "volume_ratio_5_20": None,
        "amount_ratio_5_20": None,
        "turnover_mean_20": rolling_mean(
            [bar.turnover for bar in visible_bars], index, 20
        ),
        "intraday_range_pct": (
            (current.high - current.low) / current.close
            if current.close != 0
            else None
        ),
        "close_location": (
            (2.0 * current.close - current.high - current.low)
            / (current.high - current.low)
            if current.high != current.low
            else 0.0
        ),
        "open_gap_pct": (
            current.open / current.preclose - 1.0
            if current.preclose != 0
            else None
        ),
        "rsi_14": None,
        "amihud_20": None,
        "volatility_5": None,
        "volatility_20": None,
        "volatility_60": None,
    }

    for window in (5, 20, 60):
        start = index - window + 1
        window_returns = returns[start : index + 1] if start >= 0 else []
        if len(window_returns) == window and all(
            value is not None for value in window_returns
        ):
            features[f"volatility_{window}"] = statistics.pstdev(
                value for value in window_returns if value is not None
            )

    close_mean_5 = rolling_mean(closes, index, 5)
    close_mean_20 = rolling_mean(closes, index, 20)
    close_mean_60 = rolling_mean(closes, index, 60)
    if close_mean_5 is not None and close_mean_20:
        features["sma_ratio_5_20"] = close_mean_5 / close_mean_20 - 1.0
    if close_mean_20 is not None and close_mean_60:
        features["sma_ratio_20_60"] = close_mean_20 / close_mean_60 - 1.0

    volume_5 = rolling_mean([bar.volume for bar in visible_bars], index, 5)
    volume_20 = rolling_mean([bar.volume for bar in visible_bars], index, 20)
    amount_5 = rolling_mean([bar.amount for bar in visible_bars], index, 5)
    amount_20 = rolling_mean([bar.amount for bar in visible_bars], index, 20)
    if volume_5 is not None and volume_20:
        features["volume_ratio_5_20"] = volume_5 / volume_20 - 1.0
    if amount_5 is not None and amount_20:
        features["amount_ratio_5_20"] = amount_5 / amount_20 - 1.0

    features["rsi_14"] = rsi(
        [value if value is not None else 0.0 for value in returns],
        index,
    )

    amihud_start = index - 19
    if amihud_start >= 1:
        paired = [
            (abs(returns[position] or 0.0), bars[position].amount)
            for position in range(amihud_start, index + 1)
        ]
        if len(paired) == 20 and all(amount > 0 for _, amount in paired):
            features["amihud_20"] = (
                mean([absolute_return / amount for absolute_return, amount in paired])
                * 1e8
            )

    if previous_close == 0:
        features["ret_1"] = None
    complete = {
        name: value
        for name, value in features.items()
        if value is not None and math.isfinite(value)
    }
    if len(complete) != len(FEATURE_FIELDS):
        return None
    return {name: complete[name] for name in FEATURE_FIELDS}


def format_number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"Cannot serialize non-finite value: {value!r}")
    return format(value, ".12g")


def build_dataset(
    *,
    source_fields: list[str],
    training_rows: list[dict[str, str]],
    trading_dates: list[date],
    training_rows_by_date: dict[date, dict[str, dict[str, str]]],
    histories: dict[str, list[Bar]],
) -> tuple[list[str], list[dict[str, str]], Counter, dict[str, Any]]:
    date_positions = {trading_date: i for i, trading_date in enumerate(trading_dates)}
    history_dates = {
        code: [bar.trading_date for bar in bars] for code, bars in histories.items()
    }
    output_fields = source_fields + FEATURE_FIELDS
    for horizon in LABEL_HORIZONS:
        output_fields.extend(
            [f"fwd_return_{horizon}d", f"label_{horizon}d_end_date"]
        )

    output_rows: list[dict[str, str]] = []
    drops = Counter()
    feature_source_dates: list[date] = []
    feature_as_of_dates: list[date] = []
    feature_lookback_dates: list[date] = []
    feature_as_of_pairs: list[tuple[date, date]] = []
    label_end_dates: list[date] = []

    for row in training_rows:
        as_of = parse_day(row["date"], "training row")
        code = row["code"]
        bars = histories.get(code, [])
        dates = history_dates.get(code, [])
        index = bisect.bisect_right(dates, as_of) - 1
        if index < 0 or dates[index] != as_of:
            drops["feature_history_missing"] += 1
            continue

        # The factor function only reads bars[0:index + 1], never later bars.
        if dates[index] > as_of:
            raise AssertionError("Feature history extends beyond its as-of date.")
        features = calculate_features(bars, index)
        if features is None:
            drops["feature_warmup"] += 1
            continue
        feature_source_dates.append(dates[index])
        feature_as_of_dates.append(as_of)
        feature_lookback_dates.append(dates[max(0, index - 60)])
        feature_as_of_pairs.append((dates[index], as_of))

        position = date_positions[as_of]
        label_values: dict[int, tuple[float, date]] = {}
        label_failure: str | None = None
        source_close = parse_number(row["close"], "close", f"{as_of} {code}")
        for horizon in LABEL_HORIZONS:
            target_position = position + horizon
            if target_position >= len(trading_dates):
                label_failure = "label_horizon_outside_training"
                break
            target_date = trading_dates[target_position]
            target_row = training_rows_by_date.get(target_date, {}).get(code)
            if target_row is None:
                label_failure = "label_target_missing"
                break
            target_close = parse_number(
                target_row["close"], "close", f"{target_date} {code}"
            )
            if source_close == 0:
                label_failure = "invalid_label_base_price"
                break
            label_values[horizon] = (target_close / source_close - 1.0, target_date)

        if label_failure:
            drops[label_failure] += 1
            continue

        output_row = dict(row)
        output_row.update(
            {name: format_number(features[name]) for name in FEATURE_FIELDS}
        )
        for horizon in LABEL_HORIZONS:
            label, target_date = label_values[horizon]
            if target_date not in date_positions:
                raise AssertionError("A label endpoint is outside the training split.")
            if target_date <= as_of:
                raise AssertionError("A forward label endpoint is not in the future.")
            output_row[f"fwd_return_{horizon}d"] = format_number(label)
            output_row[f"label_{horizon}d_end_date"] = target_date.isoformat()
            label_end_dates.append(target_date)
        output_rows.append(output_row)

    audit = {
        "feature_count": len(FEATURE_FIELDS),
        "feature_fields": FEATURE_FIELDS,
        "label_horizons_trading_days": list(LABEL_HORIZONS),
        "feature_lookback_start_date": (
            min(feature_lookback_dates).isoformat()
            if feature_lookback_dates
            else None
        ),
        "feature_row_start_date": (
            min(feature_as_of_dates).isoformat() if feature_as_of_dates else None
        ),
        "factor_as_of_end_date": (
            max(feature_as_of_dates).isoformat() if feature_as_of_dates else None
        ),
        "max_label_end_date": (
            max(label_end_dates).isoformat() if label_end_dates else None
        ),
        "feature_history_is_past_only": bool(feature_as_of_pairs)
        and all(source_date <= as_of for source_date, as_of in feature_as_of_pairs),
        "labels_end_within_training_dates": all(
            endpoint in date_positions for endpoint in label_end_dates
        ),
        "label_endpoints_strictly_after_signal_date": all(
            parse_day(row[f"label_{horizon}d_end_date"], "output row")
            > parse_day(row["date"], "output row")
            for row in output_rows
            for horizon in LABEL_HORIZONS
        ),
        "label_horizons_match_training_calendar_offsets": all(
            date_positions[
                parse_day(row[f"label_{horizon}d_end_date"], "output row")
            ]
            - date_positions[parse_day(row["date"], "output row")]
            == horizon
            for row in output_rows
            for horizon in LABEL_HORIZONS
        ),
        "label_lookup_uses_same_code_and_training_rows_only": True,
        "cross_sectional_normalization_fitted": False,
        "post_training_rows_used_for_features_or_labels": False,
    }
    if not audit["feature_history_is_past_only"]:
        raise AssertionError("Feature leakage audit failed.")
    if not audit["labels_end_within_training_dates"]:
        raise AssertionError("Label boundary audit failed.")
    if not audit["label_endpoints_strictly_after_signal_date"]:
        raise AssertionError("Label direction audit failed.")
    if not audit["label_horizons_match_training_calendar_offsets"]:
        raise AssertionError("Label horizon audit failed.")
    return output_fields, output_rows, drops, audit


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


def factor_definitions() -> dict[str, str]:
    return {
        "ret_1": "One previous valid stock observation close-to-close return.",
        "mom_5/mom_20/mom_60": (
            "Close-to-close return over 5/20/60 previous valid stock observations."
        ),
        "volatility_5/20/60": (
            "Population standard deviation of the latest 5/20/60 close returns."
        ),
        "sma_ratio_5_20/sma_ratio_20_60": (
            "Ratio of trailing close moving averages minus one."
        ),
        "volume_ratio_5_20": (
            "Trailing 5-observation mean volume divided by 20-observation mean, minus one."
        ),
        "amount_ratio_5_20": (
            "Trailing 5-observation mean amount divided by 20-observation mean, minus one."
        ),
        "turnover_mean_20": "Mean turnover across the latest 20 valid observations.",
        "intraday_range_pct": "(High - low) / close for the current observation.",
        "close_location": (
            "(2 * close - high - low) / (high - low), with zero for zero-range bars."
        ),
        "open_gap_pct": "Open / preclose - one for the current observation.",
        "amihud_20": (
            "Mean absolute close return / amount over 20 observations, scaled by 1e8."
        ),
        "rsi_14": "Simple-average RSI over the latest 14 close-return observations.",
        "fwd_return_5d/fwd_return_20d": (
            "Same-security close-to-close return to exactly 5/20 later training "
            "market dates; rows without an in-training endpoint are excluded."
        ),
    }


def main() -> int:
    args = parse_args()
    source_fields, training_rows, trading_dates, rows_by_date = read_training(
        args.train_input
    )
    histories, history_report = read_history(args.history_input, trading_dates[-1])
    output_fields, output_rows, drops, audit = build_dataset(
        source_fields=source_fields,
        training_rows=training_rows,
        trading_dates=trading_dates,
        training_rows_by_date=rows_by_date,
        histories=histories,
    )
    write_csv(args.output, output_fields, output_rows)

    report = {
        "created_on": date.today().isoformat(),
        "input_training_file": str(args.train_input),
        "history_context_file": str(args.history_input),
        "output_file": str(args.output),
        "training_period": {
            "start_date": trading_dates[0].isoformat(),
            "end_date": trading_dates[-1].isoformat(),
            "trading_dates": len(trading_dates),
            "input_rows": len(training_rows),
        },
        "history_context": history_report,
        "factor_hypotheses": {
            "momentum": "Recent trend may persist over short horizons; test sign and stability.",
            "volatility": "High realized volatility may proxy for risk and unstable returns.",
            "volume_and_amount": "Relative activity may represent attention, liquidity, or reversal.",
            "range_and_gap": "Intraday range and opening gap may capture price pressure.",
            "amihud": "Higher absolute price impact per traded amount may indicate illiquidity.",
            "rsi": "Recent gain/loss imbalance may capture short-horizon momentum or reversal.",
            "status": "Exploratory hypotheses only; predictive value has not been evaluated.",
        },
        "factor_definitions": factor_definitions(),
        "label_definition": {
            "basis": "Unadjusted close-to-close simple return.",
            "horizons": "5 and 20 subsequent dates in the training split trading calendar.",
            "entry_execution": (
                "Research label only; it is not an executable return model. "
                "Backtests must model signal availability and next-session fills."
            ),
        },
        "data_limitations": [
            "The raw source uses adjustflag=3 (unadjusted); corporate actions can distort both factors and labels.",
            "Factor windows count valid per-security observations and can span suspension or missing-data gaps.",
            "Training inputs have already filtered securities by the project listing-age rule.",
            "No cross-sectional scaling or imputation is applied; fit any learned transform on training data only.",
        ],
        "row_counts": {
            "input_training_rows": len(training_rows),
            "output_rows_with_all_factors_and_labels": len(output_rows),
            "dropped_rows_by_reason": {
                key: int(drops[key]) for key in sorted(drops)
            },
        },
        "leakage_audit": audit,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Training rows: {len(training_rows)} -> {len(output_rows)} "
        f"factor/label rows"
    )
    print(
        "Dropped rows: "
        + ", ".join(f"{key}={drops[key]}" for key in sorted(drops))
    )
    print(f"Leakage audit: {audit}")
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
