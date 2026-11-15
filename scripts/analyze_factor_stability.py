#!/usr/bin/env python3
"""Measure cross-sectional factor RankIC stability in train and validation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from build_train_factors import FEATURE_FIELDS, write_csv
from fit_model_baselines import average_ranks, correlation, load_labeled_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_FACTORS = PROJECT_ROOT / "data/processed/train_factor_labels.csv"
DEFAULT_VALIDATION_FACTORS = (
    PROJECT_ROOT / "data/processed/validation_factor_labels.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/analysis"
DEFAULT_HORIZONS = (5, 20)
DEFAULT_ANNUALIZATION_DAYS = 252
MIN_CROSS_SECTION_SIZE = 3

SUMMARY_FIELDS = [
    "split",
    "horizon",
    "factor",
    "status",
    "start_date",
    "end_date",
    "trading_days",
    "mean_rank_ic",
    "median_rank_ic",
    "mean_abs_rank_ic",
    "std_rank_ic",
    "ic_ir",
    "annualized_ic_ir",
    "positive_ic_ratio",
    "newey_west_lag",
    "newey_west_t_stat",
    "newey_west_p_value",
]
YEARLY_FIELDS = [
    "split",
    "horizon",
    "factor",
    "year",
    "start_date",
    "end_date",
    "trading_days",
    "mean_rank_ic",
    "median_rank_ic",
    "mean_abs_rank_ic",
    "std_rank_ic",
    "ic_ir",
    "annualized_ic_ir",
    "positive_ic_ratio",
    "newey_west_lag",
    "newey_west_t_stat",
    "newey_west_p_value",
]
CORRELATION_FIELDS = [
    "split",
    "horizon",
    "factor_a",
    "factor_b",
    "trading_days",
    "mean_daily_spearman",
    "mean_abs_daily_spearman",
    "positive_day_ratio",
]


@dataclass(frozen=True)
class DailyRanking:
    target: list[float]
    factors: dict[str, list[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute daily cross-sectional factor RankIC, Newey-West "
            "significance, calendar-year stability, and factor-pair "
            "cross-sectional correlation for train and validation files."
        )
    )
    parser.add_argument(
        "--train-factors",
        type=Path,
        default=DEFAULT_TRAIN_FACTORS,
    )
    parser.add_argument(
        "--validation-factors",
        type=Path,
        default=DEFAULT_VALIDATION_FACTORS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
    )
    parser.add_argument(
        "--annualization-days",
        type=int,
        default=DEFAULT_ANNUALIZATION_DAYS,
    )
    parser.add_argument(
        "--newey-west-lag",
        type=int,
        default=None,
        help=(
            "Override the Newey-West HAC lag. By default each horizon uses "
            "horizon - 1 trading days for overlapping forward labels."
        ),
    )
    return parser.parse_args()


def rank_cross_sections(
    rows: list[dict[str, Any]],
    factors: list[str],
    min_cross_section_size: int = MIN_CROSS_SECTION_SIZE,
) -> dict[str, DailyRanking]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["date"]].append(row)

    rankings: dict[str, DailyRanking] = {}
    for trading_date in sorted(grouped):
        day_rows = grouped[trading_date]
        if len(day_rows) < min_cross_section_size:
            continue
        target = average_ranks([float(row["target"]) for row in day_rows])
        factor_ranks = {
            factor: average_ranks([float(row[factor]) for row in day_rows])
            for factor in factors
        }
        rankings[trading_date] = DailyRanking(
            target=target,
            factors=factor_ranks,
        )
    return rankings


def compute_daily_rank_ic(
    rankings: dict[str, DailyRanking],
    factors: list[str],
) -> dict[str, list[tuple[str, float]]]:
    observations: dict[str, list[tuple[str, float]]] = {
        factor: [] for factor in factors
    }
    for trading_date in sorted(rankings):
        ranking = rankings[trading_date]
        for factor in factors:
            rank_ic = correlation(
                ranking.factors[factor],
                ranking.target,
            )
            if rank_ic is not None:
                observations[factor].append((trading_date, rank_ic))
    return observations


def newey_west_t_stat(
    values: list[float],
    lag: int,
) -> tuple[float | None, int]:
    if len(values) < 2:
        return None, 0
    if lag < 0:
        raise ValueError("Newey-West lag must not be negative.")

    effective_lag = min(lag, len(values) - 1)
    sample_size = float(len(values))
    mean = statistics.fmean(values)
    centered = [value - mean for value in values]
    long_run_variance = (
        sum(value * value for value in centered) / sample_size
    )
    for offset in range(1, effective_lag + 1):
        weight = 1.0 - offset / (effective_lag + 1.0)
        covariance = sum(
            centered[index] * centered[index - offset]
            for index in range(offset, len(centered))
        ) / sample_size
        long_run_variance += 2.0 * weight * covariance

    variance_of_mean = long_run_variance / sample_size
    if variance_of_mean <= 1e-20:
        return None, effective_lag
    return mean / math.sqrt(variance_of_mean), effective_lag


def two_sided_normal_p_value(t_stat: float | None) -> float | None:
    if t_stat is None:
        return None
    return math.erfc(abs(t_stat) / math.sqrt(2.0))


def summarize_ic_series(
    observations: list[tuple[str, float]],
    annualization_days: int,
    newey_west_lag: int,
) -> dict[str, Any]:
    if not observations:
        return {
            "status": "no_cross_sectional_variation",
            "start_date": None,
            "end_date": None,
            "trading_days": 0,
            "mean_rank_ic": None,
            "median_rank_ic": None,
            "mean_abs_rank_ic": None,
            "std_rank_ic": None,
            "ic_ir": None,
            "annualized_ic_ir": None,
            "positive_ic_ratio": None,
            "newey_west_lag": 0,
            "newey_west_t_stat": None,
            "newey_west_p_value": None,
        }
    dates = [trading_date for trading_date, _ in observations]
    values = [value for _, value in observations]
    mean_ic = statistics.fmean(values)
    std_ic = statistics.pstdev(values) if len(values) > 1 else 0.0
    ic_ir = mean_ic / std_ic if std_ic > 1e-20 else None
    t_stat, effective_lag = newey_west_t_stat(values, newey_west_lag)
    return {
        "status": "ok",
        "start_date": min(dates),
        "end_date": max(dates),
        "trading_days": len(values),
        "mean_rank_ic": mean_ic,
        "median_rank_ic": statistics.median(values),
        "mean_abs_rank_ic": statistics.fmean(abs(value) for value in values),
        "std_rank_ic": std_ic,
        "ic_ir": ic_ir,
        "annualized_ic_ir": (
            ic_ir * math.sqrt(annualization_days)
            if ic_ir is not None
            else None
        ),
        "positive_ic_ratio": sum(value > 0 for value in values) / len(values),
        "newey_west_lag": effective_lag,
        "newey_west_t_stat": t_stat,
        "newey_west_p_value": two_sided_normal_p_value(t_stat),
    }


def build_summary_records(
    split: str,
    horizon: int,
    factors: list[str],
    observations: dict[str, list[tuple[str, float]]],
    annualization_days: int,
    newey_west_lag: int,
) -> list[dict[str, Any]]:
    records = []
    for factor in factors:
        metrics = summarize_ic_series(
            observations[factor],
            annualization_days,
            newey_west_lag,
        )
        records.append(
            {
                "split": split,
                "horizon": horizon,
                "factor": factor,
                **metrics,
            }
        )
    return records


def build_yearly_records(
    split: str,
    horizon: int,
    factors: list[str],
    observations: dict[str, list[tuple[str, float]]],
    annualization_days: int,
    newey_west_lag: int,
) -> list[dict[str, Any]]:
    records = []
    for factor in factors:
        by_year: dict[int, list[tuple[str, float]]] = defaultdict(list)
        for trading_date, rank_ic in observations[factor]:
            by_year[int(trading_date[:4])].append((trading_date, rank_ic))
        for year in sorted(by_year):
            metrics = summarize_ic_series(
                by_year[year],
                annualization_days,
                newey_west_lag,
            )
            records.append(
                {
                    "split": split,
                    "horizon": horizon,
                    "factor": factor,
                    "year": year,
                    **metrics,
                }
            )
    return records


def compute_factor_pair_correlations(
    split: str,
    horizon: int,
    factors: list[str],
    rankings: dict[str, DailyRanking],
) -> list[dict[str, Any]]:
    daily_values: dict[tuple[str, str], list[float]] = {}
    for left_index, factor_a in enumerate(factors):
        for factor_b in factors[left_index + 1 :]:
            pair_values = []
            for trading_date in sorted(rankings):
                ranking = rankings[trading_date]
                value = correlation(
                    ranking.factors[factor_a],
                    ranking.factors[factor_b],
                )
                if value is not None:
                    pair_values.append(value)
            if pair_values:
                daily_values[(factor_a, factor_b)] = pair_values

    records = []
    for (factor_a, factor_b), values in daily_values.items():
        mean_correlation = statistics.fmean(values)
        records.append(
            {
                "split": split,
                "horizon": horizon,
                "factor_a": factor_a,
                "factor_b": factor_b,
                "trading_days": len(values),
                "mean_daily_spearman": mean_correlation,
                "mean_abs_daily_spearman": statistics.fmean(
                    abs(value) for value in values
                ),
                "positive_day_ratio": (
                    sum(value > 0 for value in values) / len(values)
                ),
            }
        )
    records.sort(
        key=lambda record: (
            -record["mean_abs_daily_spearman"],
            record["factor_a"],
            record["factor_b"],
        )
    )
    return records


def compact_ic_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "factor": record["factor"],
        "trading_days": record["trading_days"],
        "mean_rank_ic": record["mean_rank_ic"],
        "mean_abs_rank_ic": record["mean_abs_rank_ic"],
        "positive_ic_ratio": record["positive_ic_ratio"],
        "newey_west_t_stat": record["newey_west_t_stat"],
    }


def build_stability_diagnostics(
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
) -> dict[str, Any]:
    train_by_factor = {record["factor"]: record for record in train_records}
    validation_by_factor = {
        record["factor"]: record for record in validation_records
    }
    available_factors = [
        factor
        for factor in train_by_factor
        if (
            train_by_factor[factor]["mean_rank_ic"] is not None
            and validation_by_factor.get(factor, {}).get("mean_rank_ic")
            is not None
        )
    ]
    unavailable_factors = [
        factor
        for factor in train_by_factor
        if factor not in available_factors
    ]
    train_means = [
        train_by_factor[factor]["mean_rank_ic"] for factor in available_factors
    ]
    validation_means = [
        validation_by_factor[factor]["mean_rank_ic"]
        for factor in available_factors
    ]
    rank_correlation = correlation(
        average_ranks(train_means),
        average_ranks(validation_means),
    )

    direction_flips = [
        {
            "factor": factor,
            "train_mean_rank_ic": train_by_factor[factor]["mean_rank_ic"],
            "validation_mean_rank_ic": validation_by_factor[factor][
                "mean_rank_ic"
            ],
        }
        for factor in available_factors
        if (
            train_by_factor[factor]["mean_rank_ic"]
            * validation_by_factor[factor]["mean_rank_ic"]
            < 0
        )
    ]
    sign_consistent_factors = [
        factor
        for factor in available_factors
        if factor not in {item["factor"] for item in direction_flips}
    ]

    stable_factors = sorted(
        sign_consistent_factors,
        key=lambda factor: min(
            abs(train_by_factor[factor]["mean_rank_ic"]),
            abs(validation_by_factor[factor]["mean_rank_ic"]),
        ),
        reverse=True,
    )
    strongest_stable = []
    for factor in stable_factors:
        train_record = train_by_factor[factor]
        validation_record = validation_by_factor[factor]
        strongest_stable.append(
            {
                "factor": factor,
                "train_mean_rank_ic": train_record["mean_rank_ic"],
                "validation_mean_rank_ic": validation_record[
                    "mean_rank_ic"
                ],
                "stable_ic_magnitude": min(
                    abs(train_record["mean_rank_ic"]),
                    abs(validation_record["mean_rank_ic"]),
                ),
                "train_newey_west_t_stat": train_record[
                    "newey_west_t_stat"
                ],
                "validation_newey_west_t_stat": validation_record[
                    "newey_west_t_stat"
                ],
            }
        )

    def strongest(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ordered = sorted(
            (
                record
                for record in records
                if record["mean_rank_ic"] is not None
            ),
            key=lambda record: abs(record["mean_rank_ic"]),
            reverse=True,
        )
        return [compact_ic_record(record) for record in ordered[:8]]

    return {
        "available_factor_count": len(available_factors),
        "sign_consistent_factor_count": len(sign_consistent_factors),
        "direction_flip_count": len(direction_flips),
        "unavailable_factors": [
            {
                "factor": factor,
                "train_trading_days": train_by_factor[factor][
                    "trading_days"
                ],
                "validation_trading_days": validation_by_factor[factor][
                    "trading_days"
                ],
            }
            for factor in unavailable_factors
        ],
        "train_validation_mean_ic_spearman": rank_correlation,
        "direction_flips": direction_flips,
        "strongest_train_factors": strongest(train_records),
        "strongest_validation_factors": strongest(validation_records),
        "strongest_stable_factors": strongest_stable[:8],
    }


def format_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Cannot serialize non-finite value: {value!r}")
        return format(value, ".12g")
    return str(value)


def write_records(
    output_path: Path,
    fields: list[str],
    records: list[dict[str, Any]],
) -> None:
    rows = [
        {field: format_csv_value(record.get(field)) for field in fields}
        for record in records
    ]
    write_csv(output_path, fields, rows)


def analyze_split(
    split: str,
    input_path: Path,
    horizons: list[int],
    annualization_days: int,
    newey_west_lag: int | None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for horizon in horizons:
        rows = load_labeled_rows(input_path, horizon)
        rankings = rank_cross_sections(rows, FEATURE_FIELDS)
        observations = compute_daily_rank_ic(rankings, FEATURE_FIELDS)
        lag = horizon - 1 if newey_west_lag is None else newey_west_lag
        results[str(horizon)] = {
            "input_path": str(input_path),
            "rows": len(rows),
            "trading_dates": len(rankings),
            "start_date": min(row["date"] for row in rows),
            "end_date": max(row["date"] for row in rows),
            "newey_west_lag": lag,
            "summaries": build_summary_records(
                split,
                horizon,
                FEATURE_FIELDS,
                observations,
                annualization_days,
                lag,
            ),
            "yearly": build_yearly_records(
                split,
                horizon,
                FEATURE_FIELDS,
                observations,
                annualization_days,
                lag,
            ),
            "factor_pairs": compute_factor_pair_correlations(
                split,
                horizon,
                FEATURE_FIELDS,
                rankings,
            ),
        }
    return results


def run_analysis(
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    horizons: list[int],
    annualization_days: int = DEFAULT_ANNUALIZATION_DAYS,
    newey_west_lag: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_results = analyze_split(
        "train",
        train_path,
        horizons,
        annualization_days,
        newey_west_lag,
    )
    validation_results = analyze_split(
        "validation",
        validation_path,
        horizons,
        annualization_days,
        newey_west_lag,
    )

    summary_records = []
    yearly_records = []
    correlation_records = []
    stability: dict[str, Any] = {}
    for horizon in horizons:
        horizon_key = str(horizon)
        train_horizon = train_results[horizon_key]
        validation_horizon = validation_results[horizon_key]
        summary_records.extend(train_horizon["summaries"])
        summary_records.extend(validation_horizon["summaries"])
        yearly_records.extend(train_horizon["yearly"])
        yearly_records.extend(validation_horizon["yearly"])
        correlation_records.extend(train_horizon["factor_pairs"])
        correlation_records.extend(validation_horizon["factor_pairs"])
        stability[horizon_key] = build_stability_diagnostics(
            train_horizon["summaries"],
            validation_horizon["summaries"],
        )

    summary_path = output_dir / "factor_ic_summary.csv"
    yearly_path = output_dir / "factor_ic_yearly.csv"
    correlation_path = output_dir / "factor_correlation_pairs.csv"
    report_path = output_dir / "factor_ic_report.json"
    write_records(summary_path, SUMMARY_FIELDS, summary_records)
    write_records(yearly_path, YEARLY_FIELDS, yearly_records)
    write_records(
        correlation_path,
        CORRELATION_FIELDS,
        correlation_records,
    )

    report = {
        "created_on": date.today().isoformat(),
        "purpose": (
            "Factor-level RankIC stability diagnostics for training and "
            "validation. This analysis does not read test or final OOS data."
        ),
        "data": {
            "train_factor_file": str(train_path),
            "validation_factor_file": str(validation_path),
            "feature_count": len(FEATURE_FIELDS),
            "feature_fields": FEATURE_FIELDS,
            "horizons": horizons,
            "minimum_cross_section_size": MIN_CROSS_SECTION_SIZE,
            "train": train_results,
            "validation": validation_results,
        },
        "method": {
            "rank_ic": (
                "Daily cross-sectional Spearman correlation between each raw "
                "factor and its same-horizon forward total-return label."
            ),
            "aggregation": (
                "Simple time-series mean across daily RankIC observations; "
                "factor signs are not flipped or selected using validation."
            ),
            "ic_ir": "mean_rank_ic / population_std_rank_ic.",
            "annualized_ic_ir": (
                "ic_ir * sqrt(annualization_days), with "
                f"{annualization_days} days by default. Overlapping labels "
                "mean this is a diagnostic, not an independent-observation "
                "annualization."
            ),
            "newey_west": (
                "Bartlett-kernel HAC t-statistic for the mean daily RankIC. "
                "The default lag is horizon - 1 trading days to account for "
                "overlapping forward labels. P-values use a two-sided normal "
                "approximation."
            ),
            "yearly_stability": (
                "The same daily RankIC metrics computed within each calendar "
                "year. Partial first and last years are labeled by their "
                "actual date range."
            ),
            "factor_correlation": (
                "Mean daily cross-sectional Spearman correlation for each "
                "factor pair. It measures redundancy, not predictive value."
            ),
        },
        "stability": stability,
        "artifacts": {
            "report": str(report_path),
            "summary": str(summary_path),
            "yearly": str(yearly_path),
            "factor_correlations": str(correlation_path),
        },
        "limitations": [
            (
                "Validation contains one narrow industry universe and a short "
                "market regime; train-to-validation sign changes are not "
                "sufficient for causal claims."
            ),
            (
                "Overlapping 5/20-session labels induce serial dependence. "
                "Newey-West lags mitigate but do not eliminate this issue."
            ),
            (
                "RankIC measures cross-sectional ordering, not executable "
                "returns after costs, capacity, or A-share trading constraints."
            ),
        ],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    args = parse_args()
    horizons = list(dict.fromkeys(args.horizons))
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("Horizons must be positive integers.")
    if args.annualization_days <= 0:
        raise ValueError("Annualization days must be positive.")
    if args.newey_west_lag is not None and args.newey_west_lag < 0:
        raise ValueError("Newey-West lag must not be negative.")

    report = run_analysis(
        args.train_factors,
        args.validation_factors,
        args.output_dir,
        horizons,
        args.annualization_days,
        args.newey_west_lag,
    )
    for horizon in horizons:
        horizon_key = str(horizon)
        diagnostics = report["stability"][horizon_key]
        print(
            f"{horizon}d: "
            f"train/validation IC Spearman="
            f"{diagnostics['train_validation_mean_ic_spearman']}, "
            f"sign-consistent={diagnostics['sign_consistent_factor_count']}/"
            f"{diagnostics['available_factor_count']}"
        )
        for item in diagnostics["strongest_stable_factors"][:3]:
            print(
                f"  {item['factor']}: "
                f"train={item['train_mean_rank_ic']}, "
                f"validation={item['validation_mean_rank_ic']}"
            )
    print(f"Report: {report['artifacts']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
