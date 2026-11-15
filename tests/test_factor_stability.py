from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path


sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_factor_stability import (  # noqa: E402
    build_yearly_records,
    compute_daily_rank_ic,
    compute_factor_pair_correlations,
    newey_west_t_stat,
    rank_cross_sections,
    run_analysis,
    summarize_ic_series,
)
from build_train_factors import FEATURE_FIELDS  # noqa: E402


def make_rank_rows(
    trading_date: str,
    *,
    factor_a: list[float],
    factor_b: list[float],
    target: list[float],
) -> list[dict[str, object]]:
    return [
        {
            "date": trading_date,
            "code": f"stock_{index}",
            "factor_a": factor_a[index],
            "factor_b": factor_b[index],
            "target": target[index],
        }
        for index in range(len(target))
    ]


def make_labeled_csv_rows(
    start: date,
    trading_dates: int,
    stocks: int,
) -> list[dict[str, object]]:
    rows = []
    for day_index in range(trading_dates):
        signal_date = start + timedelta(days=day_index)
        for stock_index in range(stocks):
            value = float(stock_index + 1)
            features = {field: value for field in FEATURE_FIELDS}
            features["market_return_1"] = 0.0
            rows.append(
                {
                    "date": signal_date.isoformat(),
                    "code": f"sh.{600000 + stock_index}",
                    "fwd_return_5d": value / 1000.0,
                    "label_5d_end_date": (
                        signal_date + timedelta(days=7)
                    ).isoformat(),
                    **features,
                }
            )
    return rows


def write_labeled_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    fields = [
        "date",
        "code",
        "fwd_return_5d",
        "label_5d_end_date",
        *FEATURE_FIELDS,
    ]
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class FactorStabilityTests(unittest.TestCase):
    def test_daily_rank_ic_and_factor_pair_correlation(self) -> None:
        rows = make_rank_rows(
            "2020-01-02",
            factor_a=[1.0, 2.0, 3.0, 4.0, 5.0],
            factor_b=[5.0, 4.0, 3.0, 2.0, 1.0],
            target=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        factors = ["factor_a", "factor_b"]

        rankings = rank_cross_sections(rows, factors)
        observations = compute_daily_rank_ic(rankings, factors)
        pair_records = compute_factor_pair_correlations(
            "train",
            5,
            factors,
            rankings,
        )

        self.assertAlmostEqual(observations["factor_a"][0][1], 1.0)
        self.assertAlmostEqual(observations["factor_b"][0][1], -1.0)
        self.assertAlmostEqual(
            pair_records[0]["mean_daily_spearman"],
            -1.0,
        )

    def test_small_cross_sections_are_excluded(self) -> None:
        rows = make_rank_rows(
            "2020-01-02",
            factor_a=[1.0, 2.0],
            factor_b=[2.0, 1.0],
            target=[1.0, 2.0],
        )

        rankings = rank_cross_sections(rows, ["factor_a", "factor_b"])

        self.assertEqual(rankings, {})

    def test_newey_west_zero_lag_matches_conventional_t_stat(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        t_stat, effective_lag = newey_west_t_stat(values, 0)

        expected = statistics_mean(values) / math.sqrt(2.0 / len(values))
        self.assertEqual(effective_lag, 0)
        self.assertAlmostEqual(t_stat, expected)

    def test_summary_and_yearly_grouping(self) -> None:
        observations = [
            ("2020-12-30", 0.1),
            ("2020-12-31", 0.2),
            ("2021-01-04", -0.1),
        ]

        summary = summarize_ic_series(observations, 252, 4)
        yearly = build_yearly_records(
            "train",
            5,
            ["momentum"],
            {"momentum": observations},
            252,
            4,
        )

        self.assertAlmostEqual(summary["mean_rank_ic"], 0.2 / 3.0)
        self.assertAlmostEqual(summary["positive_ic_ratio"], 2.0 / 3.0)
        self.assertEqual([record["year"] for record in yearly], [2020, 2021])
        self.assertEqual(yearly[0]["trading_days"], 2)
        self.assertAlmostEqual(yearly[1]["mean_rank_ic"], -0.1)

    def test_empty_rank_ic_series_is_explicitly_unavailable(self) -> None:
        summary = summarize_ic_series([], 252, 4)

        self.assertEqual(summary["status"], "no_cross_sectional_variation")
        self.assertEqual(summary["trading_days"], 0)
        self.assertIsNone(summary["mean_rank_ic"])

    def test_analysis_writes_json_and_csv_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train_path = root / "train.csv"
            validation_path = root / "validation.csv"
            output_dir = root / "analysis"
            write_labeled_csv(
                train_path,
                make_labeled_csv_rows(date(2020, 1, 2), 4, 5),
            )
            write_labeled_csv(
                validation_path,
                make_labeled_csv_rows(date(2021, 1, 4), 2, 5),
            )

            report = run_analysis(
                train_path,
                validation_path,
                output_dir,
                [5],
            )

            summary_path = output_dir / "factor_ic_summary.csv"
            yearly_path = output_dir / "factor_ic_yearly.csv"
            correlation_path = output_dir / "factor_correlation_pairs.csv"
            report_path = output_dir / "factor_ic_report.json"
            self.assertTrue(summary_path.exists())
            self.assertTrue(yearly_path.exists())
            self.assertTrue(correlation_path.exists())
            self.assertTrue(report_path.exists())
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                loaded["data"]["feature_count"],
                len(FEATURE_FIELDS),
            )
            self.assertIn("5", report["stability"])
            self.assertEqual(
                report["stability"]["5"]["unavailable_factors"],
                [
                    {
                        "factor": "market_return_1",
                        "train_trading_days": 0,
                        "validation_trading_days": 0,
                    }
                ],
            )
            with summary_path.open(encoding="utf-8-sig", newline="") as source:
                self.assertEqual(len(list(csv.DictReader(source))), 42)


def statistics_mean(values: list[float]) -> float:
    return sum(values) / len(values)


if __name__ == "__main__":
    unittest.main()
