from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_point_in_time_panel import (  # noqa: E402
    build_panel,
    load_membership,
    membership_asof,
)
from download_historical_constituents import (  # noqa: E402
    normalize_snapshot_rows,
    weekly_mondays,
)


class PointInTimePanelTests(unittest.TestCase):
    def test_weekly_mondays_uses_strict_date_range(self) -> None:
        self.assertEqual(
            weekly_mondays("2020-01-01", "2020-01-20"),
            ["2020-01-06", "2020-01-13", "2020-01-20"],
        )
        with self.assertRaises(ValueError):
            weekly_mondays("2020-01-02", "2019-12-31")

    def test_snapshot_normalization_handles_baostock_field_names(self) -> None:
        rows = normalize_snapshot_rows(
            "2020-01-06",
            [
                {
                    "updateDate": "2019-12-30",
                    "code": "SH.600000",
                    "code_name": "浦发银行",
                    "industry": "J66",
                    "industryClassification": "证监会行业分类",
                },
                {
                    "updateDate": "2020-01-02",
                    "code": "sz.000001",
                    "code_name": "平安银行",
                    "industry": "J65",
                },
            ],
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["code"], "sh.600000")
        self.assertEqual(rows[0]["snapshot_date"], "2020-01-06")
        self.assertEqual(rows[0]["effective_date"], "2020-01-06")
        self.assertEqual(rows[0]["source_update_date"], "2019-12-30")

    def test_membership_asof_does_not_use_later_snapshot(self) -> None:
        events = [
            {
                "_effective_date": "2019-01-01",
                "_snapshot_date": "2020-01-06",
                "source": "fixture",
            },
            {
                "_effective_date": "2020-02-03",
                "_snapshot_date": "2020-02-03",
                "source": "fixture",
            },
        ]

        self.assertIsNone(membership_asof(events, "2019-12-31"))
        self.assertEqual(
            membership_asof(events, "2020-01-10")["source"],
            "fixture",
        )
        self.assertEqual(
            membership_asof(events, "2020-02-10")["_effective_date"],
            "2020-02-03",
        )

    def test_build_panel_applies_effective_and_snapshot_dates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            daily_path = root / "daily.csv"
            membership_path = root / "membership.csv"
            output_path = root / "out.csv"
            report_path = root / "report.json"

            with daily_path.open("w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(
                    target,
                    fieldnames=["date", "code", "close"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"date": "2019-12-31", "code": "sh.600000", "close": "10"},
                        {"date": "2020-01-10", "code": "sh.600000", "close": "11"},
                        {"date": "2020-02-10", "code": "sh.600000", "close": "12"},
                    ]
                )

            with membership_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as target:
                writer = csv.DictWriter(
                    target,
                    fieldnames=[
                        "snapshot_date",
                        "effective_date",
                        "code",
                        "industry_code",
                        "source",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "snapshot_date": "2020-01-06",
                            "effective_date": "2019-01-01",
                            "code": "sh.600000",
                            "industry_code": "J66",
                            "source": "fixture",
                        },
                        {
                            "snapshot_date": "2020-02-03",
                            "effective_date": "2020-02-03",
                            "code": "sh.600000",
                            "industry_code": "J66",
                            "source": "fixture",
                        },
                    ]
                )

            report = build_panel(
                daily_path,
                membership_path,
                output_path,
                report_path,
                "J66",
            )

            self.assertEqual(report["counts"]["input_rows"], 3)
            self.assertEqual(
                report["counts"]["dropped_without_point_in_time_membership"],
                1,
            )
            with output_path.open(encoding="utf-8-sig", newline="") as source:
                output_rows = list(csv.DictReader(source))
            self.assertEqual(len(output_rows), 2)
            self.assertEqual(output_rows[0]["membership_snapshot_date"], "2020-01-06")
            self.assertEqual(output_rows[1]["membership_effective_date"], "2020-02-03")

    def test_load_membership_rejects_future_effective_date(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "membership.csv"
            path.write_text(
                "snapshot_date,effective_date,code,industry_code\n"
                "2020-01-06,2020-01-07,sh.600000,J66\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_membership(path, "J66")


class SnapshotDownloadTests(unittest.TestCase):
    def test_download_retries_and_preserves_all_snapshot_rows(self) -> None:
        from download_historical_constituents import download_snapshots

        calls: list[str] = []

        def query(snapshot_date: str) -> list[dict[str, str]]:
            calls.append(snapshot_date)
            if len(calls) == 1:
                raise RuntimeError("temporary failure")
            return [
                {
                    "snapshot_date": snapshot_date,
                    "effective_date": snapshot_date,
                    "code": "sh.600000",
                    "code_name": "浦发银行",
                    "industry_code": "J66",
                    "industry_name": "货币金融服务",
                    "classification": "fixture",
                    "source_update_date": "",
                    "source": "fixture",
                }
            ]

        rows = download_snapshots(
            SimpleNamespace(
                start_date="2020-01-06",
                end_date="2020-01-06",
                retries=2,
                retry_delay=0.0,
            ),
            query=query,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
