from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from backtest_validation import (  # noqa: E402
    build_schedule,
    run_portfolio,
    select_codes,
)


class BacktestValidationTests(unittest.TestCase):
    def test_select_codes_ranks_scores_descending(self) -> None:
        scores = {
            "sh.600000": {"pred_knn": 0.1},
            "sh.600015": {"pred_knn": 0.3},
            "sz.000001": {"pred_knn": 0.2},
        }

        selected = select_codes(scores, "pred_knn", top_k=2, mode="top_k")

        self.assertEqual(selected, ["sh.600015", "sz.000001"])

    def test_build_schedule_executes_on_next_trading_date(self) -> None:
        predictions = {
            "2020-01-01": {
                "sh.600000": {"pred_knn": 0.2},
                "sh.600015": {"pred_knn": 0.1},
            },
            "2020-01-02": {
                "sh.600000": {"pred_knn": 0.3},
                "sh.600015": {"pred_knn": 0.1},
            },
        }
        calendar = ["2020-01-01", "2020-01-02", "2020-01-03"]

        schedule, _, _ = build_schedule(
            predictions,
            calendar,
            holding_days=2,
            top_k=1,
            score_field="pred_knn",
            mode="top_k",
        )

        self.assertEqual(len(schedule), 1)
        self.assertEqual(schedule[0], ("2020-01-02", ["sh.600000"]))

    def test_limit_up_prevents_buy(self) -> None:
        calendar = ["2020-01-01", "2020-01-02", "2020-01-03"]
        daily_by_key = {
            ("2020-01-01", "sh.600000"): {
                "date": "2020-01-01",
                "code": "sh.600000",
                "open": 10.0,
                "close": 10.0,
                "trading_status": "trading",
                "price_limit_status": "none",
            },
            ("2020-01-02", "sh.600000"): {
                "date": "2020-01-02",
                "code": "sh.600000",
                "open": 11.0,
                "close": 11.0,
                "trading_status": "trading",
                "price_limit_status": "limit_up",
            },
            ("2020-01-03", "sh.600000"): {
                "date": "2020-01-03",
                "code": "sh.600000",
                "open": 11.0,
                "close": 11.0,
                "trading_status": "trading",
                "price_limit_status": "none",
            },
        }

        result = run_portfolio(
            [("2020-01-02", ["sh.600000"])],
            calendar,
            daily_by_key,
            holding_days=1,
            initial_capital=10_000.0,
            commission_bps=0.0,
            stamp_tax_bps=0.0,
            slippage_bps=0.0,
            lot_size=1,
        )

        self.assertEqual(result["metrics"]["trade_count"], 0)
        self.assertAlmostEqual(result["metrics"]["final_nav"], 10_000.0)

    def test_buy_and_mark_to_market_uses_next_open(self) -> None:
        calendar = ["2020-01-01", "2020-01-02", "2020-01-03"]
        daily_by_key = {
            ("2020-01-02", "sh.600000"): {
                "date": "2020-01-02",
                "code": "sh.600000",
                "open": 10.0,
                "close": 11.0,
                "trading_status": "trading",
                "price_limit_status": "none",
            },
            ("2020-01-03", "sh.600000"): {
                "date": "2020-01-03",
                "code": "sh.600000",
                "open": 11.0,
                "close": 12.0,
                "trading_status": "trading",
                "price_limit_status": "none",
            },
        }

        result = run_portfolio(
            [("2020-01-02", ["sh.600000"])],
            calendar,
            daily_by_key,
            holding_days=1,
            initial_capital=10_000.0,
            commission_bps=0.0,
            stamp_tax_bps=0.0,
            slippage_bps=0.0,
            lot_size=1,
        )

        self.assertEqual(result["metrics"]["trade_count"], 1)
        self.assertAlmostEqual(result["metrics"]["final_nav"], 12_000.0)
        self.assertAlmostEqual(result["metrics"]["total_return"], 0.2)


if __name__ == "__main__":
    unittest.main()
