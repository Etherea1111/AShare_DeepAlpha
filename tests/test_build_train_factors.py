from __future__ import annotations

import sys
import unittest
from datetime import date, timedelta
from pathlib import Path


sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_train_factors import (  # noqa: E402
    Bar,
    build_dataset,
    calculate_features,
)


def make_bar(day: date, index: int, multiplier: float = 1.0) -> Bar:
    close = (10.0 + index * 0.1) * multiplier
    return Bar(
        trading_date=day,
        open=close * 0.999,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        preclose=close / 1.001,
        volume=1000.0 + index,
        amount=close * (1000.0 + index),
        turnover=0.5 + index / 1000.0,
    )


class FactorLeakageTests(unittest.TestCase):
    def test_future_bars_cannot_change_past_factors(self) -> None:
        start = date(2020, 1, 1)
        bars = [make_bar(start + timedelta(days=i), i) for i in range(90)]
        as_of_index = 70

        original = calculate_features(bars, as_of_index)
        changed_future = list(bars)
        for index in range(as_of_index + 1, len(changed_future)):
            changed_future[index] = make_bar(
                changed_future[index].trading_date,
                index,
                multiplier=100.0,
            )
        after_future_change = calculate_features(changed_future, as_of_index)

        self.assertIsNotNone(original)
        self.assertEqual(original, after_future_change)

    def test_labels_use_exact_future_training_dates_only(self) -> None:
        start = date(2020, 1, 1)
        dates = [start + timedelta(days=i) for i in range(100)]
        code = "sh.600000"
        bars = [make_bar(day, i) for i, day in enumerate(dates)]
        training_rows = [
            {"date": day.isoformat(), "code": code, "close": str(bars[i].close)}
            for i, day in enumerate(dates)
        ]
        rows_by_date = {
            day: {code: training_rows[i]} for i, day in enumerate(dates)
        }

        _, output, _, audit = build_dataset(
            source_fields=["date", "code", "close"],
            training_rows=training_rows,
            trading_dates=dates,
            training_rows_by_date=rows_by_date,
            histories={code: bars},
        )

        self.assertEqual(len(output), 20)
        self.assertTrue(audit["feature_history_is_past_only"])
        self.assertTrue(audit["labels_end_within_training_dates"])
        self.assertTrue(audit["label_endpoints_strictly_after_signal_date"])
        self.assertTrue(audit["label_horizons_match_training_calendar_offsets"])
        self.assertEqual(output[0]["label_5d_end_date"], dates[65].isoformat())
        self.assertEqual(output[0]["label_20d_end_date"], dates[80].isoformat())
        self.assertEqual(output[-1]["label_20d_end_date"], dates[-1].isoformat())


if __name__ == "__main__":
    unittest.main()
