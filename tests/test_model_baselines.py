from __future__ import annotations

import sys
import unittest
import math
from datetime import date, timedelta
from pathlib import Path


sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fit_model_baselines import (  # noqa: E402
    FEATURE_FIELDS,
    correlation,
    evaluate_direction_predictions,
    fit_elastic_net,
    fit_logistic,
    fit_preprocessor,
    fit_ridge,
    predict_logistic_probability,
    predict_knn,
    transform_features,
    walk_forward_folds,
)


class ModelBaselineTests(unittest.TestCase):
    # 验证标准化参数仅由传入的训练记录拟合。
    def test_preprocessor_fits_and_applies_training_statistics(self) -> None:
        rows = []
        for index in range(20):
            rows.append(
                {
                    "date": (date(2020, 1, 1) + timedelta(days=index)).isoformat(),
                    **{
                        field: float(index + field_index)
                        for field_index, field in enumerate(FEATURE_FIELDS)
                    },
                }
            )

        params = fit_preprocessor(rows)
        matrix = transform_features(rows, params)

        self.assertEqual(params["fit_rows"], len(rows))
        self.assertEqual(len(matrix), len(rows))
        self.assertAlmostEqual(
            sum(row[0] for row in matrix) / len(matrix),
            0.0,
            places=10,
        )

    # 验证 Ridge 在线性模拟数据上可以恢复稳定的排序信号。
    def test_ridge_recovers_synthetic_linear_signal(self) -> None:
        matrix = [
            [
                math.sin(index * (feature + 1) * 0.13)
                + math.cos(index * (feature + 3) * 0.07)
                for feature in range(len(FEATURE_FIELDS))
            ]
            for index in range(120)
        ]
        targets = [row[0] * 0.7 - row[1] * 0.3 for row in matrix]

        coefficients = fit_ridge(matrix, targets, 1e-8)
        predictions = [
            sum(value * coefficient for value, coefficient in zip(row, coefficients))
            for row in matrix
        ]

        self.assertGreater(correlation(predictions, targets) or 0.0, 0.999)

    # 验证 Elastic Net 在稀疏正则化下仍能捕获模拟线性排序信号。
    def test_elastic_net_recovers_synthetic_linear_signal(self) -> None:
        matrix = [
            [
                math.sin(index * (feature + 1) * 0.13)
                + math.cos(index * (feature + 3) * 0.07)
                for feature in range(len(FEATURE_FIELDS))
            ]
            for index in range(120)
        ]
        targets = [row[0] * 0.7 - row[1] * 0.3 for row in matrix]
        coefficients = fit_elastic_net(matrix, targets, 1e-8, 0.5)
        predictions = [
            sum(value * coefficient for value, coefficient in zip(row, coefficients))
            for row in matrix
        ]

        self.assertGreater(correlation(predictions, targets) or 0.0, 0.999)

    # 验证 KNN 使用全部因子坐标，而不是只使用前几个因子。
    def test_knn_uses_all_factor_coordinates(self) -> None:
        first = [0.0] * len(FEATURE_FIELDS)
        second = [0.0] * len(FEATURE_FIELDS)
        second[-1] = 10.0
        query = [0.0] * len(FEATURE_FIELDS)
        query[-1] = 9.0

        predictions = predict_knn(
            [first, second],
            [-1.0, 1.0],
            [query],
            n_neighbors=1,
        )

        self.assertEqual(predictions, [1.0])

    # 验证时序交叉验证训练标签终点严格早于验证窗口。
    def test_walk_forward_folds_purge_cross_boundary_labels(self) -> None:
        start = date(2020, 1, 1)
        rows = []
        for index in range(100):
            signal_date = start + timedelta(days=index)
            rows.append(
                {
                    "date": signal_date.isoformat(),
                    "label_end_date": (signal_date + timedelta(days=5)).isoformat(),
                }
            )

        for fold_start, _, train_indices, test_indices in walk_forward_folds(rows):
            self.assertTrue(train_indices)
            self.assertTrue(test_indices)
            self.assertTrue(
                all(rows[index]["label_end_date"] < fold_start for index in train_indices)
            )
            self.assertTrue(
                all(rows[index]["date"] >= fold_start for index in test_indices)
            )

    # 验证方向模型能在简单线性信号上学习上涨概率。
    def test_logistic_direction_model_learns_binary_signal(self) -> None:
        matrix = [[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]]
        labels = [0, 0, 0, 1, 1, 1]

        intercept, coefficients = fit_logistic(matrix, labels, 1e-3)
        probabilities = predict_logistic_probability(
            matrix,
            intercept,
            coefficients,
        )
        rows = [
            {"date": f"2020-01-0{index + 1}", "target": str(label)}
            for index, label in enumerate(labels)
        ]
        rows = [
            {**row, "target": float(label)}
            for row, label in zip(rows, labels)
        ]
        metrics = evaluate_direction_predictions(rows, probabilities)

        self.assertGreater(coefficients[0], 0.0)
        self.assertGreater(metrics["roc_auc"], 0.95)
        self.assertGreater(metrics["accuracy"], 0.95)


if __name__ == "__main__":
    unittest.main()
