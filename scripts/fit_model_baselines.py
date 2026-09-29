#!/usr/bin/env python3
"""Fit train-only preprocessing and evaluate cross-sectional baselines."""

from __future__ import annotations

import argparse
import csv
import heapq
import importlib.util
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from build_train_factors import (
    FEATURE_FIELDS,
    build_dataset,
    read_history,
    read_training,
    write_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_FACTORS = PROJECT_ROOT / "data/processed/train_factor_labels.csv"
DEFAULT_VALIDATION_INPUT = PROJECT_ROOT / "data/cleaned/splits/validation.csv"
DEFAULT_HISTORY_INPUT = PROJECT_ROOT / "data/j66_money_finance_daily_2018_2024.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed"
HORIZONS = (5, 20)
RIDGE_ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
ELASTIC_NET_CONFIGS = tuple(
    (alpha, l1_ratio)
    for alpha in (1e-5, 1e-4, 1e-3, 1e-2)
    for l1_ratio in (0.15, 0.5, 0.85)
)
KNN_NEIGHBOR_COUNTS = (5, 15, 30, 60)
LOGISTIC_L2_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
DIRECTION_THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
NUMBER_PATTERN = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d{1,6})?$"
)

if importlib.util.find_spec("numpy") is not None:
    import numpy as np
else:
    np = None


# 解析训练因子、验证行情、历史行情及实验产物路径。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit preprocessing on training data, select linear/KNN models "
            "with purged walk-forward folds, and report validation performance."
        )
    )
    parser.add_argument("--train-factors", type=Path, default=DEFAULT_TRAIN_FACTORS)
    parser.add_argument("--validation-input", type=Path, default=DEFAULT_VALIDATION_INPUT)
    parser.add_argument("--history-input", type=Path, default=DEFAULT_HISTORY_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


# 计算带线性插值的样本分位数。
def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


# 将模型输入文本转换为有限浮点数，无法解析时返回空值。
def parse_finite_float_or_none(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not NUMBER_PATTERN.fullmatch(normalized):
        return None
    parsed = float(normalized)
    return parsed if math.isfinite(parsed) else None


# 将有限实数压缩到逻辑函数可稳定计算的范围。
def sigmoid(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(min(value, 60.0))
    return exponent / (1.0 + exponent)


# 用带 L2 正则的对角牛顿法拟合股票上涨概率。
def fit_logistic(
    matrix: list[list[float]],
    labels: list[int],
    l2: float,
    max_iterations: int = 100,
) -> tuple[float, list[float]]:
    if not matrix or len(matrix) != len(labels):
        raise ValueError("Logistic regression requires matching non-empty inputs.")
    if l2 < 0:
        raise ValueError("Logistic L2 penalty must not be negative.")
    feature_count = len(matrix[0])
    if any(len(row) != feature_count for row in matrix):
        raise ValueError("Logistic regression rows must have equal width.")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Logistic labels must be binary.")

    positive_rate = sum(labels) / len(labels)
    positive_rate = min(max(positive_rate, 1e-6), 1.0 - 1e-6)
    intercept = math.log(positive_rate / (1.0 - positive_rate))
    coefficients = [0.0] * feature_count

    for _ in range(max_iterations):
        gradient_intercept = 0.0
        gradient = [0.0] * feature_count
        curvature_intercept = 0.0
        curvature = [0.0] * feature_count
        for row, label in zip(matrix, labels):
            score = intercept + sum(value * coefficient for value, coefficient in zip(row, coefficients))
            probability = sigmoid(score)
            residual = label - probability
            weight = max(probability * (1.0 - probability), 1e-6)
            gradient_intercept += residual
            curvature_intercept += weight
            for index, value in enumerate(row):
                gradient[index] += residual * value
                curvature[index] += weight * value * value

        scale = 1.0 / len(matrix)
        intercept_step = scale * gradient_intercept / max(curvature_intercept * scale, 1e-8)
        intercept += max(-1.0, min(1.0, intercept_step))
        largest_step = abs(intercept_step)
        for index in range(feature_count):
            regularized_gradient = scale * gradient[index] - l2 * coefficients[index]
            regularized_curvature = curvature[index] * scale + l2
            step = regularized_gradient / max(regularized_curvature, 1e-8)
            step = max(-1.0, min(1.0, step))
            coefficients[index] += step
            largest_step = max(largest_step, abs(step))
        if largest_step < 1e-6:
            break
    return intercept, coefficients


# 根据逻辑回归参数计算上涨概率。
def predict_logistic_probability(
    matrix: list[list[float]],
    intercept: float,
    coefficients: list[float],
) -> list[float]:
    return [
        sigmoid(
            intercept
            + sum(value * coefficient for value, coefficient in zip(row, coefficients))
        )
        for row in matrix
    ]


# 仅根据给定训练样本拟合缺失填补、缩尾和标准化参数。
def fit_preprocessor(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feature_params: dict[str, dict[str, float]] = {}
    for field in FEATURE_FIELDS:
        values = [
            row[field]
            for row in rows
            if row[field] is not None and math.isfinite(row[field])
        ]
        if not values:
            raise ValueError(f"Training feature {field} has no finite values.")
        median = quantile(values, 0.5)
        lower = quantile(values, 0.01)
        upper = quantile(values, 0.99)
        clipped = [min(max(value, lower), upper) for value in values]
        mean = statistics.fmean(clipped)
        scale = statistics.pstdev(clipped)
        feature_params[field] = {
            "missing_value": median,
            "clip_lower_1pct": lower,
            "clip_upper_99pct": upper,
            "mean": mean,
            "scale": scale if scale > 1e-12 else 1.0,
        }
    return {
        "fit_rows": len(rows),
        "fit_start_date": min(row["date"] for row in rows),
        "fit_end_date": max(row["date"] for row in rows),
        "features": feature_params,
    }


# 使用训练集拟合的参数填补、缩尾并标准化一组因子。
def transform_features(
    rows: list[dict[str, Any]],
    params: dict[str, Any],
) -> list[list[float]]:
    matrix: list[list[float]] = []
    for row in rows:
        transformed = []
        for field in FEATURE_FIELDS:
            config = params["features"][field]
            value = row[field]
            if value is None or not math.isfinite(value):
                value = config["missing_value"]
            value = min(max(value, config["clip_lower_1pct"]), config["clip_upper_99pct"])
            transformed.append((value - config["mean"]) / config["scale"])
        matrix.append(transformed)
    return matrix


# 按预测期限读取因子、收益标签和标签终点，并校验数值有效性。
def load_labeled_rows(
    input_path: Path,
    horizon: int,
) -> list[dict[str, Any]]:
    label_field = f"fwd_return_{horizon}d"
    endpoint_field = f"label_{horizon}d_end_date"
    rows: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"date", "code", label_field, endpoint_field, *FEATURE_FIELDS}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{input_path} is missing required modeling columns.")
        for line_number, raw in enumerate(reader, start=2):
            features: dict[str, float | None] = {}
            invalid_feature = False
            for field in FEATURE_FIELDS:
                raw_value = raw.get(field)
                if raw_value is None:
                    invalid_feature = True
                    features[field] = None
                elif not raw_value.strip():
                    features[field] = None
                else:
                    parsed = parse_finite_float_or_none(raw_value)
                    features[field] = parsed
                    invalid_feature = invalid_feature or parsed is None
            target = parse_finite_float_or_none(raw.get(label_field))
            if invalid_feature or target is None:
                raise ValueError(
                    f"Invalid modeling value at {input_path}:{line_number}."
                )
            rows.append(
                {
                    **features,
                    "date": raw["date"],
                    "code": raw["code"],
                    "target": target,
                    "label_end_date": raw[endpoint_field],
                }
            )
    if not rows:
        raise ValueError(f"No labeled rows found in {input_path}.")
    return rows


# 为每个信号交易日计算相对当日样本均值的收益标签。
def cross_sectional_targets(rows: list[dict[str, Any]]) -> list[float]:
    by_date: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_date[row["date"]].append(row["target"])
    means = {
        trading_date: statistics.fmean(values)
        for trading_date, values in by_date.items()
    }
    return [row["target"] - means[row["date"]] for row in rows]


# 仅使用拟合期标签估计缩尾区间、中心和尺度。
def fit_target_preprocessor(values: list[float]) -> dict[str, float]:
    lower = quantile(values, 0.01)
    upper = quantile(values, 0.99)
    clipped = [min(max(value, lower), upper) for value in values]
    mean = statistics.fmean(clipped)
    scale = statistics.pstdev(clipped)
    return {
        "clip_lower_1pct": lower,
        "clip_upper_99pct": upper,
        "mean": mean,
        "scale": scale if scale > 1e-12 else 1.0,
    }


# 将截面相对收益按训练期标签参数缩尾并标准化。
def transform_targets(
    values: list[float],
    params: dict[str, float],
) -> list[float]:
    return [
        (
            min(max(value, params["clip_lower_1pct"]), params["clip_upper_99pct"])
            - params["mean"]
        )
        / params["scale"]
        for value in values
    ]


# 使用全部候选因子，以欧氏距离的最近邻加权平均预测标准化收益。
def predict_knn(
    train_matrix: list[list[float]],
    train_targets: list[float],
    query_matrix: list[list[float]],
    n_neighbors: int,
    weights: str = "distance",
    block_size: int = 512,
) -> list[float]:
    if not train_matrix or not train_targets:
        raise ValueError("KNN requires at least one training sample.")
    if len(train_matrix) != len(train_targets):
        raise ValueError("KNN matrix and target lengths must match.")
    if n_neighbors < 1:
        raise ValueError("KNN n_neighbors must be positive.")
    if weights not in {"uniform", "distance"}:
        raise ValueError(f"Unsupported KNN weights: {weights!r}.")
    if not query_matrix:
        return []
    feature_count = len(FEATURE_FIELDS)
    if any(len(row) != feature_count for row in train_matrix + query_matrix):
        raise ValueError(
            f"KNN expects exactly {feature_count} standardized features per row."
        )
    neighbor_count = min(n_neighbors, len(train_matrix))

    # NumPy is optional so the repository remains runnable with only the
    # standard library; the accelerated path makes full walk-forward KNN
    # practical on the project-sized data.
    if np is not None:
        train_array = np.asarray(train_matrix, dtype=np.float64)
        target_array = np.asarray(train_targets, dtype=np.float64)
        train_squared_norm = np.sum(train_array * train_array, axis=1)
        predictions: list[float] = []
        for start in range(0, len(query_matrix), block_size):
            query_array = np.asarray(
                query_matrix[start : start + block_size],
                dtype=np.float64,
            )
            query_squared_norm = np.sum(query_array * query_array, axis=1)
            distances = (
                query_squared_norm[:, None]
                + train_squared_norm[None, :]
                - 2.0 * query_array @ train_array.T
            )
            distances = np.maximum(distances, 0.0)
            nearest = np.argpartition(
                distances,
                kth=neighbor_count - 1,
                axis=1,
            )[:, :neighbor_count]
            nearest_distances = np.take_along_axis(distances, nearest, axis=1)
            nearest_targets = target_array[nearest]
            if weights == "uniform":
                block_predictions = np.mean(nearest_targets, axis=1)
            else:
                exact = nearest_distances <= 1e-24
                exact_count = np.sum(exact, axis=1)
                exact_predictions = np.divide(
                    np.sum(nearest_targets * exact, axis=1),
                    exact_count,
                    out=np.zeros(len(query_array), dtype=np.float64),
                    where=exact_count > 0,
                )
                inverse_distance = 1.0 / np.sqrt(
                    np.maximum(nearest_distances, 1e-24)
                )
                weighted_predictions = np.sum(
                    inverse_distance * nearest_targets,
                    axis=1,
                ) / np.sum(inverse_distance, axis=1)
                block_predictions = np.where(
                    exact_count > 0,
                    exact_predictions,
                    weighted_predictions,
                )
            predictions.extend(float(value) for value in block_predictions)
        return predictions

    predictions = []
    for query in query_matrix:
        nearest_heap: list[tuple[float, int]] = []
        for train_index, train_row in enumerate(train_matrix):
            squared_distance = sum(
                (query[feature] - train_row[feature]) ** 2
                for feature in range(feature_count)
            )
            candidate = (-squared_distance, train_index)
            if len(nearest_heap) < neighbor_count:
                heapq.heappush(nearest_heap, candidate)
            elif squared_distance < -nearest_heap[0][0]:
                heapq.heapreplace(nearest_heap, candidate)
        nearest_distances = [-item[0] for item in nearest_heap]
        nearest_targets = [train_targets[item[1]] for item in nearest_heap]
        exact_targets = [
            target
            for distance, target in zip(nearest_distances, nearest_targets)
            if distance <= 1e-24
        ]
        if exact_targets:
            predictions.append(statistics.fmean(exact_targets))
        elif weights == "uniform":
            predictions.append(statistics.fmean(nearest_targets))
        else:
            inverse_distance = [
                1.0 / math.sqrt(max(distance, 1e-24))
                for distance in nearest_distances
            ]
            predictions.append(
                sum(weight * target for weight, target in zip(
                    inverse_distance,
                    nearest_targets,
                ))
                / sum(inverse_distance)
            )
    return predictions


# 将 KNN 的标准化收益预测还原到原始收益量纲。
def predict_knn_regression(
    train_matrix: list[list[float]],
    train_targets: list[float],
    query_matrix: list[list[float]],
    n_neighbors: int,
    target_params: dict[str, float],
) -> list[float]:
    scaled_predictions = predict_knn(
        train_matrix,
        train_targets,
        query_matrix,
        n_neighbors,
    )
    return [
        value * target_params["scale"] + target_params["mean"]
        for value in scaled_predictions
    ]


# 按预测期限创建扩张窗口式交叉验证区间。
def walk_forward_folds(
    rows: list[dict[str, Any]],
    fold_count: int = 4,
) -> list[tuple[str, str, list[int], list[int]]]:
    dates = sorted({row["date"] for row in rows})
    folds = []
    for fold_number in range(fold_count):
        start_fraction = 0.55 + fold_number * 0.10
        end_fraction = start_fraction + 0.10
        start_index = min(int(len(dates) * start_fraction), len(dates) - 1)
        end_index = min(int(len(dates) * end_fraction), len(dates))
        if end_index <= start_index:
            continue
        start_date = dates[start_index]
        end_date = dates[end_index - 1]
        train_indices = [
            index
            for index, row in enumerate(rows)
            if row["date"] < start_date and row["label_end_date"] < start_date
        ]
        test_indices = [
            index
            for index, row in enumerate(rows)
            if start_date <= row["date"] <= end_date
        ]
        if train_indices and test_indices:
            folds.append((start_date, end_date, train_indices, test_indices))
    if not folds:
        raise ValueError("Could not form non-empty purged walk-forward folds.")
    return folds


# 对矩阵和标签计算归一化 Gram 矩阵及特征-标签协方差。
def gram_statistics(
    matrix: list[list[float]],
    targets: list[float],
) -> tuple[list[list[float]], list[float]]:
    feature_count = len(FEATURE_FIELDS)
    sample_count = len(matrix)
    gram = [[0.0] * feature_count for _ in range(feature_count)]
    cross = [0.0] * feature_count
    for row, target in zip(matrix, targets):
        for left in range(feature_count):
            cross[left] += row[left] * target
            for right in range(left, feature_count):
                gram[left][right] += row[left] * row[right]
    for left in range(feature_count):
        cross[left] /= sample_count
        for right in range(left, feature_count):
            gram[left][right] /= sample_count
            gram[right][left] = gram[left][right]
    return gram, cross


# 使用带主元选择的高斯消元法求解小型线性方程组。
def solve_linear_system(
    matrix: list[list[float]],
    vector: list[float],
) -> list[float]:
    size = len(vector)
    augmented = [matrix[row][:] + [vector[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise ValueError("Ridge system is numerically singular.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for position in range(column, size + 1):
            augmented[column][position] /= pivot_value
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for position in range(column, size + 1):
                augmented[row][position] -= factor * augmented[column][position]
    return [augmented[row][size] for row in range(size)]


# 使用 Ridge 正则化拟合线性回归系数。
def fit_ridge(
    matrix: list[list[float]],
    targets: list[float],
    alpha: float,
) -> list[float]:
    gram, cross = gram_statistics(matrix, targets)
    return fit_ridge_from_statistics(gram, cross, alpha)


# 根据已计算的 Gram 矩阵和协方差求解 Ridge 系数。
def fit_ridge_from_statistics(
    gram: list[list[float]],
    cross: list[float],
    alpha: float,
) -> list[float]:
    regularized = [row[:] for row in gram]
    for index in range(len(cross)):
        regularized[index][index] += alpha
    return solve_linear_system(regularized, cross)


# 使用循环坐标下降拟合 Elastic Net 回归系数。
def fit_elastic_net(
    matrix: list[list[float]],
    targets: list[float],
    alpha: float,
    l1_ratio: float,
    max_iterations: int = 300,
) -> list[float]:
    gram, cross = gram_statistics(matrix, targets)
    return fit_elastic_net_from_statistics(
        gram,
        cross,
        alpha,
        l1_ratio,
        max_iterations,
    )


# 根据已计算的 Gram 矩阵和协方差执行 Elastic Net 坐标下降。
def fit_elastic_net_from_statistics(
    gram: list[list[float]],
    cross: list[float],
    alpha: float,
    l1_ratio: float,
    max_iterations: int = 300,
) -> list[float]:
    coefficients = [0.0] * len(cross)
    l1_penalty = alpha * l1_ratio
    l2_penalty = alpha * (1.0 - l1_ratio)
    for _ in range(max_iterations):
        largest_change = 0.0
        for index in range(len(coefficients)):
            partial = cross[index] - sum(
                gram[index][other] * coefficients[other]
                for other in range(len(coefficients))
                if other != index
            )
            if partial > l1_penalty:
                updated = (partial - l1_penalty) / (
                    gram[index][index] + l2_penalty
                )
            elif partial < -l1_penalty:
                updated = (partial + l1_penalty) / (
                    gram[index][index] + l2_penalty
                )
            else:
                updated = 0.0
            largest_change = max(
                largest_change,
                abs(updated - coefficients[index]),
            )
            coefficients[index] = updated
        if largest_change < 1e-8:
            break
    return coefficients


# 根据线性系数对标准化特征矩阵生成预测分数。
def predict_linear(
    matrix: list[list[float]],
    coefficients: list[float],
    target_params: dict[str, float],
) -> list[float]:
    return [
        (
            sum(value * coefficient for value, coefficient in zip(row, coefficients))
            * target_params["scale"]
            + target_params["mean"]
        )
        for row in matrix
    ]


# 计算训练集因子方向并构造等权方向因子组合。
def fit_factor_composite(
    matrix: list[list[float]],
    targets: list[float],
) -> list[int]:
    _, cross = gram_statistics(matrix, targets)
    return [1 if value > 0 else -1 if value < 0 else 0 for value in cross]


# 将方向因子组合转换为收益量纲的预测值。
def predict_factor_composite(
    matrix: list[list[float]],
    directions: list[int],
    target_params: dict[str, float],
) -> list[float]:
    scale = target_params["scale"] / math.sqrt(len(directions))
    return [
        sum(value * direction for value, direction in zip(row, directions))
        / len(directions)
        * scale
        for row in matrix
    ]


# 对并列值使用平均秩，生成 Spearman 相关所需的秩序列。
def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


# 计算两组数值的 Pearson 相关系数。
def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_norm = math.sqrt(sum(value * value for value in left_centered))
    right_norm = math.sqrt(sum(value * value for value in right_centered))
    if left_norm == 0 or right_norm == 0:
        return None
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered)
    ) / (left_norm * right_norm)


# 计算二分类上涨标签与预测概率的 ROC-AUC。
def binary_auc(labels: list[int], probabilities: list[float]) -> float | None:
    if len(labels) != len(probabilities):
        return None
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    ranks = average_ranks(probabilities)
    positive_rank_sum = sum(
        rank for rank, label in zip(ranks, labels) if label == 1
    )
    return (
        positive_rank_sum - positive_count * (positive_count - 1) / 2.0
    ) / (positive_count * negative_count)


# 评估预测后续收盘上涨/不涨方向的准确率和概率质量。
def evaluate_direction_predictions(
    rows: list[dict[str, Any]],
    probabilities: list[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    if len(rows) != len(probabilities) or not rows:
        raise ValueError("Direction evaluation requires matching non-empty inputs.")
    if not 0.0 < threshold < 1.0:
        raise ValueError("Direction threshold must be between zero and one.")
    labels = [int(row["target"] > 0.0) for row in rows]
    predicted = [int(probability >= threshold) for probability in probabilities]
    true_positive = sum(
        label == 1 and prediction == 1
        for label, prediction in zip(labels, predicted)
    )
    true_negative = sum(
        label == 0 and prediction == 0
        for label, prediction in zip(labels, predicted)
    )
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    sensitivity = true_positive / positive_count if positive_count else None
    specificity = true_negative / negative_count if negative_count else None
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["date"]].append(index)
    daily_accuracy = [
        statistics.fmean(
            labels[index] == predicted[index] for index in indices
        )
        for indices in grouped.values()
    ]
    daily_balanced_accuracy = []
    for indices in grouped.values():
        day_labels = [labels[index] for index in indices]
        day_predictions = [predicted[index] for index in indices]
        day_positive = sum(day_labels)
        day_negative = len(day_labels) - day_positive
        day_tp = sum(
            label == 1 and prediction == 1
            for label, prediction in zip(day_labels, day_predictions)
        )
        day_tn = sum(
            label == 0 and prediction == 0
            for label, prediction in zip(day_labels, day_predictions)
        )
        if day_positive and day_negative:
            daily_balanced_accuracy.append(
                0.5 * (day_tp / day_positive + day_tn / day_negative)
            )
    accuracy = statistics.fmean(
        label == prediction for label, prediction in zip(labels, predicted)
    )
    majority_accuracy = max(positive_count, negative_count) / len(labels)
    return {
        "rows": len(rows),
        "trading_dates": len(grouped),
        "accuracy": accuracy,
        "majority_baseline_accuracy": majority_accuracy,
        "accuracy_vs_majority": accuracy - majority_accuracy,
        "balanced_accuracy": (
            (sensitivity + specificity) / 2.0
            if sensitivity is not None and specificity is not None
            else None
        ),
        "sensitivity_up_recall": sensitivity,
        "specificity_down_recall": specificity,
        "roc_auc": binary_auc(labels, probabilities),
        "brier_score": statistics.fmean(
            (probability - label) ** 2
            for probability, label in zip(probabilities, labels)
        ),
        "predicted_up_rate": statistics.fmean(predicted),
        "actual_up_rate": statistics.fmean(labels),
        "mean_daily_accuracy": statistics.fmean(daily_accuracy),
        "mean_daily_balanced_accuracy": (
            statistics.fmean(daily_balanced_accuracy)
            if daily_balanced_accuracy
            else None
        ),
    }


# 按交易日计算 RankIC 和多空分组收益，并汇总误差指标。
def evaluate_predictions(
    rows: list[dict[str, Any]],
    predictions: list[float],
    centered_targets: list[float],
) -> tuple[dict[str, Any], dict[str, dict[str, float | None]]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["date"]].append(index)

    daily: dict[str, dict[str, float | None]] = {}
    for trading_date, indices in grouped.items():
        scores = [predictions[index] for index in indices]
        returns = [rows[index]["target"] for index in indices]
        rank_ic = correlation(average_ranks(scores), average_ranks(returns))
        bucket_size = max(1, math.ceil(len(indices) * 0.20))

        # 对边界处的同分样本按比例分配分组权重。
        def bucket_mean(ordered_indices: list[int]) -> float:
            remaining = float(bucket_size)
            weighted_return = 0.0
            start = 0
            while start < len(ordered_indices):
                if remaining <= 0:
                    break
                score = predictions[ordered_indices[start]]
                tied = []
                while (
                    start + len(tied) < len(ordered_indices)
                    and predictions[ordered_indices[start + len(tied)]] == score
                ):
                    tied.append(ordered_indices[start + len(tied)])
                selected = min(remaining, float(len(tied)))
                weighted_return += (
                    statistics.fmean(rows[index]["target"] for index in tied)
                    * selected
                )
                remaining -= selected
                start += len(tied)
            return weighted_return / bucket_size

        ordered = sorted(indices, key=lambda index: predictions[index])
        bottom = bucket_mean(ordered)
        top = bucket_mean(list(reversed(ordered)))
        daily[trading_date] = {
            "rank_ic": rank_ic,
            "top_bottom_spread": top - bottom,
            "top_return": top,
            "bottom_return": bottom,
        }

    rank_ics = [
        values["rank_ic"]
        for values in daily.values()
        if values["rank_ic"] is not None
    ]
    spreads = [values["top_bottom_spread"] for values in daily.values()]
    errors = [
        centered_targets[index] - predictions[index]
        for index in range(len(rows))
    ]
    mean_ic = statistics.fmean(rank_ics) if rank_ics else None
    ic_std = statistics.pstdev(rank_ics) if len(rank_ics) > 1 else None
    metrics = {
        "rows": len(rows),
        "trading_dates": len(daily),
        "mean_daily_rank_ic": mean_ic,
        "daily_rank_ic_std": ic_std,
        "rank_ic_information_ratio": (
            mean_ic / ic_std if mean_ic is not None and ic_std else None
        ),
        "positive_rank_ic_day_ratio": (
            sum(value > 0 for value in rank_ics) / len(rank_ics)
            if rank_ics
            else None
        ),
        "mean_top_bottom_20pct_spread_bps": (
            statistics.fmean(spreads) * 10_000 if spreads else None
        ),
        "positive_top_bottom_day_ratio": (
            sum(value > 0 for value in spreads) / len(spreads) if spreads else None
        ),
        "top_20pct_mean_return_bps": (
            statistics.fmean(values["top_return"] for values in daily.values())
            * 10_000
        ),
        "bottom_20pct_mean_return_bps": (
            statistics.fmean(values["bottom_return"] for values in daily.values())
            * 10_000
        ),
        "centered_target_mae_bps": (
            statistics.fmean(abs(error) for error in errors) * 10_000
        ),
        "centered_target_rmse_bps": (
            math.sqrt(statistics.fmean(error * error for error in errors)) * 10_000
        ),
    }
    return metrics, daily


# 根据 purged walk-forward 结果选择每类线性模型的正则化参数。
def select_model_configs(
    rows: list[dict[str, Any]],
    centered_targets: list[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    folds = walk_forward_folds(rows)
    oof_predictions: dict[str, list[float]] = defaultdict(list)
    oof_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    oof_centered_targets: dict[str, list[float]] = defaultdict(list)
    fold_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    direction_oof_predictions: dict[str, list[float]] = defaultdict(list)
    direction_oof_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    direction_fold_records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for start_date, end_date, train_indices, test_indices in folds:
        train_rows = [rows[index] for index in train_indices]
        test_rows = [rows[index] for index in test_indices]
        preprocessing = fit_preprocessor(train_rows)
        train_matrix = transform_features(train_rows, preprocessing)
        test_matrix = transform_features(test_rows, preprocessing)
        train_target_params = fit_target_preprocessor(
            [centered_targets[index] for index in train_indices]
        )
        train_targets = transform_targets(
            [centered_targets[index] for index in train_indices],
            train_target_params,
        )
        gram, cross = gram_statistics(train_matrix, train_targets)
        test_centered = [centered_targets[index] for index in test_indices]
        candidates: dict[str, list[float]] = {}
        directions = fit_factor_composite(train_matrix, train_targets)
        candidates["factor_composite"] = predict_factor_composite(
            test_matrix,
            directions,
            train_target_params,
        )
        for alpha in RIDGE_ALPHAS:
            coefficients = fit_ridge_from_statistics(gram, cross, alpha)
            key = f"ridge_alpha_{alpha:g}"
            candidates[key] = predict_linear(test_matrix, coefficients, train_target_params)
        for alpha, l1_ratio in ELASTIC_NET_CONFIGS:
            coefficients = fit_elastic_net_from_statistics(
                gram,
                cross,
                alpha,
                l1_ratio,
            )
            key = f"elastic_net_alpha_{alpha:g}_l1_{l1_ratio:g}"
            candidates[key] = predict_linear(test_matrix, coefficients, train_target_params)
        if np is not None:
            for n_neighbors in KNN_NEIGHBOR_COUNTS:
                key = f"knn_k_{n_neighbors}"
                candidates[key] = predict_knn_regression(
                    train_matrix,
                    train_targets,
                    test_matrix,
                    n_neighbors,
                    train_target_params,
                )

        direction_labels = [int(row["target"] > 0.0) for row in train_rows]
        direction_candidates: dict[str, list[float]] = {}
        for l2 in LOGISTIC_L2_ALPHAS:
            intercept, coefficients = fit_logistic(
                train_matrix,
                direction_labels,
                l2,
            )
            probabilities = predict_logistic_probability(
                test_matrix,
                intercept,
                coefficients,
            )
            for threshold in DIRECTION_THRESHOLDS:
                key = f"logistic_l2_{l2:g}_threshold_{threshold:g}"
                direction_candidates[key] = probabilities

        for key, predictions in candidates.items():
            metrics, _ = evaluate_predictions(test_rows, predictions, test_centered)
            fold_records[key].append(
                {
                    "start_date": start_date,
                    "end_date": end_date,
                    "train_rows": len(train_indices),
                    "test_rows": len(test_indices),
                    "metrics": metrics,
                }
            )
            oof_predictions[key].extend(predictions)
            oof_rows[key].extend(test_rows)
            oof_centered_targets[key].extend(test_centered)

        for key, probabilities in direction_candidates.items():
            threshold = float(key.rsplit("_", 1)[1])
            direction_metrics = evaluate_direction_predictions(
                test_rows,
                probabilities,
                threshold,
            )
            direction_fold_records[key].append(
                {
                    "start_date": start_date,
                    "end_date": end_date,
                    "train_rows": len(train_indices),
                    "test_rows": len(test_indices),
                    "direction_metrics": direction_metrics,
                }
            )
            direction_oof_predictions[key].extend(probabilities)
            direction_oof_rows[key].extend(test_rows)

    candidate_report: dict[str, Any] = {}
    for key in oof_predictions:
        metrics, _ = evaluate_predictions(
            oof_rows[key],
            oof_predictions[key],
            oof_centered_targets[key],
        )
        candidate_report[key] = {
            "metrics": metrics,
            "folds": fold_records[key],
        }
    for key in direction_oof_predictions:
        candidate_rows = direction_oof_rows[key]
        probabilities = direction_oof_predictions[key]
        threshold = float(key.rsplit("_", 1)[1])
        candidate_report[key] = {
            "metrics": evaluate_predictions(
                candidate_rows,
                probabilities,
                cross_sectional_targets(candidate_rows),
            )[0],
            "direction_metrics": evaluate_direction_predictions(
                candidate_rows,
                probabilities,
                threshold,
            ),
            "folds": direction_fold_records[key],
        }

    selected: dict[str, Any] = {}
    for family in ("ridge", "elastic_net", "knn"):
        candidates = [
            key
            for key in candidate_report
            if key.startswith(f"{family}_")
        ]
        if not candidates:
            continue
        selected[family] = max(
            candidates,
            key=lambda key: (
                (
                    candidate_report[key]["metrics"]["mean_daily_rank_ic"]
                    if candidate_report[key]["metrics"]["mean_daily_rank_ic"]
                    is not None
                    else -math.inf
                ),
                (
                    candidate_report[key]["metrics"][
                        "mean_top_bottom_20pct_spread_bps"
                    ]
                    if candidate_report[key]["metrics"][
                        "mean_top_bottom_20pct_spread_bps"
                    ]
                    is not None
                    else -math.inf
                ),
            ),
        )
    direction_candidates = [
        key for key in candidate_report if key.startswith("logistic_")
    ]
    selected["direction"] = max(
        direction_candidates,
        key=lambda key: (
            candidate_report[key]["direction_metrics"]["balanced_accuracy"]
            if candidate_report[key]["direction_metrics"]["balanced_accuracy"]
            is not None
            else -math.inf,
            candidate_report[key]["direction_metrics"]["roc_auc"]
            if candidate_report[key]["direction_metrics"]["roc_auc"] is not None
            else -math.inf,
            -candidate_report[key]["direction_metrics"]["brier_score"],
        ),
    )
    return selected, candidate_report


# 使用原始日行情为验证集构造过去因子和仅限验证期的前瞻标签。
def build_validation_factors(
    validation_input: Path,
    history_input: Path,
) -> tuple[list[str], list[dict[str, str]], dict[str, Any]]:
    source_fields, rows, dates, rows_by_date = read_training(validation_input)
    histories, history_report = read_history(history_input, dates[-1])
    fields, output_rows, drops, audit = build_dataset(
        source_fields=source_fields,
        training_rows=rows,
        trading_dates=dates,
        training_rows_by_date=rows_by_date,
        histories=histories,
    )
    audit["dropped_rows_by_reason"] = {
        key: int(drops[key]) for key in sorted(drops)
    }
    audit["history_context"] = history_report
    return fields, output_rows, audit


# 对完整训练集拟合选定模型，并输出验证集预测和逐模型指标。
def run_horizon_experiment(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    horizon: int,
) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    train_targets = cross_sectional_targets(train_rows)
    validation_targets = cross_sectional_targets(validation_rows)
    selected, cv_report = select_model_configs(train_rows, train_targets)

    preprocessing = fit_preprocessor(train_rows)
    train_matrix = transform_features(train_rows, preprocessing)
    validation_matrix = transform_features(validation_rows, preprocessing)
    target_params = fit_target_preprocessor(train_targets)
    scaled_targets = transform_targets(train_targets, target_params)
    gram, cross = gram_statistics(train_matrix, scaled_targets)
    validation_centered = validation_targets
    train_direction_labels = [int(row["target"] > 0.0) for row in train_rows]
    train_up_rate = statistics.fmean(train_direction_labels)
    model_predictions: dict[str, list[float]] = {
        "zero_baseline": [0.0] * len(validation_rows),
        "direction_prior": [train_up_rate] * len(validation_rows),
    }

    directions = fit_factor_composite(train_matrix, scaled_targets)
    model_predictions["factor_composite"] = predict_factor_composite(
        validation_matrix,
        directions,
        target_params,
    )

    selected_params: dict[str, Any] = {}
    for family, selected_key in selected.items():
        if family == "ridge":
            alpha = float(selected_key.rsplit("_", 1)[1])
            coefficients = fit_ridge_from_statistics(gram, cross, alpha)
            selected_params[family] = {"alpha": alpha}
            model_predictions[family] = predict_linear(
                validation_matrix,
                coefficients,
                target_params,
            )
            selected_params[f"{family}_coefficients"] = {
                field: coefficient
                for field, coefficient in sorted(
                    zip(FEATURE_FIELDS, coefficients),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )
            }
        elif family == "elastic_net":
            parts = selected_key.split("_")
            alpha = float(parts[3])
            l1_ratio = float(parts[5])
            coefficients = fit_elastic_net_from_statistics(
                gram,
                cross,
                alpha,
                l1_ratio,
            )
            selected_params[family] = {
                "alpha": alpha,
                "l1_ratio": l1_ratio,
            }
            model_predictions[family] = predict_linear(
                validation_matrix,
                coefficients,
                target_params,
            )
            selected_params[f"{family}_coefficients"] = {
                field: coefficient
                for field, coefficient in sorted(
                    zip(FEATURE_FIELDS, coefficients),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )
            }
        elif family == "direction":
            parts = selected_key.split("_")
            l2 = float(parts[2])
            threshold = float(parts[4])
            intercept, coefficients = fit_logistic(
                train_matrix,
                train_direction_labels,
                l2,
            )
            direction_probabilities = predict_logistic_probability(
                validation_matrix,
                intercept,
                coefficients,
            )
            selected_params[family] = {
                "model": "logistic_regression",
                "l2": l2,
                "threshold": threshold,
                "training_up_rate": train_up_rate,
            }
            selected_params[f"{family}_coefficients"] = {
                "intercept": intercept,
                **{
                    field: coefficient
                    for field, coefficient in sorted(
                        zip(FEATURE_FIELDS, coefficients),
                        key=lambda item: abs(item[1]),
                        reverse=True,
                    )
                },
            }
            model_predictions["logistic_direction"] = direction_probabilities
        else:
            n_neighbors = int(selected_key.rsplit("_", 1)[1])
            selected_params[family] = {
                "n_neighbors": n_neighbors,
                "weights": "distance",
                "feature_count": len(FEATURE_FIELDS),
                "features": FEATURE_FIELDS,
            }
            model_predictions[family] = predict_knn_regression(
                train_matrix,
                scaled_targets,
                validation_matrix,
                n_neighbors,
                target_params,
            )

    metrics: dict[str, Any] = {}
    daily_metrics: dict[str, dict[str, dict[str, float | None]]] = {}
    for model_name, predictions in model_predictions.items():
        model_metrics, daily = evaluate_predictions(
            validation_rows,
            predictions,
            validation_centered,
        )
        metrics[model_name] = model_metrics
        daily_metrics[model_name] = daily
    direction_metrics = {
        "direction_prior": evaluate_direction_predictions(
            validation_rows,
            model_predictions["direction_prior"],
        ),
        "logistic_direction": evaluate_direction_predictions(
            validation_rows,
            model_predictions["logistic_direction"],
            selected_params["direction"]["threshold"],
        ),
    }

    composite_metrics = metrics["factor_composite"]
    for model_name in ("ridge", "elastic_net", "knn"):
        if model_name not in metrics:
            continue
        metrics[model_name]["increment_vs_factor_composite"] = {
            "mean_daily_rank_ic": (
                metrics[model_name]["mean_daily_rank_ic"]
                - composite_metrics["mean_daily_rank_ic"]
                if metrics[model_name]["mean_daily_rank_ic"] is not None
                and composite_metrics["mean_daily_rank_ic"] is not None
                else None
            ),
            "mean_top_bottom_20pct_spread_bps": (
                metrics[model_name]["mean_top_bottom_20pct_spread_bps"]
                - composite_metrics["mean_top_bottom_20pct_spread_bps"]
            ),
        }

    prediction_rows = []
    for index, row in enumerate(validation_rows):
        result = {
            "date": row["date"],
            "code": row["code"],
            f"fwd_return_{horizon}d": format(row["target"], ".12g"),
            "cross_sectional_excess_return": format(
                validation_targets[index],
                ".12g",
            ),
        }
        for model_name, predictions in model_predictions.items():
            result[f"pred_{model_name}"] = format(predictions[index], ".12g")
        result["pred_logistic_direction_label"] = str(
            int(
                model_predictions["logistic_direction"][index]
                >= selected_params["direction"]["threshold"]
            )
        )
        prediction_rows.append(result)

    experiment = {
        "horizon_trading_days": horizon,
        "training_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "training_period": {
            "start_date": min(row["date"] for row in train_rows),
            "end_date": max(row["date"] for row in train_rows),
        },
        "validation_period": {
            "start_date": min(row["date"] for row in validation_rows),
            "end_date": max(row["date"] for row in validation_rows),
        },
        "preprocessing": preprocessing,
        "target_preprocessing": target_params,
        "selected_configs": selected_params,
        "walk_forward_selection": {
            "fold_count": len(walk_forward_folds(train_rows)),
            "selection_metric": "mean daily RankIC, tie-broken by top-bottom spread",
            "candidates": cv_report,
        },
        "validation_metrics": metrics,
        "validation_direction_metrics": direction_metrics,
        "validation_daily_metrics": daily_metrics,
    }
    return experiment, prediction_rows, {
        "preprocessing": preprocessing,
        "target_preprocessing": target_params,
        "selected_configs": selected_params,
    }


# 将单一验证期表现转化为谨慎的候选模型结论。
def summarize_recommendations(
    experiments: dict[str, Any],
) -> dict[str, Any]:
    recommendations: dict[str, Any] = {}
    for horizon, experiment in experiments.items():
        metrics = experiment["validation_metrics"]
        composite = metrics["factor_composite"]
        candidates = []
        model_names = ["factor_composite", "ridge", "elastic_net"]
        if "knn" in metrics:
            model_names.append("knn")
        for name in model_names:
            result = metrics[name]
            candidates.append(
                {
                    "model": name,
                    "mean_daily_rank_ic": result["mean_daily_rank_ic"],
                    "mean_top_bottom_20pct_spread_bps": result[
                        "mean_top_bottom_20pct_spread_bps"
                    ],
                    "increment_vs_factor_composite": result.get(
                        "increment_vs_factor_composite"
                    ),
                }
            )
        candidates.sort(
            key=lambda item: (
                item["mean_daily_rank_ic"]
                if item["mean_daily_rank_ic"] is not None
                else -math.inf,
                item["mean_top_bottom_20pct_spread_bps"]
                if item["mean_top_bottom_20pct_spread_bps"] is not None
                else -math.inf,
            ),
            reverse=True,
        )
        advancing = [
            item["model"]
            for item in candidates
            if item["model"] != "factor_composite"
            and item["mean_daily_rank_ic"] is not None
            and item["mean_daily_rank_ic"] > 0
            and item["mean_top_bottom_20pct_spread_bps"] is not None
            and item["mean_top_bottom_20pct_spread_bps"] > 0
            and metrics[item["model"]]["increment_vs_factor_composite"][
                "mean_daily_rank_ic"
            ]
            > 0
            and metrics[item["model"]]["increment_vs_factor_composite"][
                "mean_top_bottom_20pct_spread_bps"
            ]
            > 0
        ]
        recommendations[horizon] = {
            "validation_ranking": candidates,
            "models_worth_advancing_to_cost_aware_backtest": advancing,
            "direction_metrics": experiment["validation_direction_metrics"],
            "conclusion": (
                "当前验证期未显示候选模型相对简单因子组合的双指标增量；"
                "不建议据此宣称模型具有稳定选股价值。"
                if not advancing
                else "候选模型在当前验证期的 RankIC 与分组多空收益均优于因子组合；"
                "仅建议进入成本敏感性回测，不视为已验证实盘价值。"
            ),
            "factor_composite_reference": {
                "mean_daily_rank_ic": composite["mean_daily_rank_ic"],
                "mean_top_bottom_20pct_spread_bps": composite[
                    "mean_top_bottom_20pct_spread_bps"
                ],
            },
        }
    return recommendations


# 执行验证因子构造、训练期拟合、模型比较及报告保存。
def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_fields, validation_factor_rows, validation_audit = (
        build_validation_factors(args.validation_input, args.history_input)
    )
    validation_factor_path = args.output_dir / "validation_factor_labels.csv"
    write_csv(validation_factor_path, validation_fields, validation_factor_rows)

    experiments: dict[str, Any] = {}
    prediction_files: dict[str, str] = {}
    preprocessing_files: dict[str, str] = {}
    for horizon in HORIZONS:
        train_rows = load_labeled_rows(args.train_factors, horizon)
        validation_rows = load_labeled_rows(validation_factor_path, horizon)
        experiment, predictions, fitted = run_horizon_experiment(
            train_rows,
            validation_rows,
            horizon,
        )
        experiments[str(horizon)] = experiment
        prediction_path = args.output_dir / f"validation_predictions_{horizon}d.csv"
        prediction_fields = list(predictions[0])
        write_csv(prediction_path, prediction_fields, predictions)
        prediction_files[str(horizon)] = str(prediction_path)
        preprocessing_path = args.output_dir / f"preprocessing_parameters_{horizon}d.json"
        preprocessing_path.write_text(
            json.dumps(fitted, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        preprocessing_files[str(horizon)] = str(preprocessing_path)

    recommendations = summarize_recommendations(experiments)
    report = {
        "created_on": date.today().isoformat(),
        "purpose": (
            "Training-only preprocessing and baseline model comparison; "
            "validation is used for evaluation only."
        ),
        "data": {
            "training_factor_file": str(args.train_factors),
            "validation_raw_file": str(args.validation_input),
            "history_context_file": str(args.history_input),
            "validation_factor_file": str(validation_factor_path),
            "validation_factor_rows": len(validation_factor_rows),
            "validation_factor_audit": validation_audit,
            "test_and_final_oos_used": False,
        },
        "method": {
            "features": FEATURE_FIELDS,
            "target": (
                "5/20-session raw close-to-close return minus same-signal-date "
                "sample mean; used as a cross-sectional learning target, not an "
                "investable benchmark return."
            ),
            "feature_preprocessing": (
                "Per-feature 1st/99th percentile clipping, median imputation, "
                "then global standardization; every parameter is fitted on the "
                "corresponding training fold or full training set only."
            ),
            "target_preprocessing": (
                "Training-only 1st/99th percentile clipping and standardization."
            ),
            "walk_forward": (
                "Four expanding folds; each training observation is purged unless "
                "its label endpoint is strictly before that fold's validation start."
            ),
            "validation_metrics": (
                "Daily cross-sectional RankIC, top/bottom 20% forward-return spread, "
                "and centered-return prediction errors. Overlapping labels mean "
                "these are predictive diagnostics, not a portfolio backtest."
            ),
            "models": [
                "zero_baseline",
                "factor_composite",
                "ridge",
                "elastic_net",
                "knn",
                "logistic_direction",
            ],
            "direction_target": (
                "Binary label: whether the exact 5/20-session forward close return "
                "is strictly positive. Model selection uses purged walk-forward "
                "balanced accuracy, then ROC-AUC and Brier score."
            ),
            "knn": {
                "algorithm": (
                    "All available standardized factor fields are used as "
                    "Euclidean-distance coordinates."
                ),
                "weights": "distance",
                "candidate_neighbor_counts": list(KNN_NEIGHBOR_COUNTS),
                "implementation": (
                    "NumPy-accelerated exact search when NumPy is installed; "
                    "standard-library exact fallback otherwise."
                ),
            },
        },
        "experiments": experiments,
        "recommendations": recommendations,
        "artifacts": {
            "validation_factor_labels": str(validation_factor_path),
            "predictions": prediction_files,
            "preprocessing_parameters": preprocessing_files,
        },
        "limitations": [
            "Validation covers only 2023-03-20 through 2023-10-24 and one narrow J66 industry universe.",
            "Labels use unadjusted close-to-close prices and do not represent executable returns.",
            "The data uses a current industry constituent snapshot and may contain survivorship bias.",
            "No trading costs, turnover limits, market impact, or A-share execution constraints are modeled.",
            "The validation period is small and model comparisons are exploratory; test and final OOS remain untouched.",
        ],
    }
    report_path = args.output_dir / "model_baseline_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for horizon, result in recommendations.items():
        print(f"{horizon}d: {result['conclusion']}")
        for candidate in result["validation_ranking"]:
            print(
                f"  {candidate['model']}: "
                f"RankIC={candidate['mean_daily_rank_ic']}, "
                f"spread={candidate['mean_top_bottom_20pct_spread_bps']} bps"
            )
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
