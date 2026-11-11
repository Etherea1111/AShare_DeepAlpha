#!/usr/bin/env python3
"""Run a cost-aware validation-period portfolio backtest."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS_DIR = PROJECT_ROOT / "data/processed"
DEFAULT_DAILY_INPUT = (
    PROJECT_ROOT / "data/cleaned/j66_money_finance_daily_2018_2024.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/backtest"
HORIZONS = (5, 20)
MODEL_FIELDS = (
    "pred_factor_composite",
    "pred_ridge",
    "pred_elastic_net",
    "pred_knn",
    "pred_logistic_direction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest validation predictions with next-open execution, A-share "
            "tradability rules, and explicit transaction costs."
        )
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR,
    )
    parser.add_argument("--daily-input", type=Path, default=DEFAULT_DAILY_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--commission-bps", type=float, default=3.0)
    parser.add_argument("--stamp-tax-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--lot-size", type=int, default=100)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return reader.fieldnames or [], list(reader)


def parse_number(value: object, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid number in {context}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite number in {context}: {value!r}")
    return parsed


def floor_to_lot(shares: float, lot_size: int) -> int:
    if shares <= 0:
        return 0
    return int(shares // lot_size) * lot_size


def load_daily_panel(
    path: Path,
) -> tuple[list[str], dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    fields, rows = read_csv(path)
    required = {
        "date",
        "code",
        "open",
        "close",
        "trading_status",
        "price_limit_status",
    }
    missing = required - set(fields)
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    by_date: dict[str, dict[str, Any]] = defaultdict(dict)
    for index, row in enumerate(rows, start=2):
        daily_date = row["date"]
        code = row["code"]
        record = {
            "date": daily_date,
            "code": code,
            "open": parse_number(row["open"], f"{path}:{index} open"),
            "close": parse_number(row["close"], f"{path}:{index} close"),
            "trading_status": row["trading_status"],
            "price_limit_status": row["price_limit_status"],
        }
        if record["open"] <= 0 or record["close"] <= 0:
            raise ValueError(f"Non-positive price in {path}:{index}")
        by_key[(daily_date, code)] = record
        by_date[daily_date][code] = record
    return sorted(by_date), by_key, by_date


def load_predictions(
    path: Path,
) -> tuple[list[str], dict[str, dict[str, dict[str, float]]]]:
    fields, rows = read_csv(path)
    missing = {"date", "code", *MODEL_FIELDS} - set(fields)
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    predictions: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for index, row in enumerate(rows, start=2):
        predictions[row["date"]][row["code"]] = {
            field: parse_number(row[field], f"{path}:{index} {field}")
            for field in MODEL_FIELDS
        }
    return fields, predictions


def next_trading_date(calendar: list[str], signal_date: str) -> str | None:
    try:
        index = calendar.index(signal_date)
    except ValueError:
        return None
    return calendar[index + 1] if index + 1 < len(calendar) else None


def select_codes(
    scores: dict[str, dict[str, float]],
    score_field: str,
    top_k: int,
    mode: str,
) -> list[str]:
    if mode == "equal_weight":
        return sorted(scores)
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    ranked = sorted(
        (
            (entry[score_field], code)
            for code, entry in scores.items()
        ),
        key=lambda item: (-item[0], item[1]),
    )
    return [code for _, code in ranked[:top_k]]


def build_schedule(
    predictions: dict[str, dict[str, dict[str, float]]],
    calendar: list[str],
    holding_days: int,
    top_k: int,
    score_field: str | None,
    mode: str,
) -> tuple[list[tuple[str, list[str]]], str, str]:
    prediction_dates = sorted(predictions)
    if not prediction_dates:
        raise ValueError("Prediction file has no dated rows.")
    signal_dates = prediction_dates[::holding_days]
    schedule: list[tuple[str, list[str]]] = []
    last_signal_date = signal_dates[0]
    for signal_date in signal_dates:
        execution_date = next_trading_date(calendar, signal_date)
        if execution_date is None:
            continue
        field = score_field
        if mode == "top_k" and field is None:
            raise ValueError("top_k mode requires a score field.")
        codes = select_codes(
            predictions[signal_date],
            field or MODEL_FIELDS[0],
            top_k,
            mode,
        )
        if codes:
            schedule.append((execution_date, codes))
            last_signal_date = signal_date
    if not schedule:
        raise ValueError("Could not build a non-empty rebalance schedule.")
    return schedule, signal_dates[0], last_signal_date


def can_trade(
    row: dict[str, Any] | None,
    side: str,
) -> bool:
    if row is None or row["trading_status"] != "trading":
        return False
    if side == "buy" and row["price_limit_status"] == "limit_up":
        return False
    if side == "sell" and row["price_limit_status"] == "limit_down":
        return False
    return True


def position_value(
    positions: dict[str, int],
    prices: dict[str, float],
) -> float:
    return sum(shares * prices.get(code, 0.0) for code, shares in positions.items())


def run_portfolio(
    schedule: list[tuple[str, list[str]]],
    calendar: list[str],
    daily_by_key: dict[tuple[str, str], dict[str, Any]],
    holding_days: int,
    initial_capital: float,
    commission_bps: float,
    stamp_tax_bps: float,
    slippage_bps: float,
    lot_size: int,
) -> dict[str, Any]:
    cash = initial_capital
    positions: dict[str, int] = {}
    last_buy_date: dict[str, str] = {}
    last_close: dict[str, float] = {}
    schedule_by_date = {execution_date: codes for execution_date, codes in schedule}
    all_dates = [item for item, _ in schedule]
    start_index = calendar.index(all_dates[0])
    end_index = min(
        calendar.index(all_dates[-1]) + holding_days,
        len(calendar) - 1,
    )
    backtest_dates = calendar[start_index : end_index + 1]
    nav_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    turnover_notional = 0.0
    total_commission = 0.0
    total_stamp_tax = 0.0
    total_slippage = 0.0
    previous_nav = initial_capital

    for trading_date in backtest_dates:
        if trading_date in schedule_by_date:
            target_codes = schedule_by_date[trading_date]
            open_prices: dict[str, float] = {}
            for code, shares in positions.items():
                row = daily_by_key.get((trading_date, code))
                price = row["open"] if row else last_close.get(code)
                if price is not None:
                    open_prices[code] = price
            nav_at_open = cash + position_value(positions, open_prices)
            target_weight = 1.0 / len(target_codes) if target_codes else 0.0

            for code in sorted(positions):
                row = daily_by_key.get((trading_date, code))
                if row is None:
                    continue
                desired_value = (
                    nav_at_open * target_weight if code in target_codes else 0.0
                )
                desired_shares = floor_to_lot(
                    desired_value / row["open"],
                    lot_size,
                )
                sell_shares = positions[code] - desired_shares
                if sell_shares <= 0:
                    continue
                if last_buy_date.get(code) == trading_date:
                    continue
                if not can_trade(row, "sell"):
                    continue
                sell_price = row["open"] * (1.0 - slippage_bps / 10_000.0)
                gross = sell_shares * sell_price
                commission = gross * commission_bps / 10_000.0
                stamp_tax = gross * stamp_tax_bps / 10_000.0
                slippage = sell_shares * row["open"] * slippage_bps / 10_000.0
                cash += gross - commission - stamp_tax
                positions[code] -= sell_shares
                if positions[code] == 0:
                    positions.pop(code)
                    last_buy_date.pop(code, None)
                turnover_notional += gross
                total_commission += commission
                total_stamp_tax += stamp_tax
                total_slippage += slippage
                trade_rows.append(
                    {
                        "date": trading_date,
                        "code": code,
                        "side": "sell",
                        "shares": sell_shares,
                        "reference_price": row["open"],
                        "execution_price": sell_price,
                        "gross_notional": gross,
                        "commission": commission,
                        "stamp_tax": stamp_tax,
                        "slippage_cost": slippage,
                    }
                )

            for code in target_codes:
                row = daily_by_key.get((trading_date, code))
                if not can_trade(row, "buy"):
                    continue
                desired_shares = floor_to_lot(
                    nav_at_open * target_weight / row["open"],
                    lot_size,
                )
                buy_shares = desired_shares - positions.get(code, 0)
                if buy_shares <= 0:
                    continue
                buy_price = row["open"] * (1.0 + slippage_bps / 10_000.0)
                commission_rate = commission_bps / 10_000.0
                affordable = floor_to_lot(
                    cash / (buy_price * (1.0 + commission_rate)),
                    lot_size,
                )
                buy_shares = min(buy_shares, affordable)
                if buy_shares <= 0:
                    continue
                gross = buy_shares * buy_price
                commission = gross * commission_rate
                slippage = buy_shares * row["open"] * slippage_bps / 10_000.0
                cash -= gross + commission
                positions[code] = positions.get(code, 0) + buy_shares
                last_buy_date[code] = trading_date
                turnover_notional += gross
                total_commission += commission
                total_slippage += slippage
                trade_rows.append(
                    {
                        "date": trading_date,
                        "code": code,
                        "side": "buy",
                        "shares": buy_shares,
                        "reference_price": row["open"],
                        "execution_price": buy_price,
                        "gross_notional": gross,
                        "commission": commission,
                        "stamp_tax": 0.0,
                        "slippage_cost": slippage,
                    }
                )

        close_prices: dict[str, float] = {}
        for code in positions:
            row = daily_by_key.get((trading_date, code))
            if row is not None:
                last_close[code] = row["close"]
            if code in last_close:
                close_prices[code] = last_close[code]
        invested = position_value(positions, close_prices)
        nav = cash + invested
        daily_return = nav / previous_nav - 1.0 if previous_nav else 0.0
        previous_nav = nav
        nav_rows.append(
            {
                "date": trading_date,
                "nav": nav,
                "cash": cash,
                "invested": invested,
                "exposure": invested / nav if nav else 0.0,
                "daily_return": daily_return,
            }
        )

    average_nav = statistics.fmean(row["nav"] for row in nav_rows)
    nav_values = [initial_capital, *(row["nav"] for row in nav_rows)]
    daily_returns = [
        nav_values[index] / nav_values[index - 1] - 1.0
        for index in range(1, len(nav_values))
        if nav_values[index - 1] > 0
    ]
    peak = nav_values[0]
    max_drawdown = 0.0
    for value in nav_values:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value / peak - 1.0)
    day_count = max(len(daily_returns), 1)
    annualized_return = (
        (nav_values[-1] / nav_values[0]) ** (252.0 / day_count) - 1.0
        if nav_values[-1] > 0
        else -1.0
    )
    daily_volatility = (
        statistics.pstdev(daily_returns) if len(daily_returns) > 1 else 0.0
    )
    annualized_volatility = daily_volatility * math.sqrt(252.0)
    sharpe = (
        statistics.fmean(daily_returns) / daily_volatility * math.sqrt(252.0)
        if daily_volatility > 1e-12
        else None
    )
    calmar = (
        annualized_return / abs(max_drawdown)
        if max_drawdown < -1e-12
        else None
    )
    total_costs = total_commission + total_stamp_tax + total_slippage
    return {
        "nav_rows": nav_rows,
        "trade_rows": trade_rows,
        "metrics": {
            "total_return": nav_values[-1] / nav_values[0] - 1.0,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "average_exposure": statistics.fmean(
                row["exposure"] for row in nav_rows
            ),
            "turnover_ratio": turnover_notional / average_nav,
            "annualized_turnover": (
                turnover_notional / average_nav * 252.0 / day_count
            ),
            "total_commission": total_commission,
            "total_stamp_tax": total_stamp_tax,
            "total_slippage": total_slippage,
            "total_costs": total_costs,
            "total_costs_bps_of_average_nav": total_costs / average_nav * 10_000.0,
            "trade_count": len(trade_rows),
            "rebalance_count": len(schedule),
            "final_nav": nav_values[-1],
        },
    }


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive.")
    calendar, daily_by_key, _ = load_daily_panel(args.daily_input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = {
        "zero_cost": (0.0, 0.0, 0.0),
        "base": (
            args.commission_bps,
            args.stamp_tax_bps,
            args.slippage_bps,
        ),
        "high_cost": (
            args.commission_bps,
            args.stamp_tax_bps,
            args.slippage_bps * 2.0,
        ),
    }
    summary: dict[str, Any] = {
        "created_on": date.today().isoformat(),
        "scope": (
            "Validation-period portfolio diagnostic only; the test and final "
            "out-of-sample splits are not read."
        ),
        "execution": (
            "Rank at signal-date close, execute at the next market open, hold "
            "for the target horizon, and rebalance every holding-period number "
            "of signal dates."
        ),
        "parameters": {
            "initial_capital": args.initial_capital,
            "top_k": args.top_k,
            "lot_size": args.lot_size,
        },
        "scenarios": {},
    }
    nav_output: list[dict[str, Any]] = []
    trade_output: list[dict[str, Any]] = []
    predictions_by_horizon = {
        horizon: load_predictions(
            args.predictions_dir / f"validation_predictions_{horizon}d.csv"
        )[1]
        for horizon in HORIZONS
    }

    for scenario_name, (
        commission_bps,
        stamp_tax_bps,
        slippage_bps,
    ) in scenarios.items():
        scenario_summary: dict[str, Any] = {
            "commission_bps": commission_bps,
            "stamp_tax_bps": stamp_tax_bps,
            "slippage_bps": slippage_bps,
            "horizons": {},
        }
        for horizon in HORIZONS:
            prediction_path = (
                args.predictions_dir / f"validation_predictions_{horizon}d.csv"
            )
            predictions = predictions_by_horizon[horizon]
            horizon_summary: dict[str, Any] = {
                "prediction_file": str(prediction_path),
                "holding_days": horizon,
                "models": {},
            }
            model_specs = [
                (field, field.removeprefix("pred_"), "top_k")
                for field in MODEL_FIELDS
            ]
            model_specs.append((None, "universe_equal_weight", "equal_weight"))
            for score_field, model_name, mode in model_specs:
                schedule, signal_start, signal_end = build_schedule(
                    predictions,
                    calendar,
                    horizon,
                    args.top_k,
                    score_field,
                    mode,
                )
                result = run_portfolio(
                    schedule,
                    calendar,
                    daily_by_key,
                    horizon,
                    args.initial_capital,
                    commission_bps,
                    stamp_tax_bps,
                    slippage_bps,
                    args.lot_size,
                )
                horizon_summary["models"][model_name] = {
                    "signal_start_date": signal_start,
                    "signal_end_date": signal_end,
                    "execution_start_date": schedule[0][0],
                    "execution_end_date": schedule[-1][0],
                    **result["metrics"],
                }
                for row in result["nav_rows"]:
                    nav_output.append(
                        {
                            "scenario": scenario_name,
                            "horizon": horizon,
                            "model": model_name,
                            **row,
                        }
                    )
                for row in result["trade_rows"]:
                    trade_output.append(
                        {
                            "scenario": scenario_name,
                            "horizon": horizon,
                            "model": model_name,
                            **row,
                        }
                    )
            scenario_summary["horizons"][str(horizon)] = horizon_summary
        summary["scenarios"][scenario_name] = scenario_summary

    summary_path = args.output_dir / "validation_backtest_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(
        args.output_dir / "validation_backtest_nav.csv",
        [
            "scenario",
            "horizon",
            "model",
            "date",
            "nav",
            "cash",
            "invested",
            "exposure",
            "daily_return",
        ],
        nav_output,
    )
    write_csv(
        args.output_dir / "validation_backtest_trades.csv",
        [
            "scenario",
            "horizon",
            "model",
            "date",
            "code",
            "side",
            "shares",
            "reference_price",
            "execution_price",
            "gross_notional",
            "commission",
            "stamp_tax",
            "slippage_cost",
        ],
        trade_output,
    )
    print(f"Summary: {summary_path}")
    for scenario_name, scenario_result in summary["scenarios"].items():
        print(f"{scenario_name} (slippage={scenario_result['slippage_bps']:.1f} bps):")
        for horizon, horizon_result in scenario_result["horizons"].items():
            print(f"  {horizon}d:")
            for model_name, metrics in horizon_result["models"].items():
                sharpe = metrics["sharpe_ratio"]
                print(
                    f"    {model_name}: return={metrics['total_return']:.4%}, "
                    f"sharpe={'n/a' if sharpe is None else f'{sharpe:.3f}'}, "
                    f"mdd={metrics['max_drawdown']:.4%}, "
                    f"costs={metrics['total_costs_bps_of_average_nav']:.2f} bps"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
