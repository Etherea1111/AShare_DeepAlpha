#!/usr/bin/env python3
"""Download daily A-share data for the J66 monetary and financial services industry."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import ExitStack
from datetime import date
from pathlib import Path

import baostock as bs
import pandas as pd


DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = "2024-12-31"
INDUSTRY_CODE = "J66"
INDUSTRY_NAME = "货币金融服务"
ADJUSTMENT_FLAGS = {
    "backward": "1",
    "forward": "2",
    "none": "3",
}
FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "adjustflag,turn,tradestatus,pctChg,isST"
)


class ExceptionCapture:
    """Capture ordinary exceptions raised inside a with block."""

    def __init__(self) -> None:
        self.error: Exception | None = None

    def __enter__(self) -> "ExceptionCapture":
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        _traceback: object,
    ) -> bool:
        if isinstance(exc_value, Exception):
            self.error = exc_value
            return True
        return False


# 解析行情下载的日期范围和输出文件路径。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download daily data for the J66 monetary and financial services industry."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument(
        "--adjustment",
        choices=tuple(ADJUSTMENT_FLAGS),
        default="none",
        help="Price adjustment: backward=后复权, forward=前复权, none=不复权.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/j66_money_finance_daily_2018_2024.csv"),
    )
    parser.add_argument(
        "--constituents-output",
        type=Path,
        default=Path("data/j66_money_finance_constituents.csv"),
    )
    return parser.parse_args()


# 读取查询结果中的全部记录并转换为 DataFrame。
def query_all_rows(result: object) -> pd.DataFrame:
    rows: list[list[str]] = []
    while result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=result.fields)


# 查询并筛选指定行业的当前成分股。
def get_constituents() -> pd.DataFrame:
    result = bs.query_stock_industry()
    if result.error_code != "0":
        raise RuntimeError(
            f"Failed to query industry classification: "
            f"{result.error_code} {result.error_msg}"
        )

    industry_df = query_all_rows(result)
    constituents = industry_df[
        industry_df["industry"].fillna("").str.startswith(INDUSTRY_CODE)
    ].copy()
    constituents["industry_code"] = INDUSTRY_CODE
    constituents["industry_name"] = INDUSTRY_NAME
    constituents["classification"] = "证监会行业分类"
    return constituents[
        [
            "code",
            "code_name",
            "industry_code",
            "industry_name",
            "classification",
            "updateDate",
        ]
    ].sort_values("code")


# 查询单只证券的日行情，并在失败时按次数重试。
def query_daily_data(
    code: str,
    start_date: str,
    end_date: str,
    adjustflag: str,
    retries: int = 3,
) -> pd.DataFrame:
    for attempt in range(1, retries + 1):
        result = bs.query_history_k_data_plus(
            code,
            FIELDS,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag=adjustflag,
        )
        if result.error_code == "0":
            return query_all_rows(result)
        if attempt < retries:
            time.sleep(attempt * 2)
    raise RuntimeError(
        f"Failed to query {code}: {result.error_code} {result.error_msg}"
    )


# 转换行情数值列并补齐行业和数据来源信息。
def normalize_daily_data(
    daily_df: pd.DataFrame, constituents: pd.DataFrame
) -> pd.DataFrame:
    if daily_df.empty:
        return daily_df

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "adjustflag",
        "turn",
        "tradestatus",
        "pctChg",
        "isST",
    ]
    for column in numeric_columns:
        daily_df[column] = pd.to_numeric(daily_df[column], errors="coerce")

    names = constituents[["code", "code_name"]].drop_duplicates()
    daily_df = daily_df.merge(names, on="code", how="left", validate="many_to_one")
    daily_df["industry_code"] = INDUSTRY_CODE
    daily_df["industry_name"] = INDUSTRY_NAME
    daily_df["data_source"] = "Baostock"
    daily_df["adjustment"] = "none"

    return daily_df[
        [
            "date",
            "code",
            "code_name",
            "industry_code",
            "industry_name",
            "open",
            "high",
            "low",
            "close",
            "preclose",
            "volume",
            "amount",
            "turn",
            "pctChg",
            "tradestatus",
            "isST",
            "adjustflag",
            "adjustment",
            "data_source",
        ]
    ].sort_values(["date", "code"])


# 下载已登录数据源中的成分股、日行情并保存结果和元数据。
def download_logged_in(args: argparse.Namespace) -> int:
    constituents = get_constituents()
    constituents.to_csv(
        args.constituents_output,
        index=False,
        encoding="utf-8-sig",
    )

    all_daily_data: list[pd.DataFrame] = []
    failed_codes: list[dict[str, str]] = []
    total = len(constituents)
    for index, row in enumerate(constituents.itertuples(index=False), start=1):
        print(f"[{index}/{total}] {row.code} {row.code_name}", flush=True)
        capture = ExceptionCapture()
        with capture:
            daily_df = query_daily_data(
                row.code,
                args.start_date,
                args.end_date,
                ADJUSTMENT_FLAGS[args.adjustment],
            )
        if capture.error is not None:
            failed_codes.append({"code": row.code, "error": str(capture.error)})
            print(f"  failed: {capture.error}", file=sys.stderr, flush=True)
            continue
        if not daily_df.empty:
            all_daily_data.append(daily_df)
            print(f"  rows: {len(daily_df)}", flush=True)
        else:
            print("  rows: 0", flush=True)

    if not all_daily_data:
        raise RuntimeError("No daily data was downloaded.")

    daily_data = normalize_daily_data(
        pd.concat(all_daily_data, ignore_index=True), constituents
    )
    daily_data["adjustment"] = args.adjustment
    daily_data.to_csv(args.output, index=False, encoding="utf-8-sig")

    metadata = {
        "industry_code": INDUSTRY_CODE,
        "industry_name": INDUSTRY_NAME,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "adjustment": args.adjustment,
        "adjustflag": ADJUSTMENT_FLAGS[args.adjustment],
        "data_source": "Baostock",
        "download_date": date.today().isoformat(),
        "constituent_count": int(len(constituents)),
        "downloaded_code_count": int(daily_data["code"].nunique()),
        "row_count": int(len(daily_data)),
        "failed_codes": failed_codes,
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Saved {len(daily_data):,} rows for "
        f"{daily_data['code'].nunique()} symbols to {args.output}"
    )
    if failed_codes:
        print(f"Failed symbols: {len(failed_codes)}", file=sys.stderr)
    return 0


# 登录数据源，下载成分股及其日行情并保存结果和元数据。
def main() -> int:
    args = parse_args()
    if args.start_date > args.end_date:
        raise ValueError("start-date must not be later than end-date")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.constituents_output.parent.mkdir(parents=True, exist_ok=True)

    login_result = bs.login()
    if login_result.error_code != "0":
        raise RuntimeError(
            f"Baostock login failed: {login_result.error_code} "
            f"{login_result.error_msg}"
        )

    with ExitStack() as stack:
        stack.callback(bs.logout)
        return download_logged_in(args)


if __name__ == "__main__":
    raise SystemExit(main())
