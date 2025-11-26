"""Unified equity builder for trade logs + OHLC candles.

The function below normalizes candle/trade columns, builds a minute-level
timeline around the trades, and emits a DataFrame with consistent fields:
`action_price`, `deal_price`, `current_pos`, `prev_position`, `prev_action_price`,
`delta_price`, `comis_count`, `Pnl`, `Equity`, `bot_equity`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from .csv_transformer_profit import load_transform_csv

__all__ = ["build_equity"]


def _ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.floor("min")


def _find_datetime_column(df: pd.DataFrame) -> str:
    for candidate in ("DATE_TIME", "date_time"):
        if candidate in df.columns:
            return candidate
    options = [col for col in df.columns if "date" in col.lower()]
    if not options:
        raise KeyError("DataFrame must contain a datetime column (e.g. 'date_time').")
    return options[0]


def _load_candles(path: Union[str, Path], encoding: str) -> pd.DataFrame:
    candles = pd.read_csv(Path(path), encoding=encoding)
    dt_col = _find_datetime_column(candles)
    candles = candles.copy()
    candles["date_time"] = _ensure_datetime(candles[dt_col])
    candles = candles.sort_values("date_time").reset_index(drop=True)
    return candles


def _prepare_trades(raw: pd.DataFrame) -> pd.DataFrame:
    trades = raw.copy()
    trades["_order"] = trades.index
    dt_col = _find_datetime_column(trades)
    trades["date_time"] = _ensure_datetime(trades[dt_col])
    trades = trades.sort_values(["date_time", "_order"]).reset_index(drop=True)
    trades["price"] = pd.to_numeric(trades.get("price"), errors="coerce").round(2)
    trades["current_pos"] = (
        pd.to_numeric(trades.get("current_pos"), errors="coerce").fillna(0).astype(int)
    )
    profit_source = trades["position_profit"] if "position_profit" in trades else 0.0
    trades["position_profit"] = pd.to_numeric(profit_source, errors="coerce").fillna(0.0)
    return trades


def _build_timeline(trades: pd.DataFrame, time_margin: pd.Timedelta) -> pd.DataFrame:
    valid_times = trades["date_time"].dropna()
    if valid_times.empty:
        return pd.DataFrame(columns=["date_time"])
    start = valid_times.min() - time_margin
    end = valid_times.max() + time_margin
    timeline = pd.date_range(start, end, freq="1min")
    return pd.DataFrame({"date_time": timeline})


def build_equity(
    candles_csv: Union[str, Path],
    trades_csv: Union[str, Path],
    *,
    candles_encoding: str = "utf-8",
    time_margin: Union[str, pd.Timedelta] = "5min",
    comis: float = 0.0,
) -> pd.DataFrame:
    """Build equity using unified logic (ex v1+v2).

    Args:
        candles_csv: path to candles CSV.
        trades_csv: path to trade log CSV.
        candles_encoding: encoding used for candles file.
        time_margin: margin added around the trades when building the minute timeline.
        comis: commission per trade step.
    """

    trades_raw = load_transform_csv(trades_csv)
    if trades_raw.empty:
        return pd.DataFrame()
    trades = _prepare_trades(trades_raw)

    candles = _load_candles(candles_csv, candles_encoding)
    if candles["date_time"].isna().all():
        raise ValueError("Candles file does not contain valid timestamps.")

    margin = pd.Timedelta(time_margin)
    timeline = _build_timeline(trades, margin)
    if timeline.empty:
        return pd.DataFrame()

    frame_with_candles = timeline.merge(candles, on="date_time", how="left")
    merged = frame_with_candles.merge(
        trades,
        on="date_time",
        how="left",
        suffixes=("", "_trade"),
    )

    price_exists = merged["price"].notna() if "price" in merged else pd.Series(False, index=merged.index)
    pos_exists = merged["current_pos"].notna() if "current_pos" in merged else pd.Series(False, index=merged.index)
    profit_exists = merged["position_profit"].notna() if "position_profit" in merged else pd.Series(False, index=merged.index)
    candle_exists = merged["CLOSE"].notna() if "CLOSE" in merged else pd.Series(False, index=merged.index)
    mask_empty = ~(price_exists | pos_exists | profit_exists | candle_exists)
    merged = merged.loc[~mask_empty].reset_index(drop=True)

    merged = merged.sort_values(["date_time", "_order"], kind="mergesort", na_position="last").reset_index(
        drop=True
    )

    merged["price_from_log"] = pd.to_numeric(merged.get("price"), errors="coerce")
    merged["current_pos"] = (
        pd.to_numeric(merged.get("current_pos"), errors="coerce").ffill().fillna(0).astype(int)
    )
    merged["position_profit"] = pd.to_numeric(merged.get("position_profit"), errors="coerce").fillna(0.0)
    merged["prev_position"] = merged["current_pos"].shift(1).fillna(0).astype(int)

    merged["action_price"] = merged["price_from_log"]
    if "CLOSE" in merged:
        merged["action_price"] = merged["action_price"].fillna(
            pd.to_numeric(merged["CLOSE"], errors="coerce")
        )
    if "HIGH" in merged and "LOW" in merged:
        mask_nan = merged["action_price"].isna()
        merged.loc[mask_nan & (merged["prev_position"] < 0), "action_price"] = pd.to_numeric(
            merged.loc[mask_nan, "HIGH"], errors="coerce"
        )
        merged.loc[mask_nan & (merged["prev_position"] >= 0), "action_price"] = pd.to_numeric(
            merged.loc[mask_nan, "LOW"], errors="coerce"
        )
    merged["action_price"] = merged["action_price"].ffill()

    merged["deal_price"] = merged["action_price"].where(merged["current_pos"] != merged["prev_position"])
    merged["prev_action_price"] = merged["action_price"].shift(1)
    if not merged["action_price"].empty:
        merged.loc[0, "prev_action_price"] = merged.loc[0, "action_price"]
    merged["delta_price"] = merged["action_price"] - merged["prev_action_price"]

    step = pd.to_numeric(merged.get("1_step_price"), errors="coerce").ffill().fillna(1.0)
    merged["comis_count"] = (merged["current_pos"] - merged["prev_position"]).abs() * comis
    merged["Pnl"] = merged["prev_position"] * merged["delta_price"] * step - merged["comis_count"]
    merged["Equity"] = merged["Pnl"].fillna(0.0).cumsum()
    merged["bot_equity"] = merged["position_profit"].cumsum()

    return merged.drop(columns="_order", errors="ignore")
