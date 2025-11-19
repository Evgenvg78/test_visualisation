"""Новая версия сборщика equity на основе minute timeline + лог."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd

from .csv_transformer_profit import load_transform_csv

__all__ = ["build_equity_v2"]


def _ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.floor("min")


def _find_datetime_column(candles: pd.DataFrame) -> str:
    for candidate in ("DATE_TIME", "date_time"):
        if candidate in candles.columns:
            return candidate
    possibilities = [col for col in candles.columns if "date" in col.lower()]
    if not possibilities:
        raise KeyError("В файле свечей не найден столбец даты (ожидалось 'DATE_TIME')")
    return possibilities[0]


def _load_candles(path: Union[str, Path], encoding: str) -> pd.DataFrame:
    candles = pd.read_csv(Path(path), encoding=encoding)
    dt_col = _find_datetime_column(candles)
    candles = candles.copy()
    candles["date_time"] = _ensure_datetime(candles[dt_col])
    return candles.sort_values("date_time").reset_index(drop=True)


def _build_timeline(trades: pd.DataFrame, margin: pd.Timedelta) -> pd.DataFrame:
    valid_times = trades["date_time"].dropna()
    if valid_times.empty:
        return pd.DataFrame(columns=["date_time"])
    start = valid_times.min() - margin
    end = valid_times.max() + margin
    timeline = pd.date_range(start, end, freq="1min")
    return pd.DataFrame({"date_time": timeline})


def build_equity_v2(
    candles_csv: Union[str, Path],
    trades_csv: Union[str, Path],
    *,
    candles_encoding: str = "utf-8",
    time_margin: Union[str, pd.Timedelta] = "5min",
    comis: float = 0.0,
) -> pd.DataFrame:
    """Собрать equity на основе timeline ±5 минут и полного лога."""

    trades = load_transform_csv(trades_csv)
    if trades.empty:
        return pd.DataFrame()

    trades = trades.copy()
    trades["_order"] = trades.index
    trades["date_time"] = _ensure_datetime(trades["date_time"])
    trades = trades.sort_values(["date_time", "_order"]).reset_index(drop=True)
    if "position_profit" in trades.columns:
        trades["position_profit"] = pd.to_numeric(trades["position_profit"], errors="coerce").fillna(0.0)
    else:
        trades["position_profit"] = 0.0
    if "price" in trades.columns:
        trades["price"] = pd.to_numeric(trades["price"], errors="coerce").round(2)
    if "current_pos" in trades.columns:
        trades["current_pos"] = (
            pd.to_numeric(trades["current_pos"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    else:
        trades["current_pos"] = 0

    margin = pd.Timedelta(time_margin)
    frame = _build_timeline(trades, margin)
    if frame.empty:
        return pd.DataFrame()

    candles = _load_candles(candles_csv, candles_encoding)
    if candles["date_time"].isna().all():
        raise ValueError("Не удалось определить таймстемпы свечей")

    frame_with_candles = frame.merge(candles, on="date_time", how="left")
    merged = frame_with_candles.merge(
        trades,
        on="date_time",
        how="left",
        suffixes=("", "_trade"),
    )

    price_exists = merged["price"].notna()
    pos_exists = merged["current_pos"].notna()
    profit_exists = merged["position_profit"].notna()
    candle_exists = merged["CLOSE"].notna() if "CLOSE" in merged else pd.Series(False, index=merged.index)
    mask_empty = ~(price_exists | pos_exists | profit_exists | candle_exists)
    merged = merged.loc[~mask_empty].reset_index(drop=True)

    merged = merged.sort_values(["date_time", "_order"], kind="mergesort", na_position="last").reset_index(
        drop=True
    )

    merged["price_from_log"] = merged["price"]
    merged["final_price"] = merged["price_from_log"].fillna(merged.get("CLOSE"))
    merged["price_source"] = merged["price_from_log"].notna()

    merged["current_pos"] = merged["current_pos"].ffill().fillna(0).astype(int)
    merged["position_profit"] = merged["position_profit"].fillna(0.0)
    merged["prev_position"] = merged["current_pos"].shift(1).fillna(0).astype(int)
    merged["action_price"] = merged["final_price"]

    position_changed = merged["current_pos"] != merged["prev_position"]
    merged["deal_price"] = merged["action_price"].where(position_changed)
    merged["prev_action_price"] = merged["action_price"].shift(1)
    if not merged["action_price"].empty:
        merged.loc[0, "prev_action_price"] = merged.loc[0, "action_price"]
    merged["delta_price"] = merged["action_price"] - merged["prev_action_price"]

    step = merged["1_step_price"].ffill().fillna(1.0)
    merged["comis_count"] = (merged["current_pos"] - merged["prev_position"]).abs() * comis
    merged["Pnl"] = merged["prev_position"] * merged["delta_price"] * step - merged["comis_count"]
    merged["Equity"] = merged["Pnl"].fillna(0.0).cumsum()
    merged["bot_equity"] = merged["position_profit"].cumsum()

    return merged.drop(columns="_order", errors="ignore")
