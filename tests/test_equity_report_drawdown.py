from __future__ import annotations

import pandas as pd
import pytest

from src.data.equity_report import _longest_drawdown_window, _max_drawdown


def _timestamps(count: int) -> pd.Series:
    return pd.Series(pd.date_range("2024-01-01 09:00:00", periods=count, freq="1h"))


def test_max_drawdown_detects_peak_to_trough():
    equity = pd.Series([100, 120, 90, 110, 70, 130], dtype="float64")
    ts = _timestamps(len(equity))

    drawdown, start, end = _max_drawdown(equity, ts)

    assert drawdown == pytest.approx(50)
    assert start == ts.iloc[1]
    assert end == ts.iloc[4]


def test_max_drawdown_monotonic_growth_has_no_drawdown():
    equity = pd.Series([10, 20, 50, 70], dtype="float64")
    ts = _timestamps(len(equity))

    drawdown, start, end = _max_drawdown(equity, ts)

    assert drawdown == 0
    assert start is None
    assert end is None


def test_max_drawdown_single_point():
    equity = pd.Series([42.0], dtype="float64")
    ts = _timestamps(len(equity))

    drawdown, start, end = _max_drawdown(equity, ts)

    assert drawdown == 0
    assert start is None
    assert end is None


def test_longest_drawdown_identifies_longest_window():
    equity = pd.Series([100, 95, 80, 90, 70, 60, 120], dtype="float64")
    ts = _timestamps(len(equity))

    minutes, start, end = _longest_drawdown_window(equity, ts)

    assert minutes == pytest.approx((ts.iloc[6] - ts.iloc[1]).total_seconds() / 60)
    assert start == ts.iloc[1]
    assert end == ts.iloc[6]


def test_longest_drawdown_monotonic_growth_returns_zero():
    equity = pd.Series([1, 2, 3, 4], dtype="float64")
    ts = _timestamps(len(equity))

    minutes, start, end = _longest_drawdown_window(equity, ts)

    assert minutes == 0
    assert start is None
    assert end is None


def test_longest_drawdown_monotonic_decline_extends_to_last_bar():
    equity = pd.Series([100, 90, 80], dtype="float64")
    ts = _timestamps(len(equity))

    minutes, start, end = _longest_drawdown_window(equity, ts)

    assert minutes == pytest.approx((ts.iloc[-1] - ts.iloc[1]).total_seconds() / 60)
    assert start == ts.iloc[1]
    assert end == ts.iloc[-1]
