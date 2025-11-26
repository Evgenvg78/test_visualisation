from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.equity_combination import LogEquitySnapshot, combine_equity_logs


def _snapshot(name: str, values, *, go_requirement: float, commission: float) -> LogEquitySnapshot:
    timestamps = pd.date_range("2024-02-01 10:00:00", periods=len(values), freq="1h")
    equity = pd.DataFrame({"date_time": timestamps, "Equity": values})
    series = equity.set_index("date_time")["Equity"].astype(float)
    report = SimpleNamespace(go_requirement=go_requirement, total_commission=commission)
    return LogEquitySnapshot(
        log_path=Path(f"{name}.csv"),
        ticker=name,
        equity=equity,
        series=series,
        report=report,
    )


def test_combine_equity_logs_from_dataframes():
    base_ts = pd.date_range("2024-01-01 00:00:00", periods=3, freq="1h")
    equity_a = pd.DataFrame({"date_time": base_ts, "Equity": [10.0, 15.0, 20.0]})
    equity_b = pd.DataFrame({"date_time": base_ts[:2], "Equity": [5.0, -5.0]})

    result = combine_equity_logs(
        [("Alpha", equity_a), ("Beta", equity_b)],
        timezone=None,
        build_plot=False,
    )

    combined = result.combined
    assert {"date_time", "Equity", "Alpha", "Beta"}.issubset(combined.columns)

    alpha = combined["Alpha"].tolist()
    beta = combined["Beta"].tolist()
    total = combined["Equity"].tolist()
    assert alpha == [10.0, 15.0, 20.0]
    # Beta series should forward-fill the last known value for the missing timestamp.
    assert beta == [5.0, -5.0, -5.0]
    assert total == [15.0, 10.0, 15.0]


def test_combine_equity_logs_aggregates_metrics_with_reports():
    snapshots = [
        _snapshot("First", [0.0, 100.0, 50.0], go_requirement=1000.0, commission=7.5),
        _snapshot("Second", [0.0, -20.0, 40.0], go_requirement=500.0, commission=2.5),
    ]

    result = combine_equity_logs(snapshots)

    metrics = result.metrics
    assert metrics.go_requirement == pytest.approx(1500.0)
    assert metrics.final_equity == pytest.approx(90.0)
    assert metrics.return_percent == pytest.approx(90.0 / 1500.0 * 100)
    assert metrics.drawdown_currency == pytest.approx(0.0)
    assert metrics.drawdown_percent_of_go == pytest.approx(0.0)
    assert metrics.total_commission == pytest.approx(10.0)


def test_combine_equity_logs_requires_data():
    with pytest.raises(ValueError):
        combine_equity_logs([])
