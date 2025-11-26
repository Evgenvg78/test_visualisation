from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.services import ohlc


def _mock_candles_frame(timestamps: list[pd.Timestamp], *, ticker: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "begin": pd.Series(timestamps),
            "open": [100 + idx for idx, _ in enumerate(timestamps)],
            "high": [101 + idx for idx, _ in enumerate(timestamps)],
            "low": [99 + idx for idx, _ in enumerate(timestamps)],
            "close": [100 + idx for idx, _ in enumerate(timestamps)],
            "volume": [10] * len(timestamps),
        }
    )


def _ohlc_frame(timestamps: list[pd.Timestamp], *, ticker: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DATE_TIME": timestamps,
            "OPEN": [100 + idx for idx, _ in enumerate(timestamps)],
            "HIGH": [101 + idx for idx, _ in enumerate(timestamps)],
            "LOW": [99 + idx for idx, _ in enumerate(timestamps)],
            "CLOSE": [100 + idx for idx, _ in enumerate(timestamps)],
            "VOL": [idx + 1 for idx in range(len(timestamps))],
            "TICKER": ticker,
            "1_step_price": [1.0] * len(timestamps),
        }
    )


def test_fetch_ohlc_uses_fake_ticker(fake_ticker_class):
    start = pd.Timestamp("2024-05-01 10:00:00")
    timestamps = [
        pd.Timestamp("2024-05-01 10:00:00+00:00"),
        pd.Timestamp("2024-05-01 10:01:00+00:00"),
        pd.Timestamp("2024-05-01 10:02:00+00:00"),
    ]
    fake_ticker_class.queue_response(_mock_candles_frame(timestamps, ticker="SIL"))

    frame = ohlc.fetch_ohlc("SIL", start, start + pd.Timedelta(minutes=2))

    assert list(frame.columns) == ohlc.REQUIRED_COLUMNS
    assert frame["TICKER"].unique().tolist() == ["SIL"]
    assert frame["DATE_TIME"].min() == pd.Timestamp("2024-05-01 10:00:00")
    assert frame["DATE_TIME"].max() == pd.Timestamp("2024-05-01 10:02:00")


def test_get_ohlc_cached_hits_existing_file(tmp_path: Path, monkeypatch):
    ticker = "RI"
    start = pd.Timestamp("2024-04-01 09:00:00")
    timestamps = [start + pd.Timedelta(minutes=i) for i in range(3)]
    cache_dir = tmp_path / "ohlc"
    cache_dir.mkdir()
    cache_path = cache_dir / f"{ticker}_1m.csv"
    _ohlc_frame(timestamps, ticker=ticker).assign(
        DATE_TIME=lambda df: df["DATE_TIME"].dt.strftime("%Y-%m-%d %H:%M:%S")
    ).to_csv(cache_path, index=False)

    def _fail_fetch(*args, **kwargs):
        raise AssertionError("fetch_ohlc should not be called when cache is warm")

    monkeypatch.setattr(ohlc, "fetch_ohlc", _fail_fetch)

    result = ohlc.get_ohlc_cached(
        ticker,
        start,
        start + pd.Timedelta(minutes=2),
        cache_dir=cache_dir,
    )

    assert len(result) == 3
    assert result["DATE_TIME"].tolist() == timestamps


def test_get_ohlc_cached_downloads_missing_range(tmp_path: Path, monkeypatch):
    ticker = "BR"
    start = pd.Timestamp("2024-04-02 10:00:00")
    required = [start + pd.Timedelta(minutes=i) for i in range(4)]
    cache_dir = tmp_path / "ohlc"
    cache_dir.mkdir()
    cache_path = cache_dir / f"{ticker}_1m.csv"
    existing = _ohlc_frame(required[:2], ticker=ticker)
    existing.assign(DATE_TIME=lambda df: df["DATE_TIME"].dt.strftime("%Y-%m-%d %H:%M:%S")).to_csv(
        cache_path, index=False
    )

    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def _fake_fetch(ticker_arg, start_arg, end_arg, timeframe="1m", **kwargs):
        calls.append((start_arg, end_arg))
        span = pd.date_range(start_arg, end_arg, freq="1min")
        return _ohlc_frame(list(span), ticker=ticker_arg)

    monkeypatch.setattr(ohlc, "fetch_ohlc", _fake_fetch)

    result = ohlc.get_ohlc_cached(
        ticker,
        start,
        start + pd.Timedelta(minutes=3),
        cache_dir=cache_dir,
    )

    assert calls == [(required[2], required[3])]
    assert result["DATE_TIME"].tolist() == required
    # Cache should now include all required timestamps.
    reloaded = pd.read_csv(cache_path)
    assert reloaded["DATE_TIME"].tolist() == [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in required]
