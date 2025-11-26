"""Сервис для загрузки и кеширования OHLC минуток через moexalgo."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
TimeLike = Union[str, pd.Timestamp]

REQUIRED_COLUMNS = [
    "DATE_TIME",
    "OPEN",
    "HIGH",
    "LOW",
    "CLOSE",
    "VOL",
    "TICKER",
    "1_step_price",
]


def _to_minute(ts: TimeLike) -> pd.Timestamp:
    return pd.to_datetime(ts, errors="coerce").tz_localize(None).floor("min")


def _enforce_schema(df: pd.DataFrame, *, ticker: Optional[str] = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({col: pd.Series(dtype="float64") for col in REQUIRED_COLUMNS})

    out = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out["DATE_TIME"] = pd.to_datetime(out["DATE_TIME"], errors="coerce").dt.floor("min")
    for col in ("OPEN", "HIGH", "LOW", "CLOSE"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["VOL"] = pd.to_numeric(out["VOL"], errors="coerce").fillna(0).astype(int)
    out["TICKER"] = (
        out["TICKER"].astype(str).str.strip() if "TICKER" in out.columns else ticker or ""
    )
    out["1_step_price"] = pd.to_numeric(out["1_step_price"], errors="coerce")

    out = out.dropna(subset=["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE"])
    if ticker:
        out["TICKER"] = ticker
    out = out.drop_duplicates(subset=["TICKER", "DATE_TIME"], keep="last")
    out = out.sort_values(["TICKER", "DATE_TIME"]).reset_index(drop=True)
    return out[REQUIRED_COLUMNS]


def _load_cached(path: Path, *, ticker: Optional[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    try:
        df = pd.read_csv(path)
    except Exception:
        logger.warning("Не удалось прочитать кэш OHLC: %s", path)
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    return _enforce_schema(df, ticker=ticker)


def _save_cached(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["DATE_TIME"] = out["DATE_TIME"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)


def _missing_minutes(existing: pd.DataFrame, required: Iterable[pd.Timestamp]) -> Tuple[pd.Timestamp, pd.Timestamp] | None:
    req_index = pd.DatetimeIndex(required)
    if req_index.empty:
        return None
    if existing.empty or "DATE_TIME" not in existing.columns:
        return req_index.min(), req_index.max()
    have = pd.DatetimeIndex(pd.to_datetime(existing["DATE_TIME"], errors="coerce").dropna().dt.floor("min").unique())
    diff = req_index.difference(have)
    if diff.empty:
        return None
    return diff.min(), diff.max()


def fetch_ohlc(
    ticker: str,
    start: TimeLike,
    end: TimeLike,
    timeframe: str = "1m",
    *,
    board: Optional[str] = None,
) -> pd.DataFrame:
    """
    Загружает минутные свечи через moexalgo.Ticker.candles.
    """
    if timeframe != "1m":
        raise ValueError("Поддерживается только timeframe='1m'")
    try:
        from moexalgo import Ticker  # type: ignore
    except Exception as exc:  # pragma: no cover - опциональная зависимость
        raise ImportError("Не установлен moexalgo; установите пакет для загрузки OHLC") from exc

    start_ts = _to_minute(start)
    end_ts = _to_minute(end)
    if pd.isna(start_ts) or pd.isna(end_ts):
        raise ValueError("Некорректные даты start/end")
    if start_ts > end_ts:
        raise ValueError("start не может быть позже end")

    instrument = Ticker(ticker, board) if board else Ticker(ticker)
    raw = instrument.candles(start=start_ts.date().isoformat(), end=end_ts.date().isoformat(), period=1)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    frame = raw.copy()
    frame["DATE_TIME"] = pd.to_datetime(frame["begin"], errors="coerce").dt.tz_localize(None).dt.floor("min")
    mask = (frame["DATE_TIME"] >= start_ts) & (frame["DATE_TIME"] <= end_ts)
    frame = frame.loc[mask]
    if frame.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = pd.DataFrame(
        {
            "DATE_TIME": frame["DATE_TIME"],
            "OPEN": frame["open"],
            "HIGH": frame["high"],
            "LOW": frame["low"],
            "CLOSE": frame["close"],
            "VOL": pd.to_numeric(frame["volume"], errors="coerce").fillna(0).astype(int),
            "TICKER": ticker,
            "1_step_price": pd.NA,
        }
    )
    return _enforce_schema(df, ticker=ticker)


def get_ohlc_cached(
    ticker: str,
    start: TimeLike,
    end: TimeLike,
    timeframe: str = "1m",
    *,
    cache_dir: PathLike = Path("OHCL"),
    board: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Возвращает OHLC в запрошенном диапазоне, докачивая недостающие минуты и сохраняя кэш.
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{ticker}_{timeframe}.csv"

    start_ts = _to_minute(start)
    end_ts = _to_minute(end)
    required = pd.date_range(start_ts, end_ts, freq="1min")

    existing = _load_cached(cache_path, ticker=ticker)
    missing = _missing_minutes(existing, required) if not force_refresh else (start_ts, end_ts)

    if missing is None:
        logger.info("OHLC cache hit for %s: %s .. %s", ticker, start_ts, end_ts)
        result = existing
    else:
        miss_start, miss_end = missing
        logger.info(
            "OHLC cache miss for %s: докачка %s .. %s (запрошено %s .. %s)",
            ticker,
            miss_start,
            miss_end,
            start_ts,
            end_ts,
        )
        fetched = fetch_ohlc(ticker, miss_start, miss_end, timeframe=timeframe, board=board)
        merged = pd.concat([existing, fetched], ignore_index=True)
        result = _enforce_schema(merged, ticker=ticker)
        _save_cached(result, cache_path)

    # Возвращаем только требуемое окно.
    window = result[
        (result["DATE_TIME"] >= start_ts) & (result["DATE_TIME"] <= end_ts)
    ].copy()
    return _enforce_schema(window, ticker=ticker)


__all__ = ["fetch_ohlc", "get_ohlc_cached"]
