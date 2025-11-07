from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
from moexalgo import Ticker

from .candle_loader import DEFAULT_STEPS_PATH, _load_step_price

__all__ = [
    "load_last_test_trades",
    "build_download_plan",
    "download_moex_ohlc",
]

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

DEFAULT_DATETIME_COLUMN = "Дата и время"
DEFAULT_TICKER_COLUMN = "Код инструмента"
DEFAULT_LOG_ENCODING = "cp1251"
DEFAULT_BOARD_ID: Optional[str] = None
DEFAULT_MARGIN_MINUTES = 1
DEFAULT_BUFFER_MINUTES = 3
CHUNK_SIZE_MINUTES = 9500
CHUNK_OVERLAP_MINUTES = 60

EXPECTED_COLUMNS = [
    "DATE_TIME",
    "OPEN",
    "HIGH",
    "LOW",
    "CLOSE",
    "VOL",
    "TICKER",
    "1_step_price",
]

_COLUMN_DTYPES = {
    "DATE_TIME": "datetime64[ns]",
    "OPEN": "float64",
    "HIGH": "float64",
    "LOW": "float64",
    "CLOSE": "float64",
    "VOL": "int64",
    "TICKER": "object",
    "1_step_price": "float64",
}

_REFERENCE_CACHE: Dict[Path, pd.DataFrame] = {}

try:
    from .log_loader import load_log as _external_log_loader  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency
    _external_log_loader = None


def _empty_result() -> pd.DataFrame:
    """Return an empty dataframe matching the expected schema."""
    return pd.DataFrame(
        {
            "DATE_TIME": pd.Series(dtype="datetime64[ns]"),
            "OPEN": pd.Series(dtype="float64"),
            "HIGH": pd.Series(dtype="float64"),
            "LOW": pd.Series(dtype="float64"),
            "CLOSE": pd.Series(dtype="float64"),
            "VOL": pd.Series(dtype="int64"),
            "TICKER": pd.Series(dtype="object"),
            "1_step_price": pd.Series(dtype="float64"),
        }
    )


def _normalize_ticker(value: Union[str, float, int]) -> str:
    """Trim surrounding whitespace and keep ticker casing as-is."""
    text = str(value).strip()
    if not text:
        return ""
    return text


def _load_log_dataframe(
    log_path: PathLike,
    *,
    encoding: str = DEFAULT_LOG_ENCODING,
    sep: str = ";",
) -> pd.DataFrame:
    """
    Load a raw trade log using an existing helper when available.

    Falls back to plain pandas.read_csv when no specialised loader is present.
    """
    if _external_log_loader is not None:
        df = _external_log_loader(log_path)  # type: ignore[misc]
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected load_log(...) to return a pandas.DataFrame")
        return df
    return pd.read_csv(log_path, sep=sep, encoding=encoding)


def _extract_last_test(
    trades: pd.DataFrame,
    *,
    datetime_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return the subset of rows that belong to the last backtest segment.

    The accompanying Series contains parsed datetime values aligned with the subset.
    """
    if datetime_column not in trades.columns:
        raise KeyError(f"Column '{datetime_column}' is missing in the trade log")

    raw_dt = trades[datetime_column].astype(str).str.strip()
    parsed_dt = pd.to_datetime(
        raw_dt,
        format="%d.%m.%Y %H:%M:%S",
        errors="coerce",
    )
    valid_dt = parsed_dt.dropna()
    if valid_dt.empty:
        raise ValueError(
            f"Column '{datetime_column}' does not contain valid datetime values"
        )

    chunk_boundaries = valid_dt < valid_dt.shift()
    chunk_ids = chunk_boundaries.cumsum()
    last_chunk_id = int(chunk_ids.iloc[-1])
    last_indices = chunk_ids[chunk_ids == last_chunk_id].index

    last_segment = trades.loc[last_indices].copy()
    segment_datetimes = parsed_dt.loc[last_indices].dt.floor("min")
    mask = segment_datetimes.notna()
    last_segment = last_segment.loc[mask]
    return last_segment, segment_datetimes.loc[mask]


def load_last_test_trades(
    log_path: PathLike,
    *,
    datetime_column: str = DEFAULT_DATETIME_COLUMN,
    ticker_column: str = DEFAULT_TICKER_COLUMN,
    encoding: str = DEFAULT_LOG_ENCODING,
) -> pd.DataFrame:
    """
    Load the original trade log and keep only rows from the last test run.

    The resulting dataframe retains all original columns and adds:
        * ticker_original — trimmed ticker string exactly as in the log.
        * ticker — normalised ticker (upper-case) used for MOEX requests.
        * date_time — parsed timestamp rounded down to the minute.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Result CSV not found: {log_path}")

    trades = _load_log_dataframe(log_path, encoding=encoding)
    if ticker_column not in trades.columns:
        raise KeyError(f"Column '{ticker_column}' is missing in the trade log")

    last_segment, parsed_dt = _extract_last_test(
        trades,
        datetime_column=datetime_column,
    )
    last_segment = last_segment.copy()
    last_segment["ticker_original"] = (
        last_segment[ticker_column].astype(str).str.strip()
    )
    last_segment["ticker"] = last_segment["ticker_original"].map(_normalize_ticker)
    last_segment["date_time"] = parsed_dt

    last_segment = last_segment.dropna(subset=["date_time"])
    last_segment = last_segment[last_segment["ticker"] != ""]
    last_segment = last_segment.sort_index()
    return last_segment.reset_index(drop=True)


def build_download_plan(
    last_test_trades: pd.DataFrame,
    *,
    timestamp_column: str = "date_time",
    ticker_column: str = "ticker",
    margin_minutes: int = DEFAULT_MARGIN_MINUTES,
) -> pd.DataFrame:
    """
    Build a per-ticker table with time ranges that must be covered by OHLC data.
    """
    if last_test_trades.empty:
        return pd.DataFrame(columns=["ticker", "start_dt", "end_dt"])

    if timestamp_column not in last_test_trades.columns:
        raise KeyError(f"Timestamp column '{timestamp_column}' is missing")
    if ticker_column not in last_test_trades.columns:
        raise KeyError(f"Ticker column '{ticker_column}' is missing")

    timestamps = pd.to_datetime(
        last_test_trades[timestamp_column],
        errors="coerce",
    ).dt.floor("min")
    tickers = last_test_trades[ticker_column].astype(str).str.strip()

    frame = pd.DataFrame({"ticker": tickers, "date_time": timestamps})
    frame = frame.dropna(subset=["ticker", "date_time"])
    frame = frame[frame["ticker"] != ""]
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "start_dt", "end_dt"])

    grouped = frame.groupby("ticker")["date_time"]
    boundaries = grouped.agg(["min", "max"]).rename(
        columns={"min": "min_dt", "max": "max_dt"}
    )

    offset = pd.Timedelta(minutes=margin_minutes)
    boundaries["start_dt"] = (boundaries["min_dt"] - offset).dt.floor("min")
    boundaries["end_dt"] = (boundaries["max_dt"] + offset).dt.ceil("min")

    plan = boundaries.reset_index()[["ticker", "start_dt", "end_dt"]]
    return plan.sort_values(["ticker", "start_dt"]).reset_index(drop=True)


def _compute_missing_minutes(
    existing: pd.DataFrame,
    *,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> pd.DatetimeIndex:
    """Return minute timestamps that are absent within the requested range."""
    required = pd.date_range(
        start=start_dt.floor("min"),
        end=end_dt.floor("min"),
        freq="min",
    )
    if required.empty:
        return required

    if existing.empty or "DATE_TIME" not in existing.columns:
        return required

    existing_minutes = (
        pd.to_datetime(existing["DATE_TIME"], errors="coerce")
        .dropna()
        .dt.floor("min")
    )
    if existing_minutes.empty:
        return required

    return required.difference(pd.DatetimeIndex(existing_minutes.unique()))


def _extract_base_ticker(ticker: str) -> str:
    """Attempt to reduce futures contract code to its base instrument name."""
    clean = _normalize_ticker(ticker)
    if "-" in clean:
        return clean.split("-", 1)[0]
    if len(clean) >= 2 and clean[-1].isdigit() and clean[-2].isalpha():
        return clean[:-2]
    return clean


def _load_reference_table(path: Path) -> pd.DataFrame:
    """Read and cache the step-price reference."""
    resolved = path.resolve()
    cached = _REFERENCE_CACHE.get(resolved)
    if cached is not None:
        return cached
    table = pd.read_csv(resolved)
    _REFERENCE_CACHE[resolved] = table
    return table


def _extract_short_code(ticker: str) -> str:
    """
    Obtain base instrument code suitable for SECID_SHORT matching.

    Examples:
        SiZ5 -> Si
        LKZ4 -> LK
        BR-12.25 -> BR
    """
    base = _extract_base_ticker(ticker)
    if base and base[-1].isdigit():
        # Handles exotic cases when month digit not stripped above
        return base.rstrip("0123456789")
    return base


def _resolve_step_price(ticker: str, steps_path: Path) -> float:
    """Load step price value, trying both contract and base instrument naming."""
    candidates = []
    contract = _normalize_ticker(ticker)
    if contract:
        candidates.append(contract)
    base = _extract_base_ticker(contract)
    if base and base not in candidates:
        candidates.append(base)

    for name in candidates:
        try:
            return _load_step_price(name, steps_path)
        except KeyError:
            continue

    reference = _load_reference_table(steps_path)
    if "one_step_price" not in reference.columns:
        raise KeyError(
            f"Reference file {steps_path} is missing required column 'one_step_price'"
        )

    short_code = _extract_short_code(contract)
    if short_code and "SECID_SHORT" in reference.columns:
        matches = reference.loc[
            reference["SECID_SHORT"]
            .astype(str)
            .str.strip()
            .str.casefold()
            == short_code.casefold(),
            "one_step_price",
        ]
        if not matches.empty:
            return float(matches.iloc[0])

    if "SECID" in reference.columns:
        matches = reference.loc[
            reference["SECID"].astype(str).str.strip().str.casefold()
            == contract.casefold(),
            "one_step_price",
        ]
        if not matches.empty:
            return float(matches.iloc[0])

    raise KeyError(
        f"Ticker '{ticker}' is missing in step price reference {steps_path}"
    )


def _download_ticker_minutes(
    ticker: str,
    *,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    board: Optional[str] = DEFAULT_BOARD_ID,
) -> pd.DataFrame:
    """
    Request minute candles from MOEX for the supplied range (inclusive).
    """
    if start_dt > end_dt:
        raise ValueError("start_dt must not be after end_dt")

    normalized = _normalize_ticker(ticker)
    try:
        if board:
            instrument = Ticker(normalized, board)
        else:
            instrument = Ticker(normalized)
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"Failed to create MOEX ticker '{ticker}': {exc}") from exc

    chunk_size = max(int(CHUNK_SIZE_MINUTES), 1)
    overlap = max(min(int(CHUNK_OVERLAP_MINUTES), chunk_size - 1), 0)

    current_start = start_dt.floor("min")
    final_end = end_dt.ceil("min")

    collected: list[pd.DataFrame] = []

    while current_start <= final_end:
        chunk_end = min(final_end, current_start + pd.Timedelta(minutes=chunk_size - 1))
        start_date = current_start.date().isoformat()
        end_date = chunk_end.date().isoformat()

        try:
            raw = instrument.candles(start=start_date, end=end_date, period=1)
        except Exception as exc:  # pragma: no cover - network dependent
            raise RuntimeError(
                f"Failed to download candles for '{ticker}' between {start_date} and {end_date}: {exc}"
            ) from exc

        if raw is not None and not raw.empty:
            frame = raw.copy()
            frame["DATE_TIME"] = (
                pd.to_datetime(frame["begin"], errors="coerce").dt.tz_localize(None)
            )
            mask = (frame["DATE_TIME"] >= current_start) & (
                frame["DATE_TIME"] <= chunk_end
            )
            frame = frame.loc[mask]
            if not frame.empty:
                chunk_df = pd.DataFrame(
                    {
                        "DATE_TIME": frame["DATE_TIME"].dt.floor("min"),
                        "OPEN": pd.to_numeric(frame["open"], errors="coerce"),
                        "HIGH": pd.to_numeric(frame["high"], errors="coerce"),
                        "LOW": pd.to_numeric(frame["low"], errors="coerce"),
                        "CLOSE": pd.to_numeric(frame["close"], errors="coerce"),
                        "VOL": pd.to_numeric(frame["volume"], errors="coerce")
                        .fillna(0)
                        .astype(int),
                        "TICKER": normalized,
                    }
                )
                chunk_df["TICKER"] = chunk_df["TICKER"].astype(str)
                chunk_df["1_step_price"] = float("nan")
                chunk_df = chunk_df.dropna(
                    subset=["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE"]
                )
                if not chunk_df.empty:
                    collected.append(chunk_df)

        if chunk_end >= final_end:
            break

        next_start = chunk_end - pd.Timedelta(minutes=overlap)
        if next_start <= current_start:
            next_start = chunk_end + pd.Timedelta(minutes=1)
        current_start = next_start.floor("min")

    if not collected:
        return _empty_result()

    candles = pd.concat(collected, ignore_index=True)
    candles = candles.sort_values(["TICKER", "DATE_TIME"])
    candles = candles.drop_duplicates(subset=["TICKER", "DATE_TIME"], keep="last")
    return candles.reset_index(drop=True)


def _enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe has the expected columns and dtypes."""
    if df.empty:
        return _empty_result()

    data = df.copy()
    for col in EXPECTED_COLUMNS:
        if col not in data.columns:
            data[col] = pd.NA

    data = data[EXPECTED_COLUMNS]
    data["DATE_TIME"] = pd.to_datetime(data["DATE_TIME"], errors="coerce").dt.floor(
        "min"
    )
    data["TICKER"] = data["TICKER"].astype(str).str.strip()

    for col in ("OPEN", "HIGH", "LOW", "CLOSE", "1_step_price"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["VOL"] = pd.to_numeric(data["VOL"], errors="coerce").fillna(0).astype(int)

    data = data.dropna(subset=["DATE_TIME", "TICKER"])
    data = data.dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE"])
    data = data.drop_duplicates(subset=["TICKER", "DATE_TIME"], keep="last")
    return data.sort_values(["TICKER", "DATE_TIME"]).reset_index(drop=True)


def _read_existing(path: Path) -> pd.DataFrame:
    """Load previously saved OHLC data if available."""
    if not path.exists():
        return _empty_result()

    df = pd.read_csv(path)
    if df.empty:
        return _empty_result()
    return _enforce_schema(df)


def _save_result(df: pd.DataFrame, target: Path) -> None:
    """Persist candles dataframe using ISO timestamps."""
    output = df.copy()
    output["DATE_TIME"] = output["DATE_TIME"].dt.strftime("%Y-%m-%d %H:%M:%S")
    target.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(target, index=False)


def _ticker_cache_path(root: Path, ticker: str) -> Path:
    """Return the per-ticker cache file path."""
    normalized = _normalize_ticker(ticker)
    if not normalized:
        raise ValueError("Ticker name must not be empty when building cache path")
    return root / f"{normalized}_ohcl.csv"


def _combine_frames(
    frames: list[pd.DataFrame],
    *,
    steps_path: Path,
) -> pd.DataFrame:
    """Merge multiple OHLC chunks, enforce schema, and back-fill step prices."""
    valid_frames = [frame for frame in frames if not frame.empty]
    if not valid_frames:
        return _empty_result()
    combined = pd.concat(valid_frames, ignore_index=True)
    combined = _enforce_schema(combined)
    return _fill_missing_step_prices(combined, steps_path)


# ---------------------- Interval coverage (per ticker) ----------------------

def _coverage_dir(root: Path) -> Path:
    d = root / "_coverage"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _coverage_path(root: Path, ticker: str) -> Path:
    return _coverage_dir(root) / f"{_normalize_ticker(ticker)}_intervals.csv"


def _read_coverage(root: Path, ticker: str) -> pd.DataFrame:
    """Read covered intervals for a ticker. Columns: start_dt, end_dt (datetime64[ns])."""
    path = _coverage_path(root, ticker)
    if not path.exists():
        return pd.DataFrame(columns=["start_dt", "end_dt"]).astype(
            {"start_dt": "datetime64[ns]", "end_dt": "datetime64[ns]"}
        )
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["start_dt", "end_dt"]).astype(
            {"start_dt": "datetime64[ns]", "end_dt": "datetime64[ns]"}
        )
    df["start_dt"] = pd.to_datetime(df["start_dt"], errors="coerce").dt.floor("min")
    df["end_dt"] = pd.to_datetime(df["end_dt"], errors="coerce").dt.ceil("min")
    df = df.dropna(subset=["start_dt", "end_dt"]).sort_values("start_dt")
    return df.reset_index(drop=True)


def _write_coverage(root: Path, ticker: str, intervals: pd.DataFrame) -> None:
    if intervals.empty:
        # create or truncate file
        _coverage_path(root, ticker).write_text("start_dt,end_dt\n", encoding="utf-8")
        return
    out = intervals.copy()
    out["start_dt"] = out["start_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out["end_dt"] = out["end_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(_coverage_path(root, ticker), index=False)


def _merge_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping [start_dt, end_dt] rows."""
    if intervals.empty:
        return intervals
    df = intervals.sort_values("start_dt").reset_index(drop=True)
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur_start = df.loc[0, "start_dt"]
    cur_end = df.loc[0, "end_dt"]
    for i in range(1, len(df)):
        s = df.loc[i, "start_dt"]
        e = df.loc[i, "end_dt"]
        if s <= cur_end + pd.Timedelta(minutes=1):
            if e > cur_end:
                cur_end = e
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))
    return pd.DataFrame(merged, columns=["start_dt", "end_dt"]).astype(
        {"start_dt": "datetime64[ns]", "end_dt": "datetime64[ns]"}
    )


def _add_coverage(intervals: pd.DataFrame, new_start: pd.Timestamp, new_end: pd.Timestamp) -> pd.DataFrame:
    add = pd.DataFrame({"start_dt": [new_start.floor("min")], "end_dt": [new_end.ceil("min")]})
    if intervals.empty:
        return add.astype({"start_dt": "datetime64[ns]", "end_dt": "datetime64[ns]"})
    return _merge_intervals(pd.concat([intervals, add], ignore_index=True))


def _subtract_coverage(required_start: pd.Timestamp, required_end: pd.Timestamp, intervals: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return uncovered sub-intervals of [required_start, required_end] given covered intervals."""
    start = required_start.floor("min")
    end = required_end.ceil("min")
    if start > end:
        return []
    uncovered: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    for row in intervals.sort_values("start_dt").itertuples(index=False):
        s, e = row.start_dt, row.end_dt
        if e < cursor:
            continue
        if s > end:
            break
        if s > cursor:
            uncovered.append((cursor, min(s - pd.Timedelta(minutes=1), end)))
        cursor = max(cursor, e + pd.Timedelta(minutes=1))
        if cursor > end:
            break
    if cursor <= end:
        uncovered.append((cursor, end))
    return [(s, e) for s, e in uncovered if s <= e]


def _fill_missing_step_prices(df: pd.DataFrame, steps_path: Path) -> pd.DataFrame:
    """Ensure each ticker has a populated 1_step_price column."""
    result = df.copy()
    for ticker in result["TICKER"].unique():
        mask = result["TICKER"] == ticker
        current = result.loc[mask, "1_step_price"].dropna()
        if not current.empty:
            value = float(current.iloc[0])
        else:
            try:
                value = _resolve_step_price(ticker, steps_path)
            except Exception:  # pragma: no cover - data quality guard
                logger.warning("Step price is missing for ticker '%s'", ticker)
                value = float("nan")
        result.loc[mask, "1_step_price"] = value
    return result


def download_moex_ohlc(
    log_path: PathLike,
    *,
    datetime_column: str = DEFAULT_DATETIME_COLUMN,
    ticker_column: str = DEFAULT_TICKER_COLUMN,
    encoding: str = DEFAULT_LOG_ENCODING,
    margin_minutes: int = DEFAULT_MARGIN_MINUTES,
    buffer_minutes: int = DEFAULT_BUFFER_MINUTES,
    steps_path: Optional[PathLike] = None,
    output_dir: PathLike = Path("OHCL"),
    board: Optional[str] = DEFAULT_BOARD_ID,
    save: bool = True,
) -> pd.DataFrame:
    """
    Ensure that minute OHLC data is available for every ticker found in the last test.

    The resulting dataframe is returned and, optionally, stored under
    OHCL/<log_name>_ohcl.csv. Subsequent runs reuse existing data and request only
    missing minute slots (with a small buffer to minimise boundary issues).

    The final combined dataset is clipped to the date range of the last
    test segment found in the supplied log to avoid long empty stretches
    when wider cached data is present.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Result CSV not found: {log_path}")

    steps_reference = Path(steps_path) if steps_path else DEFAULT_STEPS_PATH
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{log_path.stem}_ohcl.csv"

    last_trades = load_last_test_trades(
        log_path,
        datetime_column=datetime_column,
        ticker_column=ticker_column,
        encoding=encoding,
    )
    plan = build_download_plan(
        last_trades,
        timestamp_column="date_time",
        ticker_column="ticker",
        margin_minutes=margin_minutes,
    )
    if plan.empty:
        logger.warning("No trades detected in the last test segment; nothing to download.")
        return _empty_result()

    legacy = _read_existing(output_path)
    ticker_frames: dict[str, pd.DataFrame] = {}
    ticker_paths: dict[str, Path] = {}

    def _ensure_ticker_cache(ticker: str) -> pd.DataFrame:
        cache_path = ticker_paths.setdefault(
            ticker,
            _ticker_cache_path(output_dir, ticker),
        )
        if ticker in ticker_frames:
            return ticker_frames[ticker]

        sources: list[pd.DataFrame] = []
        on_disk = _read_existing(cache_path)
        if not on_disk.empty:
            sources.append(on_disk)
        if not legacy.empty:
            legacy_slice = legacy.loc[legacy["TICKER"] == ticker]
            if not legacy_slice.empty:
                sources.append(legacy_slice)
        ticker_frames[ticker] = (
            _combine_frames(sources, steps_path=steps_reference)
            if sources
            else _empty_result()
        )
        return ticker_frames[ticker]

    ticker_coverage: dict[str, pd.DataFrame] = {}

    for row in plan.itertuples(index=False):
        ticker = row.ticker
        ticker_existing = _ensure_ticker_cache(ticker)

        # Coverage-driven logic: compute uncovered intervals and only fetch them.
        coverage = ticker_coverage.get(ticker)
        if coverage is None:
            coverage = _read_coverage(output_dir, ticker)
            # Если покрытия нет, попробуем восстановить его из имеющихся свечей
            if coverage.empty and not ticker_existing.empty:
                min_dt = pd.to_datetime(ticker_existing["DATE_TIME"], errors="coerce").min()
                max_dt = pd.to_datetime(ticker_existing["DATE_TIME"], errors="coerce").max()
                if pd.notna(min_dt) and pd.notna(max_dt):
                    coverage = pd.DataFrame(
                        {"start_dt": [min_dt.floor("min")], "end_dt": [max_dt.ceil("min")]}
                    ).astype({"start_dt": "datetime64[ns]", "end_dt": "datetime64[ns]"})
            ticker_coverage[ticker] = coverage

        uncovered = _subtract_coverage(row.start_dt, row.end_dt, coverage)
        if not uncovered:
            logger.info(
                "Ticker %s already covered for %s .. %s",
                ticker,
                row.start_dt,
                row.end_dt,
            )
            # включаем имеющийся кэш тикера в общий результат
            ticker_frames[ticker] = ticker_existing
            continue

        for (u_start, u_end) in uncovered:
            fetch_start = (u_start - pd.Timedelta(minutes=buffer_minutes)).floor("min")
            fetch_end = (u_end + pd.Timedelta(minutes=buffer_minutes)).ceil("min")

            logger.info(
                "Downloading MOEX candles for %s: %s -> %s",
                ticker,
                fetch_start,
                fetch_end,
            )
            candles = _download_ticker_minutes(
                ticker,
                start_dt=fetch_start,
                end_dt=fetch_end,
                board=board,
            )
            if candles.empty:
                logger.warning(
                    "No candles received for %s within %s .. %s",
                    ticker,
                    fetch_start,
                    fetch_end,
                )
                # Even если пусто, считаем интервал проверенным, чтобы не перекачивать заново.
                ticker_coverage[ticker] = _add_coverage(coverage, u_start, u_end)
                coverage = ticker_coverage[ticker]
                continue

            # Оставляем только часть, которая нужна по плану / интервалу
            left = max(row.start_dt, u_start)
            right = min(row.end_dt, u_end)
            window = candles[(candles["DATE_TIME"] >= left) & (candles["DATE_TIME"] <= right)].copy()
            if window.empty:
                logger.warning(
                    "Downloaded candles for %s do not fall inside %s .. %s",
                    ticker,
                    left,
                    right,
                )
                ticker_coverage[ticker] = _add_coverage(coverage, u_start, u_end)
                coverage = ticker_coverage[ticker]
                continue

            window["TICKER"] = ticker
            window["1_step_price"] = _resolve_step_price(ticker, steps_reference)
            ticker_frames[ticker] = _combine_frames(
                [ticker_existing, window],
                steps_path=steps_reference,
            )
            ticker_existing = ticker_frames[ticker]
            # Помечаем интервал как покрытый независимо от количества минут внутри (биржевые перерывы допустимы)
            ticker_coverage[ticker] = _add_coverage(coverage, u_start, u_end)
            coverage = ticker_coverage[ticker]

    ordered_tickers = plan["ticker"].astype(str).str.strip().unique()
    selected_frames = [
        ticker_frames.get(ticker, _empty_result())
        for ticker in ordered_tickers
        if not ticker_frames.get(ticker, _empty_result()).empty
    ]
    if selected_frames:
        result = _combine_frames(selected_frames, steps_path=steps_reference)
    else:
        logger.info("No candles available; returning empty dataset.")
        result = _empty_result()

    # Clip the final dataset strictly to dates present in the last test log
    # to prevent wide gaps without trades when broader cached data exists.
    if not result.empty and not last_trades.empty:
        dt_min = pd.to_datetime(last_trades["date_time"], errors="coerce").min()
        dt_max = pd.to_datetime(last_trades["date_time"], errors="coerce").max()
        if pd.notna(dt_min) and pd.notna(dt_max):
            dt_min = dt_min.floor("min")
            dt_max = dt_max.ceil("min")
            mask = (result["DATE_TIME"] >= dt_min) & (result["DATE_TIME"] <= dt_max)
            result = result.loc[mask]
            if not result.empty:
                result = result.sort_values(["TICKER", "DATE_TIME"]).reset_index(drop=True)

    if save:
        for ticker, frame in ticker_frames.items():
            cache_path = ticker_paths[ticker]
            if not frame.empty:
                _save_result(frame, cache_path)
        for ticker, coverage in ticker_coverage.items():
            _write_coverage(output_dir, ticker, coverage)
        _save_result(result, output_path)

    return result.reset_index(drop=True)
