from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

DEFAULT_DATA_DIR = r"G:\My Drive\data_fut"
DEFAULT_STEPS_PATH = Path(__file__).with_name("moex_tickers_steps.csv")


def _read_csv_with_fallback(
    path: Path,
    encodings: Sequence[str],
    **kwargs,
) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError("No encodings provided for CSV reading")


def _load_raw_candles(ticker: str, data_dir: Path) -> pd.DataFrame:
    cols = [
        "TICKER",
        "PER",
        "DATE",
        "TIME",
        "OPEN",
        "HIGH",
        "LOW",
        "CLOSE",
        "VOL",
    ]
    source_path = data_dir / f"{ticker}.txt"
    if not source_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {source_path}")

    frame = pd.read_csv(source_path, names=cols, sep=",", header=0)
    frame.columns = [col.strip("<>") for col in frame.columns]

    timestamps = pd.to_datetime(
        frame["DATE"].astype(str) + frame["TIME"].astype(str).str.zfill(6),
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )
    if timestamps.isna().any():
        raise ValueError(f"Invalid DATE/TIME values in {source_path}")

    frame.index = timestamps
    candles = (
        frame[["OPEN", "HIGH", "LOW", "CLOSE", "VOL"]]
        .astype(float, copy=False)
        .sort_index()
    )
    return candles


def _load_step_price(ticker: str, steps_path: Path) -> float:
    if not steps_path.exists():
        raise FileNotFoundError(f"Step price reference not found: {steps_path}")

    reference = _read_csv_with_fallback(
        steps_path,
        encodings=("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp1251"),
    )
    if "SEC_NAME" not in reference.columns or "one_step_price" not in reference.columns:
        raise ValueError(
            "Reference file must contain 'SEC_NAME' and 'one_step_price' columns"
        )

    mask = reference["SEC_NAME"].astype(str).str.strip() == ticker
    matches = reference.loc[mask, "one_step_price"]
    if matches.empty:
        raise KeyError(
            f"Ticker '{ticker}' is missing in step price reference {steps_path}"
        )
    return float(matches.iloc[0])


def load_candles(
    result_csv_path: str,
    *,
    data_dir: str = DEFAULT_DATA_DIR,
    steps_path: Optional[str] = None,
    datetime_column: str = "Дата и время",
    ticker_column: str = "Код инструмента",
    resample_rule: Optional[str] = None,
    save: bool = False,
    output_path: Optional[str] = None,
    encoding: str = "cp1251",
) -> pd.DataFrame:
    """
    Build an OHLCV dataframe for the period described in the result CSV.

    Args:
        result_csv_path: Path to CSV with trade results.
        data_dir: Directory containing raw minute data files.
        steps_path: Optional custom path to the step price reference CSV.
        datetime_column: Column name with timestamps in the result CSV.
        ticker_column: Column name with the ticker symbol.
        resample_rule: Pandas resample rule, for example '5min'.
        save: Save the resulting dataframe when True.
        output_path: Optional destination path for the saved CSV.
        encoding: Encoding used for the result CSV.

    Returns:
        A dataframe with DATE_TIME, OHLCV columns, and 1_step_price.
    """
    result_path = Path(result_csv_path)
    if not result_path.exists():
        raise FileNotFoundError(f"Result CSV not found: {result_path}")

    trades = pd.read_csv(result_path, sep=";", encoding=encoding)

    for column in (datetime_column, ticker_column):
        if column not in trades.columns:
            raise KeyError(f"Column '{column}' is missing in {result_path}")

    ticker_series = trades[ticker_column]
    start_ticker_idx = ticker_series.first_valid_index()
    if start_ticker_idx is None:
        raise ValueError(f"Could not determine ticker from '{ticker_column}' column")
    ticker = str(ticker_series.loc[start_ticker_idx]).strip()
    if not ticker:
        raise ValueError(f"Empty ticker value in column '{ticker_column}'")

    datetime_series = trades[datetime_column].astype(str).str.strip()
    parsed_datetimes = pd.to_datetime(
        datetime_series,
        format="%d.%m.%Y %H:%M:%S",
        errors="coerce",
    )
    valid_datetimes = parsed_datetimes.dropna()
    if valid_datetimes.empty:
        raise ValueError(f"Column '{datetime_column}' does not contain valid dates")

    # Определяем сегменты тестов: новый сегмент начинается, когда время откатывается назад
    chunk_boundaries = valid_datetimes < valid_datetimes.shift()
    test_ids = chunk_boundaries.cumsum()
    last_test_id = int(test_ids.iloc[-1])
    last_segment_datetimes = valid_datetimes.loc[test_ids == last_test_id]
    if last_segment_datetimes.empty:
        raise ValueError("Failed to identify the last test segment by date")

    start_dt = last_segment_datetimes.iloc[0] - pd.Timedelta(minutes=2)
    end_dt = last_segment_datetimes.iloc[-1]

    if start_dt > end_dt:
        raise ValueError("Start date is after end date")

    raw_candles = _load_raw_candles(ticker, Path(data_dir))
    window = raw_candles.loc[start_dt:end_dt]
    if window.empty:
        raise ValueError(f"No OHLC data for {ticker} in {start_dt}..{end_dt}")

    if resample_rule:
        rule = resample_rule
        if rule.endswith("T"):
            prefix = rule[:-1] or "1"
            rule = f"{prefix}min"
        window = (
            window.resample(rule)
            .agg(
                {
                    "OPEN": "first",
                    "HIGH": "max",
                    "LOW": "min",
                    "CLOSE": "last",
                    "VOL": "sum",
                }
            )
            .dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE"], how="all")
        )
        window = window.loc[start_dt:end_dt]

    window = window.sort_index()
    step_price = _load_step_price(
        ticker,
        Path(steps_path) if steps_path else DEFAULT_STEPS_PATH,
    )

    result_df = window.assign(
        DATE_TIME=window.index,
        **{"1_step_price": step_price},
    )[["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOL", "1_step_price"]]

    if save:
        target = (
            Path(output_path)
            if output_path is not None
            else result_path.with_name(f"{result_path.stem}_candle_data.csv")
        )
        result_df.to_csv(target, index=False)

    return result_df.reset_index(drop=True)
