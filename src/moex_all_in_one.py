"""
High-level helper that orchestrates downloading MOEX candles, caching them,
building equity and (optionally) showing a plot/report in a single call.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from .data.candle_loader import DEFAULT_STEPS_PATH
from .data.downloader_my import download_moex_securities_data
from .data.equity_builder import build_equity
from .data.equity_report import EquityReport, build_equity_report as _build_equity_report
from .data.moex_ohcl_loader import (
    DEFAULT_BUFFER_MINUTES,
    DEFAULT_DATETIME_COLUMN,
    DEFAULT_LOG_ENCODING,
    DEFAULT_MARGIN_MINUTES,
    DEFAULT_TICKER_COLUMN,
    download_moex_ohlc,
)

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)


@dataclass
class AllInOneResult:
    """Container returned by `process_log`."""

    log_path: Path
    candles_path: Path
    candles: pd.DataFrame
    equity: Optional[pd.DataFrame] = None
    report: Optional[EquityReport] = None
    plot_figure: Optional[Any] = None
    steps_path: Optional[Path] = None


def _ensure_steps_reference(
    preferred: Optional[PathLike],
    help_data_dir: PathLike,
) -> Path:
    """
    Return a path to moex_tickers_steps.csv, downloading to /help_data when absent.
    """

    help_dir = Path(help_data_dir)
    help_dir.mkdir(parents=True, exist_ok=True)
    fallback = help_dir / DEFAULT_STEPS_PATH.name

    candidate_order = []
    if preferred:
        candidate_order.append(Path(preferred))
    candidate_order.append(DEFAULT_STEPS_PATH)
    candidate_order.append(fallback)

    for candidate in candidate_order:
        if candidate.exists():
            logger.debug("Using step-price reference at %s", candidate)
            return candidate

    logger.info(
        "Step-price reference not found, downloading via downloader_my into %s",
        fallback,
    )
    downloaded = Path(download_moex_securities_data())
    if not downloaded.exists():
        raise FileNotFoundError(
            "downloader_my did not create moex_tickers_steps.csv as expected"
        )

    shutil.copy2(downloaded, fallback)
    return fallback


def _build_plot(
    candles: pd.DataFrame,
    column: str,
    ticker_hint: str,
) -> Any:
    """Create a simple matplotlib figure for the requested column."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for plotting, but is not installed") from exc

    if column not in candles.columns:
        raise KeyError(f"Column '{column}' is missing in the candles dataframe")

    x = pd.to_datetime(candles["DATE_TIME"], errors="coerce")
    y = pd.to_numeric(candles[column], errors="coerce")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, label=f"{ticker_hint} {column}")
    ax.set_xlabel("Date")
    ax.set_ylabel(column)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def process_log(
    log_path: PathLike,
    *,
    datetime_column: str = DEFAULT_DATETIME_COLUMN,
    ticker_column: str = DEFAULT_TICKER_COLUMN,
    encoding: str = DEFAULT_LOG_ENCODING,
    steps_path: Optional[PathLike] = None,
    help_data_dir: PathLike = Path("help_data"),
    ohcl_dir: PathLike = Path("OHCL"),
    board: Optional[str] = None,
    margin_minutes: int = DEFAULT_MARGIN_MINUTES,
    buffer_minutes: int = DEFAULT_BUFFER_MINUTES,
    save_candles: bool = True,
    build_equity_report: bool = False,
    show_plot: bool = False,
    plot_column: str = "CLOSE",
) -> AllInOneResult:
    """
    Execute the full pipeline: load log, ensure OHCL cache, optionally plot/report.

    Returns:
        AllInOneResult with paths to the produced artifacts.
    """

    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log CSV not found: {log_path}")

    steps_reference = _ensure_steps_reference(steps_path, help_data_dir)

    output_dir = Path(ohcl_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candles = download_moex_ohlc(
        log_path,
        datetime_column=datetime_column,
        ticker_column=ticker_column,
        encoding=encoding,
        margin_minutes=margin_minutes,
        buffer_minutes=buffer_minutes,
        steps_path=steps_reference,
        output_dir=output_dir,
        board=board,
        save=save_candles,
    )
    candles_path = output_dir / f"{log_path.stem}_ohcl.csv"
    if build_equity_report and not candles_path.exists():
        logger.debug("Saving candles to %s for downstream equity build", candles_path)
        candles.to_csv(candles_path, index=False)

    equity_df: Optional[pd.DataFrame] = None
    report_obj: Optional[EquityReport] = None
    plot_fig: Optional[Any] = None

    if build_equity_report:
        if candles.empty:
            logger.warning("Candles dataframe is empty; skipping equity/report build.")
        else:
            equity_df = build_equity(
                candles_csv=candles_path,
                trades_csv=log_path,
            )
            if equity_df.empty:
                logger.warning("Equity dataframe is empty; report cannot be built.")
            else:
                report_obj = _build_equity_report(
                    equity_df,
                    ticker_info_csv=steps_reference,
                )

    if show_plot and not candles.empty:
        first_ticker = candles["TICKER"].astype(str).str.strip().dropna().unique()
        ticker_hint = first_ticker[0] if len(first_ticker) else log_path.stem
        plot_fig = _build_plot(candles, plot_column, ticker_hint)

    return AllInOneResult(
        log_path=log_path,
        candles_path=candles_path,
        candles=candles,
        equity=equity_df,
        report=report_obj,
        plot_figure=plot_fig,
        steps_path=steps_reference,
    )


__all__ = ["AllInOneResult", "process_log"]
