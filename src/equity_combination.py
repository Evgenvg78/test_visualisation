from __future__ import annotations

import logging
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .data.equity_report import EquityReport, _longest_drawdown_window, _max_drawdown
from .moex_all_in_one_v2 import process_log_v2

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LogEquitySnapshot:
    """Per-log equity data and the related report."""

    log_path: Path
    ticker: str
    equity: pd.DataFrame
    series: pd.Series
    report: Optional[EquityReport]


@dataclass(frozen=True)
class CombinedEquityMetrics:
    """Metrics derived from the combined equity curve."""

    go_requirement: float
    final_equity: float
    return_percent: float
    drawdown_currency: float
    drawdown_percent_of_go: float
    drawdown_start: Optional[pd.Timestamp]
    drawdown_end: Optional[pd.Timestamp]
    longest_drawdown_minutes: float
    longest_drawdown_start: Optional[pd.Timestamp]
    longest_drawdown_end: Optional[pd.Timestamp]
    total_commission: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp


@dataclass(frozen=True)
class CombinedEquityResult:
    """Result of combining multiple equity logs."""

    combined: pd.DataFrame
    metrics: CombinedEquityMetrics
    per_logs: Tuple[LogEquitySnapshot, ...]
    figure: Optional[Any]


def combine_equity_logs(
    log_paths: Sequence[PathLike[str]],
    *,
    resample_rule: Optional[str] = None,
    process_kwargs: Optional[Mapping[str, Any]] = None,
    timezone: Optional[str] = None,
    build_plot: bool = False,
    plot_title: str = "Total Equity",
) -> CombinedEquityResult:
    """
    Combine equity curves from multiple logs and return a consolidated result.

    Args:
        log_paths: paths to trade log CSVs.
        resample_rule: optional resampling rule for the combined frame.
        process_kwargs: extra arguments forwarded to process_log_v2.
        timezone: target timezone for timestamps (if not None).
        build_plot: build a Plotly figure that overlays all series plus the total.
        plot_title: title for the combined figure.

    Raises:
        ValueError: if no equity data is available after processing the logs.
    """

    process_options: dict[str, Any] = {"build_equity_report": True, "show_plot": False}
    if process_kwargs:
        process_options.update(process_kwargs)
    process_options["build_equity_report"] = True

    snapshots: list[LogEquitySnapshot] = []
    for path in log_paths:
        log_path = Path(path)
        result = process_log_v2(log_path, **process_options)
        equity = _normalize_equity(result.equity, timezone=timezone)
        if equity is None or equity.empty:
            logger.warning("Skipping %s because equity is empty", log_path)
            continue
        ticker = _resolve_ticker(equity, log_path)
        series = _series_from_equity(equity, ticker)
        snapshots.append(
            LogEquitySnapshot(
                log_path=log_path,
                ticker=ticker,
                equity=equity,
                series=series,
                report=result.report,
            )
        )

    if not snapshots:
        raise ValueError("No equity data was produced by any of the logs")

    combined_df = _build_combined_dataframe(snapshots, resample_rule=resample_rule)
    go_requirement = _total_go(snapshots)
    total_commission = _total_commission(snapshots)
    metrics = _build_metrics(combined_df, go_requirement, total_commission)

    figure = None
    if build_plot:
        instr_columns = _instrument_columns(combined_df)
        total_series = combined_df.set_index("date_time")["Equity"]
        per = {
            col: combined_df.set_index("date_time")[col] for col in instr_columns
        }
        figure = _build_plot(per, total_series, title=plot_title)

    return CombinedEquityResult(
        combined=combined_df,
        metrics=metrics,
        per_logs=tuple(snapshots),
        figure=figure,
    )


def _normalize_equity(
    equity: Optional[pd.DataFrame],
    *,
    timezone: Optional[str],
) -> Optional[pd.DataFrame]:
    if equity is None or equity.empty:
        return None
    normalized = equity.copy()
    timestamp_col = _locate_timestamp_column(normalized)
    localized = pd.to_datetime(normalized[timestamp_col], errors="coerce")
    if localized.dt.tz is not None:
        localized = localized.dt.tz_convert("UTC")
        if timezone:
            localized = localized.dt.tz_convert(timezone)
    elif timezone:
        localized = localized.dt.tz_localize(timezone)

    normalized["date_time"] = localized
    normalized["Equity"] = pd.to_numeric(normalized["Equity"], errors="coerce")
    normalized = normalized.dropna(subset=["date_time", "Equity"])
    normalized = normalized.sort_values("date_time").reset_index(drop=True)
    normalized = normalized[~normalized["date_time"].duplicated(keep="last")]
    return normalized


def _locate_timestamp_column(df: pd.DataFrame) -> str:
    for candidate in ("date_time", "DATE_TIME"):
        if candidate in df.columns:
            return candidate
    raise KeyError("DataFrame must contain either 'date_time' or 'DATE_TIME'")


def _resolve_ticker(equity: pd.DataFrame, log_path: Path) -> str:
    if "TICKER" in equity.columns:
        tickers = equity["TICKER"].dropna().astype(str).str.strip()
        if not tickers.empty:
            return tickers.iloc[0]
    return log_path.stem


def _series_from_equity(equity: pd.DataFrame, name: str) -> pd.Series:
    series = equity.set_index("date_time")["Equity"].astype(float)
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series.name = name
    return series


def _unique_column_name(base: str, existing: set[str]) -> str:
    clean = base.strip() or "Equity"
    candidate = clean
    counter = 1
    while candidate in existing or candidate in {"Equity", "date_time"}:
        candidate = f"{clean}_{counter}"
        counter += 1
    existing.add(candidate)
    return candidate


def _build_combined_dataframe(
    snapshots: Sequence[LogEquitySnapshot],
    *,
    resample_rule: Optional[str],
) -> pd.DataFrame:
    idx = snapshots[0].series.index
    for snap in snapshots[1:]:
        idx = idx.union(snap.series.index)

    builder = pd.DataFrame(index=idx)
    seen: set[str] = set()
    columns: list[str] = []
    for snap in snapshots:
        column = _unique_column_name(snap.ticker, seen)
        values = snap.series.reindex(idx)
        values = values.ffill().fillna(method="bfill")
        if values.isna().all():
            continue
        builder[column] = values
        columns.append(column)

    if not columns:
        raise ValueError("Unable to build any columns for the combined Equity frame")

    builder["Equity"] = builder[columns].sum(axis=1)
    if resample_rule:
        builder = (
            builder.resample(resample_rule)
            .last()
            .dropna(how="all")
        )
        if builder.empty:
            raise ValueError("Resample dropped all equity rows")
        builder["Equity"] = builder[columns].sum(axis=1)

    builder = builder.reset_index().rename(columns={"index": "date_time"})
    return builder


def _instrument_columns(combined: pd.DataFrame) -> list[str]:
    return [col for col in combined.columns if col not in {"date_time", "Equity"}]


def _total_go(snapshots: Sequence[LogEquitySnapshot]) -> float:
    return sum(
        snap.report.go_requirement for snap in snapshots if snap.report is not None
    )


def _total_commission(snapshots: Sequence[LogEquitySnapshot]) -> float:
    return sum(
        snap.report.total_commission for snap in snapshots if snap.report is not None
    )


def _build_metrics(
    combined: pd.DataFrame,
    go_requirement: float,
    total_commission: float,
) -> CombinedEquityMetrics:
    timestamps = combined["date_time"]
    equity_series = combined["Equity"].astype(float)
    drawdown_currency, dd_start, dd_end = _max_drawdown(equity_series, timestamps)
    longest_minutes, longest_start, longest_end = _longest_drawdown_window(
        equity_series, timestamps
    )
    final_equity = float(equity_series.iloc[-1])
    return_percent = (
        float("nan") if go_requirement == 0 else final_equity / go_requirement * 100
    )
    drawdown_percent = (
        float("nan") if go_requirement == 0 else drawdown_currency / go_requirement * 100
    )
    return CombinedEquityMetrics(
        go_requirement=go_requirement,
        final_equity=final_equity,
        return_percent=return_percent,
        drawdown_currency=drawdown_currency,
        drawdown_percent_of_go=drawdown_percent,
        drawdown_start=dd_start,
        drawdown_end=dd_end,
        longest_drawdown_minutes=longest_minutes,
        longest_drawdown_start=longest_start,
        longest_drawdown_end=longest_end,
        total_commission=total_commission,
        start_time=timestamps.iloc[0],
        end_time=timestamps.iloc[-1],
    )


def _build_plot(
    instrument_series: Mapping[str, pd.Series],
    total_series: pd.Series,
    *,
    title: str,
) -> Optional[Any]:
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.debug("plotly is not installed, skipping combined equity plot")
        return None

    fig = go.Figure()
    for name, series in instrument_series.items():
        fig.add_trace(
            go.Scattergl(
                x=series.index,
                y=series.values,
                mode="lines",
                name=name,
                line=dict(width=1),
            )
        )
    fig.add_trace(
        go.Scattergl(
            x=total_series.index,
            y=total_series.values,
            mode="lines",
            name="Total Equity",
            line=dict(width=2, color="black"),
        )
    )
    fig.update_layout(
        title=title,
        margin=dict(l=40, r=40, t=40, b=40),
        width=1000,
        height=500,
        legend=dict(orientation="h", y=1.02, x=0),
    )
    return fig


__all__ = [
    "CombinedEquityResult",
    "CombinedEquityMetrics",
    "LogEquitySnapshot",
    "combine_equity_logs",
]
