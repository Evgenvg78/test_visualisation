"""
Версия 2 all-in-one модуля: повторяет ручной пайплайн загрузки свечей,
трансформации логов и построения equity/отчета с минимальной оберткой.
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
class AllInOneResultV2:
    """Результат пайплайна версии 2."""

    log_path: Path
    candles_path: Path
    candles: pd.DataFrame
    equity: Optional[pd.DataFrame]
    report: Optional[EquityReport]
    plot_figure: Optional[Any]
    steps_path: Path


def _ensure_steps_reference_v2(
    preferred: Optional[PathLike],
    help_data_dir: PathLike,
) -> Path:
    """
    Найти или скачать moex_tickers_steps.csv.

    Порядок:
        1. Явно переданный путь (если есть).
        2. Стандартный файл рядом с модулями данных.
        3. Копия в help_data (создается через downloader_my при необходимости).
    """

    help_dir = Path(help_data_dir)
    help_dir.mkdir(parents=True, exist_ok=True)
    fallback = help_dir / DEFAULT_STEPS_PATH.name

    for candidate in (
        Path(preferred) if preferred else None,
        DEFAULT_STEPS_PATH,
        fallback if fallback.exists() else None,
    ):
        if candidate and candidate.exists():
            return candidate

    logger.info("Файл шагов не найден, запускаю downloader_my для создания %s", fallback)
    downloaded = Path(download_moex_securities_data())
    if not downloaded.exists():
        raise FileNotFoundError("downloader_my не создал moex_tickers_steps.csv")

    shutil.copy2(downloaded, fallback)
    return fallback


def _maybe_build_plot(
    equity: pd.DataFrame,
    column: str,
    ticker_hint: str,
) -> Any:
    """Построить интерактивный график Plotly на базе equity."""
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception as exc:  # pragma: no cover - зависит от окружения
        raise RuntimeError("Для отображения графиков требуется plotly") from exc

    if column not in equity.columns:
        raise KeyError(f"Столбец '{column}' отсутствует в equity DataFrame")

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=equity["DATE_TIME"],
            y=equity[column],
            mode="lines",
            line=dict(width=1, color="blue"),
        )
    )
    fig.update_layout(
        width=1000,
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
        title=f"{ticker_hint} {column}",
    )
    return fig


def process_log_v2(
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
    plot_column: str = "Equity",
    comis: float = 0.0,
) -> AllInOneResultV2:
    """
    Полный сценарий:
        1. Загружает/кеширует OHLC через moex_ohcl_loader.
        2. Убедится, что есть справочник шагов.
        3. Передает готовые CSV в equity_builder (который уже использует csv_transformer).
        4. (опц.) учитывает комиссию `comis` при построении equity.
        5. (опц.) рисует график и строит EquityReport.
    """

    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Лог-файл не найден: {log_path}")

    steps_reference = _ensure_steps_reference_v2(steps_path, help_data_dir)

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
    if save_candles and not candles_path.exists():
        candles.to_csv(candles_path, index=False)

    equity_df: Optional[pd.DataFrame] = None
    report_obj: Optional[EquityReport] = None
    plot_fig: Optional[Any] = None

    need_equity = build_equity_report or show_plot
    if need_equity:
        if candles.empty:
            logger.warning("Свечи пустые, пропускаю расчет equity/report.")
        else:
            equity_df = build_equity(
                candles_csv=candles_path,
                trades_csv=log_path,
                comis=comis,
            )
            if equity_df.empty:
                logger.warning("Equity получился пустым; отчет построить нельзя.")
            elif build_equity_report:
                report_obj = _build_equity_report(
                    equity_df,
                    ticker_info_csv=steps_reference,
                )

    if show_plot and equity_df is not None and not equity_df.empty:
        tickers = candles["TICKER"].astype(str).str.strip().dropna().unique()
        ticker_hint = tickers[0] if len(tickers) else log_path.stem
        plot_fig = _maybe_build_plot(equity_df, plot_column, ticker_hint)

    return AllInOneResultV2(
        log_path=log_path,
        candles_path=candles_path,
        candles=candles,
        equity=equity_df,
        report=report_obj,
        plot_figure=plot_fig,
        steps_path=steps_reference,
    )


__all__ = ["AllInOneResultV2", "process_log_v2"]
