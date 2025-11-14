from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import warnings

import pandas as pd

DEFAULT_TICKER_INFO = Path(__file__).with_name("moex_tickers_steps.csv")


@dataclass
class EquityReport:
    """Container for all computed equity statistics."""

    ticker: str
    margin_per_contract: float
    max_contracts: int
    drawdown_points: float
    drawdown_currency: float
    drawdown_percent_of_go: float
    go_requirement: float
    entry_count: int
    exit_count: int
    total_trades: int
    closed_trades: int
    closed_cycles: int
    max_drawdown_start: Optional[pd.Timestamp]
    max_drawdown_end: Optional[pd.Timestamp]
    longest_drawdown_minutes: float
    longest_drawdown_start: Optional[pd.Timestamp]
    longest_drawdown_end: Optional[pd.Timestamp]
    final_equity: float
    return_percent: float
    total_commission: float

    def to_pretty_text(self) -> str:
        """Render the report as an emoji-friendly multi-line string."""

        def fmt_money(value: float) -> str:
            return f"{value:,.0f}".replace(",", " ")

        def fmt_num(value: float, precision: int = 2) -> str:
            return f"{value:.{precision}f}"

        def fmt_ts(ts: Optional[pd.Timestamp]) -> str:
            if ts is None or pd.isna(ts):
                return "—"
            return ts.strftime("%Y-%m-%d %H:%M")

        clr = "🟢" if self.return_percent >= 0 else "🔴"
        dd_line = (
            f"📉 Просадка: {fmt_money(self.drawdown_currency)} ₽ "
            f"({fmt_num(self.drawdown_points)} п.) = {fmt_num(self.drawdown_percent_of_go)}% от ГО"
        )
        go_line = (
            f"🛡️ ГО для торговли: {fmt_money(self.go_requirement)} ₽ "
            f"(маржа {fmt_money(self.margin_per_contract)} ₽ × {self.max_contracts} "
            f"+ буфер на просадку)"
        )
        trades_line = (
            f"🤝 Сделки: {self.total_trades} (входов {self.entry_count}, выходов {self.exit_count}, "
            f"закрытых {self.closed_trades}, циклов {self.closed_cycles})"
        )
        dd_duration = (
            f"⏱️ Длиннейшая просадка: {fmt_num(self.longest_drawdown_minutes, 1)} мин "
            f"({fmt_ts(self.longest_drawdown_start)} → {fmt_ts(self.longest_drawdown_end)})"
        )
        pnl_line = (
            f"{clr} Доходность: {fmt_money(self.final_equity)} ₽ "
            f"({fmt_num(self.return_percent)}% от ГО)"
        )
        commission_line = f"💸 Комиссии: {fmt_money(self.total_commission)} ₽"
        hot_line = (
            f"🔥 Максимальная просадка наблюдалась {fmt_ts(self.max_drawdown_start)} → "
            f"{fmt_ts(self.max_drawdown_end)}"
        )
        return "\n".join(
            [
                f"✨ Equity Report для {self.ticker}",
                go_line,
                dd_line,
                hot_line,
                dd_duration,
                trades_line,
                pnl_line,
                commission_line,
            ]
        )


def build_equity_report(
    equity_df: pd.DataFrame,
    *,
    ticker: Optional[str] = None,
    ticker_info_csv: Union[str, Path] = DEFAULT_TICKER_INFO,
) -> EquityReport:
    """Calculate the requested MOEX equity statistics."""

    if equity_df.empty:
        raise ValueError("Equity DataFrame is empty")

    working_df = equity_df.copy()
    timestamp_col = _locate_timestamp_column(working_df)
    timestamps = pd.to_datetime(working_df[timestamp_col])
    working_df = working_df.assign(_ts=timestamps).sort_values("_ts")

    ticker_name = ticker or _infer_ticker(working_df)
    max_contracts = int(working_df["current_pos"].abs().max())
    step_value = float(working_df["1_step_price"].dropna().iloc[0]) if "1_step_price" in working_df else 1.0

    equity_series = working_df["Equity"].astype(float).reset_index(drop=True)
    ts_series = working_df["_ts"].reset_index(drop=True)
    reference_ts = ts_series.iloc[-1]
    margin_per_contract = _load_initial_margin(ticker_name, ticker_info_csv, reference_ts)

    # Equity is already in currency units; _max_drawdown returns currency drawdown.
    drawdown_currency, dd_start, dd_end = _max_drawdown(equity_series, ts_series)
    # Convert currency drawdown into points using step value.
    drawdown_points = (drawdown_currency / step_value) if step_value else 0.0
    go_requirement = max_contracts * margin_per_contract + drawdown_currency
    drawdown_percent = (drawdown_currency / go_requirement * 100) if go_requirement else 0.0

    longest_minutes, longest_start, longest_end = _longest_drawdown_window(equity_series, ts_series)

    entry_count, exit_count = _count_trade_events(working_df)
    closed_cycles = int(((working_df["current_pos"] == 0) & (working_df["prev_position"] != 0)).sum())
    closed_trades = min(entry_count, exit_count)
    total_trades = entry_count + exit_count

    final_equity = float(equity_series.iloc[-1])
    total_commission = (
        float(working_df["comis_count"].fillna(0.0).sum())
        if "comis_count" in working_df
        else 0.0
    )
    return_percent = (final_equity / go_requirement * 100) if go_requirement else 0.0

    return EquityReport(
        ticker=ticker_name,
        margin_per_contract=margin_per_contract,
        max_contracts=max_contracts,
        drawdown_points=drawdown_points,
        drawdown_currency=drawdown_currency,
        drawdown_percent_of_go=drawdown_percent,
        go_requirement=go_requirement,
        entry_count=entry_count,
        exit_count=exit_count,
        total_trades=total_trades,
        closed_trades=closed_trades,
        closed_cycles=closed_cycles,
        max_drawdown_start=dd_start,
        max_drawdown_end=dd_end,
        longest_drawdown_minutes=longest_minutes,
        longest_drawdown_start=longest_start,
        longest_drawdown_end=longest_end,
        final_equity=final_equity,
        return_percent=return_percent,
        total_commission=total_commission,
    )


def _locate_timestamp_column(df: pd.DataFrame) -> str:
    for candidate in ("DATE_TIME", "date_time"):
        if candidate in df.columns:
            return candidate
    raise KeyError("Neither 'DATE_TIME' nor 'date_time' column found in equity DataFrame")


def _infer_ticker(df: pd.DataFrame) -> str:
    if "TICKER" not in df.columns:
        raise KeyError("Equity frame must contain 'TICKER' column or ticker must be provided explicitly")
    tickers = df["TICKER"].dropna().unique()
    if len(tickers) == 0:
        raise ValueError("Ticker column is empty, specify ticker explicitly")
    if len(tickers) > 1:
        counts = df["TICKER"].value_counts(dropna=True)
        top = str(counts.index[0])
        warnings.warn(
            f"Multiple tickers detected ({tickers}); using the most frequent one: {top}",
            stacklevel=2,
        )
        return top
    return str(tickers[0])


def _load_initial_margin(
    ticker: str,
    csv_path: Union[str, Path],
    reference_date: pd.Timestamp,
) -> float:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"MOEX ticker info CSV not found: {csv_path}")
    ticker_info = pd.read_csv(csv_path)
    match = ticker_info.loc[ticker_info["SECID"] == ticker]
    if match.empty:
        base = ticker[:-2]
        approx = ticker_info[ticker_info["SECID"].str.startswith(base, na=False)]
        if approx.empty:
            raise KeyError(f"Ticker {ticker!r} not found inside {csv_path}")
        approx = approx.copy()
        approx["LASTTRADEDATE"] = pd.to_datetime(approx["LASTTRADEDATE"], errors="coerce")
        approx["distance"] = (approx["LASTTRADEDATE"] - pd.to_datetime(reference_date)).abs()
        approx = approx.sort_values(["distance", "LASTTRADEDATE"])
        fallback = approx.iloc[0]
        warnings.warn(
            f"Ticker {ticker!r} not found; using closest available contract {fallback['SECID']!r}",
            stacklevel=2,
        )
        match = pd.DataFrame([fallback])
    margin = match["INITIALMARGIN"].iloc[0]
    return float(margin)


def _max_drawdown(
    equity: pd.Series,
    timestamps: pd.Series,
) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    running_max = equity.cummax()
    drawdown = running_max - equity
    max_dd = float(drawdown.max())
    if max_dd <= 0:
        return 0.0, None, None

    dd_end_idx = int(drawdown.idxmax())
    dd_end_time = timestamps.iloc[dd_end_idx]
    peak_mask = equity.loc[:dd_end_idx] == running_max.loc[:dd_end_idx]
    peak_indices = peak_mask[peak_mask].index
    dd_start_time = timestamps.iloc[int(peak_indices[-1])] if len(peak_indices) else timestamps.iloc[0]
    return max_dd, dd_start_time, dd_end_time


def _longest_drawdown_window(
    equity: pd.Series,
    timestamps: pd.Series,
) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    running_max = equity.cummax()
    best_duration = pd.Timedelta(0)
    best_window: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]] = (None, None)
    drawdown_active = False
    window_start: Optional[pd.Timestamp] = None

    for value, ts, peak in zip(equity, timestamps, running_max):
        if value >= peak:
            if drawdown_active:
                current_duration = ts - window_start  # type: ignore[operator]
                if current_duration > best_duration:
                    best_duration = current_duration
                    best_window = (window_start, ts)
                drawdown_active = False
            window_start = ts
            continue

        if not drawdown_active:
            drawdown_active = True
            window_start = ts

    if drawdown_active:
        ts = timestamps.iloc[-1]
        current_duration = ts - window_start  # type: ignore[operator]
        if current_duration > best_duration:
            best_duration = current_duration
            best_window = (window_start, ts)

    minutes = best_duration.total_seconds() / 60 if best_duration.total_seconds() else 0.0
    return minutes, best_window[0], best_window[1]


def _count_trade_events(df: pd.DataFrame) -> Tuple[int, int]:
    entries = exits = 0
    prev_series = df["prev_position"].fillna(0)
    curr_series = df["current_pos"].fillna(0)
    for prev_pos, curr_pos in zip(prev_series, curr_series):
        if prev_pos == curr_pos:
            continue
        if prev_pos == 0 and curr_pos != 0:
            entries += 1
        elif curr_pos == 0 and prev_pos != 0:
            exits += 1
        elif (prev_pos > 0 and curr_pos > 0) or (prev_pos < 0 and curr_pos < 0):
            if abs(curr_pos) > abs(prev_pos):
                entries += 1
            else:
                exits += 1
        else:
            exits += 1
            entries += 1
    return entries, exits


__all__ = ["EquityReport", "build_equity_report"]
