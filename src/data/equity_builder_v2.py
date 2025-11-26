"""Wrapper kept for backward compatibility with v2 naming."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from .equity_builder import build_equity

__all__ = ["build_equity_v2"]


def build_equity_v2(
    candles_csv: Union[str, Path],
    trades_csv: Union[str, Path],
    *,
    candles_encoding: str = "utf-8",
    time_margin: Union[str, pd.Timedelta] = "5min",
    comis: float = 0.0,
) -> pd.DataFrame:
    """Thin wrapper that forwards to the unified `build_equity`."""

    return build_equity(
        candles_csv=candles_csv,
        trades_csv=trades_csv,
        candles_encoding=candles_encoding,
        time_margin=time_margin,
        comis=comis,
    )
