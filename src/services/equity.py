"""Service facade to build combined equity for UI/CLI consumers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

from ..equity_combination import CombinedEquityResult, build_single_equity, combine_equity_logs

logger = logging.getLogger(__name__)


def build_equity_bundle(
    log_paths: Sequence[Path],
    *,
    comis: float = 0.0,
    tz: Optional[str] = None,
    cache_key: Optional[str] = None,
    build_plot: bool = False,
    plot_title: Optional[str] = None,
) -> CombinedEquityResult:
    """
    Combine multiple equity logs into a CombinedEquityResult.

    Args:
        log_paths: paths to .csv log files.
        comis: commission per trade forwarded to the core processor.
        tz: target timezone applied to timestamps.
        cache_key: optional external cache identifier (reserved for callers).
        build_plot: whether to include a Plotly figure inside the result.
        plot_title: optional title for the combined equity plot.
    """

    process_kwargs = {"comis": comis}
    equities = [Path(p) for p in log_paths]
    result = combine_equity_logs(
        equities,
        process_kwargs=process_kwargs,
        timezone=tz,
        build_plot=build_plot,
        plot_title=plot_title or "Total Equity",
    )
    return result


__all__ = ["build_equity_bundle", "build_single_equity", "combine_equity_logs"]
