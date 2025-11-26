"""Headless CLI to build combined equity curves from trade logs.

Usage example:
    python -m scripts.combine_equity --logs "trade_data/*.t.csv" --comis 0.5 --out result.csv --html result.html
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MASK = "*t.csv"
WILDCARD_CHARS = ("*", "?", "[")
DiscoverLogsFn = Callable[[Path, str], Iterable[Any]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine equity curves from multiple logs (headless mode)."
    )
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="Folder(s) or glob(s) with logs, e.g. trade_data/*.t.csv or trade_data/",
    )
    parser.add_argument(
        "--mask",
        default=None,
        help=f"Glob mask when --logs points to a directory (default: {DEFAULT_MASK!r}).",
    )
    parser.add_argument(
        "--comis",
        type=float,
        default=0.0,
        help="Commission per trade to pass to the equity builder.",
    )
    parser.add_argument(
        "--tz",
        default=None,
        help="Target timezone for timestamps (e.g. 'Europe/Moscow').",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("combined_equity.csv"),
        help="CSV path for the combined equity curve.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Optional JSON path for metrics (defaults to <out>.metrics.json).",
    )
    parser.add_argument(
        "--per-log-dir",
        type=Path,
        default=None,
        help="Optional directory to dump per-log equity CSVs.",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="Optional HTML output with an interactive plot (Plotly if available).",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        help="Optional PNG output (matplotlib fallback).",
    )
    parser.add_argument(
        "--plot-title",
        default=None,
        help="Custom title for exported plots.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def _load_services():
    try:
        from src.services import build_equity_bundle, discover_logs  # type: ignore
        from src.services.files import DEFAULT_MASK as SERVICE_DEFAULT_MASK  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name == "moexalgo":
            raise SystemExit(
                "Dependency 'moexalgo' is required to build equity curves. "
                "Install it with `pip install moexalgo`."
            ) from exc
        raise
    return build_equity_bundle, discover_logs, SERVICE_DEFAULT_MASK


def _has_wildcard(text: str) -> bool:
    return any(char in text for char in WILDCARD_CHARS)


def _split_source(source: str, mask_override: str | None, default_mask: str) -> tuple[Path, str]:
    raw = Path(source).expanduser()
    if _has_wildcard(source):
        folder = raw.parent if raw.parent != Path("") else Path(".")
        mask = raw.name or default_mask
    elif raw.suffix.lower() in {".csv", ".txt", ".log"}:
        folder = raw.parent if raw.parent != Path("") else Path(".")
        mask = raw.name
    else:
        folder = raw
        mask = mask_override or default_mask
    if mask_override:
        mask = mask_override
    return folder.resolve(), mask


def _unique_paths(infos: Iterable[Any]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for info in infos:
        resolved = info.path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(resolved)
    return result


def discover_log_paths(
    sources: Sequence[str],
    mask_override: str | None,
    discover_logs_fn: DiscoverLogsFn,
    default_mask: str,
) -> list[Path]:
    discovered: list[Any] = []
    for source in sources:
        folder, mask = _split_source(source, mask_override, default_mask)
        batch = discover_logs_fn(folder, mask)
        if not batch:
            logger.warning("No logs found in %s with mask %s", folder, mask)
        else:
            logger.info("Found %d log(s) in %s (mask=%s)", len(batch), folder, mask)
        discovered.extend(batch)
    return _unique_paths(discovered)


def _ts_to_iso(value) -> str | None:
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except Exception:
        return str(value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        return ts.isoformat()
    return ts.tz_convert("UTC").isoformat()


def _prepare_equity_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date_time" in out.columns:
        out["date_time"] = pd.to_datetime(out["date_time"], errors="coerce")
        out["date_time"] = out["date_time"].dt.tz_localize(None)
    return out


def save_equity_csv(df: pd.DataFrame, path: Path) -> None:
    prepared = _prepare_equity_frame(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(path, index=False)
    logger.info("Combined equity saved to %s", path)


def _per_log_payload(per_logs) -> list[dict]:
    payload = []
    for snap in per_logs:
        eq = snap.equity
        start = _ts_to_iso(eq["date_time"].iloc[0]) if not eq.empty else None
        end = _ts_to_iso(eq["date_time"].iloc[-1]) if not eq.empty else None
        payload.append(
            {
                "path": str(snap.log_path),
                "ticker": snap.ticker,
                "points": len(eq.index),
                "start": start,
                "end": end,
            }
        )
    return payload


def save_metrics(result, path: Path) -> None:
    metrics = result.metrics
    payload = {
        "go_requirement": metrics.go_requirement,
        "final_equity": metrics.final_equity,
        "return_percent": metrics.return_percent,
        "drawdown_currency": metrics.drawdown_currency,
        "drawdown_percent_of_go": metrics.drawdown_percent_of_go,
        "drawdown_start": _ts_to_iso(metrics.drawdown_start),
        "drawdown_end": _ts_to_iso(metrics.drawdown_end),
        "longest_drawdown_minutes": metrics.longest_drawdown_minutes,
        "longest_drawdown_start": _ts_to_iso(metrics.longest_drawdown_start),
        "longest_drawdown_end": _ts_to_iso(metrics.longest_drawdown_end),
        "total_commission": metrics.total_commission,
        "start_time": _ts_to_iso(metrics.start_time),
        "end_time": _ts_to_iso(metrics.end_time),
        "logs": _per_log_payload(result.per_logs),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Metrics saved to %s", path)


def dump_per_log_equity(per_logs, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for snap in per_logs:
        name = snap.log_path.stem or snap.ticker
        path = folder / f"{name}.csv"
        prepared = _prepare_equity_frame(snap.equity)
        prepared.to_csv(path, index=False)
        logger.info("Per-log equity saved to %s", path)


def _build_matplotlib_figure(result):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skipping static plot export")
        return None

    df = _prepare_equity_frame(result.combined)
    df = df.dropna(subset=["date_time"])

    fig, ax = plt.subplots(figsize=(10, 6))
    instrument_cols = [c for c in df.columns if c not in {"date_time", "Equity"}]
    for col in instrument_cols:
        ax.plot(df["date_time"], df[col], label=col, linewidth=1, alpha=0.6)
    if "Equity" in df.columns:
        ax.plot(df["date_time"], df["Equity"], label="Total Equity", color="black", linewidth=2)
    ax.set_title("Combined Equity")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend()
    fig.tight_layout()
    return fig


def _write_matplotlib_png(fig, path: Path) -> None:
    fig.savefig(path, format="png", dpi=120)
    logger.info("Static plot saved to %s", path)


def _write_matplotlib_html(fig, path: Path) -> None:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("ascii")
    html = f"<html><body><img src='data:image/png;base64,{encoded}' alt='equity plot'></body></html>"
    path.write_text(html, encoding="utf-8")
    logger.info("Static HTML plot saved to %s", path)


def export_plots(result, html_path: Path | None, png_path: Path | None) -> None:
    if html_path is None and png_path is None:
        return

    figure = result.figure
    plot_title = getattr(result, "plot_title", None)
    if figure is not None and plot_title:
        try:
            figure.update_layout(title=plot_title)
        except Exception:
            pass

    if html_path:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        if figure is not None and hasattr(figure, "write_html"):
            try:
                figure.write_html(str(html_path))
                logger.info("Interactive plot saved to %s", html_path)
            except Exception as exc:
                logger.warning("Plotly HTML export failed (%s); falling back to matplotlib.", exc)
                figure = None
        if figure is None:
            fallback = _build_matplotlib_figure(result)
            if fallback is not None:
                _write_matplotlib_html(fallback, html_path)

    if png_path:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        if figure is not None and hasattr(figure, "write_image"):
            try:
                figure.write_image(str(png_path))
                logger.info("Plot image saved to %s", png_path)
                return
            except Exception as exc:
                logger.warning("Plotly image export failed (%s); falling back to matplotlib.", exc)
        fallback = _build_matplotlib_figure(result)
        if fallback is not None:
            _write_matplotlib_png(fallback, png_path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    build_equity_bundle_fn, discover_logs_fn, service_default_mask = _load_services()
    default_mask = service_default_mask or DEFAULT_MASK

    log_paths = discover_log_paths(
        args.logs,
        args.mask,
        discover_logs_fn,
        default_mask,
    )
    if not log_paths:
        logger.error("No log files matched the provided inputs.")
        return 1

    logger.info("Building equity bundle for %d log(s)...", len(log_paths))
    result = build_equity_bundle_fn(
        log_paths,
        comis=args.comis,
        tz=args.tz,
        cache_key=None,
        build_plot=args.html is not None,
        plot_title=args.plot_title or "Combined Equity",
    )

    save_equity_csv(result.combined, args.out)
    metrics_path = args.metrics or args.out.with_suffix(".metrics.json")
    save_metrics(result, metrics_path)

    if args.per_log_dir:
        dump_per_log_equity(result.per_logs, args.per_log_dir)

    export_plots(result, args.html, args.png)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
