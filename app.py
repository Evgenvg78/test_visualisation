"""
Streamlit dashboard for equity visualization.

Features:
- Remember last logs folder and commission across runs.
- Choose multiple log files and switch between combined vs per-log view.
- Plot combined equity with Plotly; show per-log charts.
- Export per-log equity CSVs.
"""
from __future__ import annotations

import fnmatch
import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import streamlit as st

from src.equity_combination import CombinedEquityResult, LogEquitySnapshot, combine_equity_logs

logger = logging.getLogger(__name__)

STATE_FILE = Path("help_data/dashboard_state.json")
LOG_EXTENSIONS = ("*t.csv",)
DEFAULT_COMMISSION = 0.0
DEFAULT_MODE = "Комбинация"
# ---------- Persistence helpers ----------
def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(data: dict) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("Failed to save dashboard state")

# ---------- File discovery ----------
def list_log_files(folder: Path, extensions: Sequence[str] = LOG_EXTENSIONS) -> List[Path]:
    if not folder.exists():
        return []
    files: list[Path] = []
    for ext in extensions:
        # Allow wildcards like "*t.csv"
        pattern = ext if any(ch in ext for ch in "*?") else f"*{ext}"
        files.extend(folder.glob(pattern))
    return sorted({f.resolve() for f in files if f.is_file()})


def apply_mask(files: Sequence[Path], mask: str) -> list[Path]:
    """Filter filenames by glob-like mask, case-insensitive."""
    if not mask.strip():
        return list(files)
    normalized = mask.strip()
    return [p for p in files if fnmatch.fnmatch(p.name.lower(), normalized.lower())]


def pick_folder_dialog() -> Optional[str]:
    """Open native folder picker."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory()
    root.destroy()
    return folder if folder else None


# ---------- Visualization helpers ----------
def plot_per_log(snapshot: LogEquitySnapshot):
    import plotly.graph_objects as go

    df = snapshot.equity
    if "Equity" not in df.columns:
        st.warning(f"В логе {snapshot.log_path.name} нет столбца Equity")
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=df["date_time"],
            y=df["Equity"],
            mode="lines",
            name="Equity",
            line=dict(width=2),
        )
    )
    if "bot_equity" in df.columns:
        fig.add_trace(
            go.Scattergl(
                x=df["date_time"],
                y=df["bot_equity"],
                mode="lines",
                name="bot_equity",
                line=dict(width=1),
            )
        )
    fig.update_layout(
        title=f"{snapshot.ticker} ({snapshot.log_path.name})",
        height=400,
        margin=dict(l=10, r=10, t=40, b=30),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    return fig


@st.cache_data(show_spinner=False)
def process_logs(log_paths: Iterable[Path], commission: float) -> CombinedEquityResult:
    return combine_equity_logs(
        log_paths,
        process_kwargs={"comis": commission, "build_equity_report": True},
        build_plot=True,
        plot_title="Общая кривая Equity",
    )


# ---------- UI ----------
def main() -> None:
    st.set_page_config(page_title="Equity Dashboard", layout="wide")

    state = _load_state()
    st.title("Equity Dashboard (Streamlit)")
    st.caption("Минималистичный дашборд для логов торговли с комбинированными графиками Plotly.")

    folder_key = "dashboard_folder"
    commission_key = "dashboard_commission"
    mask_key = "dashboard_mask"
    mode_key = "dashboard_mode"

    with st.sidebar:
        st.header("Настройки")
        if folder_key not in st.session_state:
            st.session_state[folder_key] = state.get("folder", "")
        folder_input = st.text_input(
            "Папка с логами",
            value=st.session_state[folder_key],
            placeholder="Например: C:/data/logs",
            key=folder_key,
        )
        if st.button("Выбрать папку через проводник"):
            selected = pick_folder_dialog()
            if selected:
                st.session_state[folder_key] = selected

        if commission_key not in st.session_state:
            st.session_state[commission_key] = state.get("commission", DEFAULT_COMMISSION)
        commission = st.number_input(
            "Комиссия (per trade)",
            value=st.session_state[commission_key],
            step=0.01,
            format="%.4f",
            key=commission_key,
        )

        if mask_key not in st.session_state:
            st.session_state[mask_key] = state.get("mask", "")
        mask = st.text_input(
            "Маска файлов (glob)",
            value=st.session_state[mask_key],
            placeholder="Например: Gld*",
            help="Показывать только файлы, имя которых совпадает с маской (glob). По умолчанию — без фильтра.",
            key=mask_key,
        )

        if mode_key not in st.session_state:
            default_mode = state.get("mode", DEFAULT_MODE)
            if default_mode not in {"Комбинация", "По одному"}:
                default_mode = DEFAULT_MODE
            st.session_state[mode_key] = default_mode
        mode_options = ["Комбинация", "По одному"]
        mode = st.selectbox(
            "Режим отображения",
            options=mode_options,
            key=mode_key,
        )

        if st.button("Сохранить настройки"):
            _save_state(
                {
                    "folder": st.session_state[folder_key],
                    "commission": st.session_state[commission_key],
                    "mode": st.session_state[mode_key],
                    "mask": st.session_state[mask_key],
                }
            )
            st.success("Настройки сохранены")

    folder_path = Path(folder_input).expanduser() if folder_input else None
    if not folder_path or not folder_path.exists():
        st.info("Укажите корректную папку с логами в сайдбаре.")
        return

    files = list_log_files(folder_path)
    files = apply_mask(files, mask)
    if not files:
        st.warning("В указанной папке не найдено файлов, подходящих под маску (*.t.csv по умолчанию).")
        return

    # Main selection
    st.subheader("Выбор логов")
    options = {f"{p.name}": p for p in files}
    selected_labels = st.multiselect(
        "Файлы для анализа",
        options=list(options.keys()),
        default=list(options.keys())[:1],
    )
    selected_files = [options[label] for label in selected_labels]

    if not selected_files:
        st.info("Выберите хотя бы один файл.")
        return

    run = st.button("Обновить графики")
    if not run:
        st.stop()

    status = st.empty()
    progress = st.progress(0.0)
    try:
        status.info("Загружаются данные OHCL из интернета . . .")
        progress.progress(0.33)
        result = process_logs(selected_files, commission)
        status.info("Формируется отчёт . . .")
        progress.progress(0.66)
        status.info("Формируется график . . .")
        progress.progress(0.9)
    except Exception as exc:
        status.error(f"Не удалось обработать логи: {exc}")
        return
    finally:
        progress.progress(1.0)
    status.success("Готово!")

    # Combined view
    st.subheader("Комбинированная кривая")
    metrics = result.metrics
    cols = st.columns(5)
    cols[0].metric("GO", f"{metrics.go_requirement:,.0f}")
    cols[1].metric("Equity итог", f"{metrics.final_equity:,.0f}")
    cols[2].metric("Доходность, % GO", f"{metrics.return_percent:,.2f}")
    cols[3].metric("ДД, % GO", f"{metrics.drawdown_percent_of_go:,.2f}")
    cols[4].metric("Комиссия суммарно", f"{metrics.total_commission:,.2f}")

    if result.figure is not None:
        # Подсветить общую эквити красным цветом
        for trace in result.figure.data:
            if str(trace.name).lower().strip() in {"total equity", "equity"}:
                if hasattr(trace, "line") and trace.line is not None:
                    trace.line.color = "red"
        st.plotly_chart(result.figure, use_container_width=True)
    else:
        st.info("График не построен (plotly недоступен).")

    # Mode switch
    st.subheader(f"Режим: {mode}")
    if mode == "Комбинация":
        st.write("Используется объединённая кривая из выбранных логов.")
    else:
        if st.button("Развернуть все"):
            for snap in result.per_logs:
                st.session_state[f"expander_{snap.log_path.stem}"] = True
        for snap in result.per_logs:
            exp_key = f"expander_{snap.log_path.stem}"
            if exp_key not in st.session_state:
                st.session_state[exp_key] = False
            expanded = st.session_state.get(exp_key, False)
            with st.expander(f"{snap.log_path.name} ({snap.ticker})", expanded=expanded):
                left, right = st.columns([2, 1])
                fig = plot_per_log(snap)
                if fig:
                    left.plotly_chart(fig, use_container_width=True)
                else:
                    left.info("Не удалось построить график для этого лога.")

                if snap.report:
                    report_lines = [
                        f"- **Тикер:** {snap.report.ticker}",
                        f"- **GO:** {snap.report.go_requirement:,.0f}",
                        f"- **Финальная equity:** {snap.report.final_equity:,.0f}",
                        f"- **Доходность:** {snap.report.return_percent:.2f}%",
                        f"- **DD (валюта):** {snap.report.drawdown_currency:,.0f}",
                        f"- **DD (% GO):** {snap.report.drawdown_percent_of_go:.2f}%",
                        f"- **Сделки:** вход {snap.report.entry_count}, выход {snap.report.exit_count}",
                        f"- **Закрытых:** {snap.report.closed_trades} сделок в {snap.report.closed_cycles} циклах",
                        f"- **Комиссия:** {snap.report.total_commission:,.0f}",
                    ]
                    right.markdown("**Отчёт**")
                    right.markdown("\n".join(report_lines))
                else:
                    right.info("Отчёт пока не собран.")

                csv_bytes = snap.equity.to_csv(index=False).encode("utf-8")
                right.download_button(
                    label="Скачать CSV (equity)",
                    data=csv_bytes,
                    file_name=f"{snap.log_path.stem}_equity.csv",
                    mime="text/csv",
                    key=f"download_{snap.log_path.stem}",
                )

    # Tips
    st.info(
        "Советы: \n"
        "- Комиссия передаётся в обработку логов (`comis`).\n"
        "- Настройки папки и режима можно сохранить в сайдбаре.\n"
        "- При множественном выборе логов переключайте режим «Комбинация»/«По одному»."
    )


if __name__ == "__main__":
    main()
