"""
Streamlit dashboard for equity visualization.

Features:
- Remember last logs folder and commission across runs.
- Choose multiple log files and switch between combined vs per-log view.
- Plot combined equity with Plotly; show per-log charts.
- Export per-log equity CSVs.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import streamlit as st

from src.equity_combination import CombinedEquityResult, LogEquitySnapshot
from src.services import LogFileInfo, build_equity_bundle, discover_logs

logger = logging.getLogger(__name__)

STATE_FILE = Path("help_data/dashboard_state.json")
DEFAULT_COMMISSION = 0.5
DEFAULT_MASK = "*t.csv"
LOGS_STATE_KEY = "logs"
RESULT_STATE_KEY = "result"
DESCRIPTORS_STATE_KEY = "last_descriptors"


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


def _prompt_for_folder(initial_path: Optional[str] = None) -> Optional[str]:
    """Show a native folder picker and return the chosen directory."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        logger.exception("Tkinter is not available for folder selection")
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        initial_dir = None
        if initial_path:
            candidate = Path(initial_path).expanduser()
            if candidate.exists():
                initial_dir = str(candidate)
        selected = filedialog.askdirectory(initialdir=initial_dir)
    finally:
        root.destroy()

    if not selected:
        return None
    return str(Path(selected))


# ---------- File discovery ----------
def _get_log_infos() -> list[LogFileInfo]:
    return list(st.session_state.get(LOGS_STATE_KEY, []))


def _set_log_infos(infos: Sequence[LogFileInfo]) -> None:
    st.session_state[LOGS_STATE_KEY] = list(infos)


def _build_log_descriptors(logs: Sequence[LogFileInfo]) -> Tuple[Tuple[str, float], ...]:
    """Return tuples (absolute_path, mtime) for caching."""
    descriptors: list[Tuple[str, float]] = []
    for info in logs:
        descriptors.append((str(Path(info.path).resolve()), float(info.mtime)))
    return tuple(descriptors)


# ---------- Visualization helpers ----------
def plot_per_log(snapshot: LogEquitySnapshot):
    import plotly.graph_objects as go

    df = snapshot.equity
    if "Equity" not in df.columns:
        st.warning(f"В файле {snapshot.log_path.name} нет столбца Equity.")
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


def _process_logs(log_paths: Iterable[Path], commission: float) -> CombinedEquityResult:
    paths = [Path(p) for p in log_paths]
    cache_key = ';'.join(str(p.resolve()) for p in paths)
    return build_equity_bundle(
        paths,
        comis=commission,
        tz=None,
        cache_key=cache_key,
        build_plot=True,
    )


@st.cache_data(show_spinner=False)
def process_logs_cached(
    log_descriptors: Tuple[Tuple[str, float], ...],
    commission: float,
) -> CombinedEquityResult:
    log_paths = [Path(path_str) for path_str, _ in log_descriptors]
    return _process_logs(log_paths, commission)


# ---------- UI ----------
def main() -> None:
    st.set_page_config(page_title="Equity Dashboard", layout="wide")

    state = _load_state()
    st.title("Дашборд Equity")
    st.caption("Загружайте CSV-логи, объединяйте их и просматривайте отчёты напрямую в Streamlit.")

    folder_key = "dashboard_folder"
    commission_key = "dashboard_commission"
    mask_key = "dashboard_mask"
    with st.sidebar:
        st.header("Параметры запуска")
        folder_pending_key = "_dashboard_folder_pending"
        folder_feedback_key = "_dashboard_folder_feedback"
        if folder_pending_key in st.session_state:
            st.session_state[folder_key] = st.session_state.pop(folder_pending_key)
        if folder_key not in st.session_state:
            st.session_state[folder_key] = state.get("folder", "")
        folder_input = st.text_input(
            "Папка с логами",
            value=st.session_state[folder_key],
            placeholder="Например: C:/data/logs",
            key=folder_key,
        )

        if st.button("Выбрать папку через проводник", key="pick_logs_folder"):
            selected_folder = _prompt_for_folder(st.session_state.get(folder_key))
            if selected_folder:
                st.session_state[folder_pending_key] = selected_folder
                st.session_state[folder_feedback_key] = ("success", "Папка успешно обновлена.")
            else:
                st.session_state[folder_feedback_key] = ("info", "Папка не была выбрана.")
            st.rerun()

        feedback = st.session_state.pop(folder_feedback_key, None)
        if feedback:
            level, message = feedback
            if level == "success":
                st.success(message)
            elif level == "info":
                st.info(message)
            else:
                st.write(message)

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
            st.session_state[mask_key] = state.get("mask", DEFAULT_MASK)
        current_mask = st.session_state[mask_key]
        if not str(current_mask).strip():
            current_mask = DEFAULT_MASK
            st.session_state[mask_key] = DEFAULT_MASK
        mask = st.text_input(
            "Фильтр файлов (glob)",
            value=current_mask,
            placeholder="Например: *t.csv или Gold*.csv",
            help="Используйте glob-шаблоны: *t.csv, Gold*.csv, 2024_* и т.д.",
            key=mask_key,
        )
        normalized_mask = mask.strip() or DEFAULT_MASK
        if normalized_mask != st.session_state[mask_key]:
            st.session_state[mask_key] = normalized_mask
        mask_value = normalized_mask

        folder_path = Path(folder_input).expanduser() if folder_input else None
        if st.button("Найти логи"):
            if not folder_path or not folder_path.exists():
                st.error("Укажите существующую папку с логами.")
            else:
                discovered = discover_logs(folder_path, mask_value)
                _set_log_infos(list(discovered))
                st.session_state.pop(RESULT_STATE_KEY, None)
                st.session_state.pop(DESCRIPTORS_STATE_KEY, None)
                st.success(f"Найдено {len(discovered)} логов.")

        if st.button("Сохранить настройки"):
            _save_state(
                {
                    "folder": st.session_state[folder_key],
                    "commission": st.session_state[commission_key],
                    "mask": st.session_state[mask_key],
                }
            )
            st.success("Настройки сохранены.")

    available_logs = _get_log_infos()
    if not available_logs:
        st.info("Добавьте логи через панель слева и нажмите «Найти логи», чтобы увидеть список файлов.")
        return

    st.subheader("Выбор логов для анализа")
    option_map = {
        f"[{idx + 1}] {info.path.name} - {Path(info.path).parent}": info
        for idx, info in enumerate(available_logs)
    }
    options = list(option_map.keys())
    default_selection = options[: min(len(options), 3)]
    selected_labels = st.multiselect(
        "Выберите файлы, которые нужно объединить",
        options=options,
        default=default_selection,
    )
    selected_infos = [option_map[label] for label in selected_labels]

    if not selected_infos:
        st.info("Отметьте хотя бы один лог, чтобы продолжить.")
        return

    st.write("1) Обновите список логов после изменения параметров. 2) Нажмите «Запустить анализ».")
    run_clicked = st.button("Запустить анализ", type="primary")
    descriptors = _build_log_descriptors(selected_infos)
    cached_descriptors = st.session_state.get(DESCRIPTORS_STATE_KEY)
    result: Optional[CombinedEquityResult] = st.session_state.get(RESULT_STATE_KEY)

    if run_clicked or result is None or cached_descriptors != descriptors:
        if not run_clicked:
            st.info("Параметры изменились — нажмите «Запустить анализ» ещё раз.")
            return

        status = st.empty()
        progress = st.progress(0.0)
        try:
            status.info("Обрабатываем выбранные файлы...")
            progress.progress(0.3)
            result = process_logs_cached(descriptors, commission)
            st.session_state[RESULT_STATE_KEY] = result
            st.session_state[DESCRIPTORS_STATE_KEY] = descriptors
            status.info("Готовим графики и метрики...")
            progress.progress(0.7)
        except Exception as exc:
            status.error(f"Не удалось обработать equity: {exc}")
            return
        finally:
            progress.progress(1.0)
        status.success("Готово!")
    elif result is None:
        st.info("Параметры изменились — нажмите «Запустить анализ» ещё раз.")
        return

    st.subheader("Итоги по портфелю")
    metrics = result.metrics
    cols = st.columns(5)
    cols[0].metric("ГО", f"{metrics.go_requirement:,.0f}")
    cols[1].metric("Equity (итог)", f"{metrics.final_equity:,.0f}")
    cols[2].metric("Доходность, % ГО", f"{metrics.return_percent:,.2f}")
    cols[3].metric("Просадка, % ГО", f"{metrics.drawdown_percent_of_go:,.2f}")
    cols[4].metric("Комиссия", f"{metrics.total_commission:,.2f}")

    if result.figure is not None:
        for trace in result.figure.data:
            if str(trace.name).lower().strip() in {"total equity", "equity"}:
                if hasattr(trace, "line") and trace.line is not None:
                    trace.line.color = "red"
        st.plotly_chart(result.figure, use_container_width=True)
    else:
        st.info("График не построен — возможно, Plotly не смог обработать данные.")

    st.subheader("Детализация по каждому логу")
    expander_keys = [f"expander_{snap.log_path.stem}" for snap in result.per_logs]
    if st.button("Развернуть все блоки"):
        for key in expander_keys:
            if not st.session_state.get(key, False):
                st.session_state[key] = True
    for snap in result.per_logs:
        exp_key = f"expander_{snap.log_path.stem}"
        if exp_key not in st.session_state:
            st.session_state[exp_key] = False
        expanded = st.session_state.get(exp_key, False)
        title = f"{snap.log_path.name} ({snap.ticker})"


        with st.expander(title, expanded=expanded):
            left, right = st.columns([2, 1])
            fig = plot_per_log(snap)
            if fig:
                left.plotly_chart(fig, use_container_width=True)
            else:
                left.info("Нет данных для построения графика.")

            if snap.report:
                report_lines = [
                    f"- **Тикер:** {snap.report.ticker}",
                    f"- **ГО:** {snap.report.go_requirement:,.0f}",
                    f"- **Финальное equity:** {snap.report.final_equity:,.0f}",
                    f"- **Доходность:** {snap.report.return_percent:.2f}%",
                    f"- **Просадка (валюта):** {snap.report.drawdown_currency:,.0f}",
                    f"- **Просадка (% ГО):** {snap.report.drawdown_percent_of_go:.2f}%",
                    f"- **Сделки:** входов {snap.report.entry_count}, выходов {snap.report.exit_count}",
                    f"- **Закрыто трейдов:** {snap.report.closed_trades}",
                    f"- **Комиссия:** {snap.report.total_commission:,.0f}",
                ]
                right.markdown("**Отчёт**")
                right.markdown("\n".join(report_lines))
            else:
                right.info("Отчёт по сделкам не найден.")

            csv_bytes = snap.equity.to_csv(index=False).encode("utf-8")
            right.download_button(
                label="Скачать CSV (equity)",
                data=csv_bytes,
                file_name=f"{snap.log_path.stem}_equity.csv",
                mime="text/csv",
                key=f"download_{snap.log_path.stem}",
            )

    st.info(
        "Подсказки:\n"
        "- После изменения папки или маски обновите список логов, иначе анализ будет проводиться по предыдущему набору.\n"
        "- Если файлов много, сократите выборку — это ускорит вычисления и уменьшит нагрузку на Plotly.\n"
        "- Скачивайте CSV из каждого блока, чтобы разбирать equity во внешних инструментах."
    )


if __name__ == "__main__":
    main()
