"""Сборка единой таблицы Equity из свечей и сделок.

Функция build_equity читает CSV со свечами и CSV со сделками, приводит столбцы дат к единому
формату, выполняет left-merge по минутному таймстемпу свечей и рассчитывает колонки
action_price, deal_price, current_pos, prev_position, prev_action_price, delta_price,
Pnl и Equity согласно заданию.

Пример использования:
    from src.data.equity_builder import build_equity
    df = build_equity(candles_csv='candles.csv', trades_csv='trades.csv')
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from .csv_transformer import load_transform_csv

__all__ = ["build_equity"]


def _ensure_datetime(col: pd.Series) -> pd.Series:
    """Привести серию к pd.Timestamp и округлить вниз до минуты."""
    return pd.to_datetime(col, errors="coerce").dt.floor("min")


def build_equity(
    candles_csv: Union[str, Path],
    trades_csv: Union[str, Path],
    candles_encoding: str = "utf-8",
    trades_use_transformer: bool = True,
    comis: float = 0.0,
) -> pd.DataFrame:
    """Построить DataFrame с Equity по минутам.

    Args:
        candles_csv: путь к CSV со свечами. Ожидается колонка DATE_TIME (или date_time)
            и колонки OPEN,HIGH,LOW,CLOSE,VOL,1_step_price.
        trades_csv: путь к CSV со сделками. Можно использовать модульный парсер
            (csv_transformer.load_transform_csv) — по умолчанию используется он.
        candles_encoding: encoding при чтении файла со свечами.
        trades_use_transformer: если True, используем load_transform_csv для чтения сделок
            (корректно обработает cp1251/sep=';'). Если False, прочитаем через pandas
            автоматически (может потребоваться корректировка разделителя/кодировки).
        comis: комиссия в рублях за контракт на каждое действие (вход/выход).

    Returns:
        pd.DataFrame объединённый и дополненный колонками по заданию.
        Новая колонка `comis_count` показывает комиссию за действие в рублях.
    """
    candles_path = Path(candles_csv)
    if not candles_path.exists():
        raise FileNotFoundError(f"Candles CSV not found: {candles_csv}")

    # 1) Читать свечи
    candles = pd.read_csv(candles_path, encoding=candles_encoding)

    # Поддержать возможно разные имена столбца даты
    if "DATE_TIME" in candles.columns:
        dt_col = "DATE_TIME"
    elif "date_time" in candles.columns:
        dt_col = "date_time"
    else:
        # Попробуем найти столбец похожий на дату
        possible = [c for c in candles.columns if "date" in c.lower()]
        if not possible:
            raise KeyError(
                "Не найден столбец даты в файле свечей (ожидалось 'DATE_TIME')"
            )
        dt_col = possible[0]

    candles = candles.copy()
    candles["date_time"] = _ensure_datetime(candles[dt_col])

    # 2) Читать сделки
    if trades_use_transformer:
        # ����������� ������ ���������� ��������������� �������: date_time, price, current_pos
        trades = load_transform_csv(trades_csv)
        trades = trades.copy()
        trades["_order"] = trades.index
        # ����������� ���� � ������ �������
        if "date_time" in trades.columns:
            trades["date_time"] = _ensure_datetime(trades["date_time"])
        if "price" in trades.columns:
            trades["price"] = pd.to_numeric(trades["price"], errors="coerce").round(2)
        if "current_pos" in trades.columns:
            trades["current_pos"] = (
                pd.to_numeric(trades["current_pos"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
    else:
        trades_path = Path(trades_csv)
        if not trades_path.exists():
            raise FileNotFoundError(f"Trades CSV not found: {trades_csv}")
        # Попробуем прочитать с sep=';'
        trades = pd.read_csv(trades_path, sep=";", encoding="cp1251")
        trades["_order"] = trades.index
        # Привести названия и типы к унифицированному виду (date_time, price, current_pos)
        # Поддержим старые русские имена при чтении вручную
        if "Дата и время" in trades.columns:
            trades = trades.copy()
            trades["date_time"] = _ensure_datetime(trades["Дата и время"])
        if "Цена последней сделки" in trades.columns:
            trades = trades.copy()
            trades["price"] = pd.to_numeric(
                trades["Цена последней сделки"]
                .astype(str)
                .str.replace(r"\s+", "", regex=True)
                .str.replace(",", "."),
                errors="coerce",
            )
        if "Текущее количество контрактов" in trades.columns:
            trades = trades.copy()
            cleaned_current_pos = (
                trades["Текущее количество контрактов"]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(r"[^\d\.\-]", "", regex=True)
            )
            trades["current_pos"] = (
                pd.to_numeric(cleaned_current_pos, errors="coerce")
                .fillna(0.0)
                .round()
                .astype(int)
            )
        if "date_time" in trades.columns:
            trades["date_time"] = _ensure_datetime(trades["date_time"])

    # Убедиться, что в trades есть колонка date_time
    if "date_time" not in trades.columns:
        # ������� ����� ������� � �����
        poss = [c for c in trades.columns if "date" in c.lower()]
        if poss:
            trades["date_time"] = _ensure_datetime(trades[poss[0]])
        else:
            raise KeyError(
                "�� ������ ������� ���� � ����� ������ (��������� 'date_time' ��� '���� � �����')"
            )

    trades = trades.sort_values(["date_time", "_order"])
    trades = trades.drop(columns="_order")

    # Merge: свечной файл — основной (left join по минутной дате)
    # Прежде чем мержить, переименуем торговые колонки, чтобы избежать конфликтов
    trades = trades.copy()
    # Выполняем merge — ожидаем унифицированные колонки: date_time, price, current_pos
    right = ["date_time"]
    if "price" in trades.columns:
        right.append("price")
    if "current_pos" in trades.columns:
        right.append("current_pos")

    merged = pd.merge(
        candles,
        trades[right].drop_duplicates(subset=["date_time"], keep="last"),
        how="left",
        left_on="date_time",
        right_on="date_time",
    )

    # Переименуем поля price -> action_price и оставим current_pos как есть
    rename_map = {}
    if "price" in merged.columns:
        rename_map["price"] = "action_price"
    if "current_pos" in merged.columns:
        rename_map["current_pos"] = "current_pos"
    merged = merged.rename(columns=rename_map)

    # Заполнить current_pos вниз (forward fill), начальное значение — 0
    merged["current_pos"] = merged["current_pos"].ffill().fillna(0).astype(int)

    # prev_position = current_pos.shift(1), top = 0
    merged["prev_position"] = merged["current_pos"].shift(1).fillna(0).astype(int)

    # В action_price пустые ячейки заполнить: если prev_position < 0 -> HIGH, иначе LOW
    # Сначала убедимся, что колонок HIGH/LOW есть
    if "HIGH" not in merged.columns or "LOW" not in merged.columns:
        raise KeyError("В файле свечей должны быть колонки HIGH и LOW")

    mask_nan = merged["action_price"].isna()
    mask_neg = mask_nan & (merged["prev_position"] < 0)
    mask_nonneg = mask_nan & (merged["prev_position"] >= 0)
    merged.loc[mask_neg, "action_price"] = merged.loc[mask_neg, "HIGH"]
    merged.loc[mask_nonneg, "action_price"] = merged.loc[mask_nonneg, "LOW"]

    # ������� �������� action_price -> deal_price ������ ��� �������� �������
    position_changed = merged["current_pos"] != merged["prev_position"]
    merged["deal_price"] = merged["action_price"].where(position_changed)

    # prev_action_price = action_price.shift(1), top = current action_price (т.е. first value)
    merged["prev_action_price"] = merged["action_price"].shift(1)
    if not merged["action_price"].empty:
        merged.iloc[0, merged.columns.get_loc("prev_action_price")] = merged.iloc[0][
            "action_price"
        ]

    # delta_price = action_price - prev_action_price
    merged["delta_price"] = merged["action_price"] - merged["prev_action_price"]

    # comis_count = abs(trade_size) * comis
    merged["comis_count"] = (
        (merged["current_pos"] - merged["prev_position"]).abs() * comis
    )

    # Pnl = prev_position * delta_price * 1_step_price - commission
    step_col = None
    for candidate in ("1_step_price", "1_step_price ", "one_step_price"):
        if candidate in merged.columns:
            step_col = candidate
            break
    if step_col is None:
        raise KeyError(
            "В файле свечей не найдена колонка с ценой шага (ожидалось '1_step_price')"
        )

    merged["Pnl"] = (
        merged["prev_position"] * merged["delta_price"] * merged[step_col]
        - merged["comis_count"].fillna(0.0)
    )

    # Equity = Pnl.cumsum()
    merged["Equity"] = merged["Pnl"].fillna(0).cumsum()

    return merged
