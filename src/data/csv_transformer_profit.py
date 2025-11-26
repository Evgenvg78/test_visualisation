"""Модуль для чтения и преобразования CSV с форматами, как в
``test_result_SP.csv``.

Функция:
        load_transform_csv(file_path, save_csv=None, save_encoding='cp1251') -> pandas.DataFrame

Поведение:
 - читает CSV с encoding='cp1251' и sep=';'
 - оставляет только столбцы: 'Дата и время' или его аналоги,
     'Цена последней сделки' / 'Цена', 'Текущее количество контрактов'
 - преобразует:
         'Дата и время' -> 'date_time' (pd.Timestamp), округлённая вниз до минуты
         'Цена последней сделки' / 'Цена' -> float (2 знака после запятой)
         'Текущее количество контрактов' -> int
 - при указании ``save_csv`` сохраняет результат в CSV с указанной кодировкой
     и sep=';'.

Замечания/предположения:
 - Для парсинга даты используется ``dayfirst=True`` (обычный для данных формата
     d/m/Y). Если в ваших данных иной формат, передайте строку в ISO или
     измените код.
 - В числовых полях десятичный разделитель может быть запятой — модуль
     автоматически заменяет запятые на точки перед преобразованием.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

import pandas as pd

__all__ = ["load_transform_csv"]

_COLUMN_ALIASES = {
    "date_time": (
        "Дата и время",
        "Дата и Время",
    ),
    "price": (
        "Цена последней сделки",
        "Цена",
    ),
    "current_pos": (
        "Текущее количество контрактов",
        "Текущее количество контрактов, шт",
    ),
    "position_profit": (
        "Доход позиции",
        "Доход",
    ),
}


def _normalize_column_name(name: str) -> str:
    """Нормализует имя столбца для сравнения."""
    return " ".join(name.strip().casefold().split())


def _resolve_column(columns: Iterable[str], candidates: Iterable[str]) -> str:
    """Возвращает имя первого найденного столбца из числа кандидатов."""
    normalized_lookup = {_normalize_column_name(col): col for col in columns}
    for candidate in candidates:
        normalized_candidate = _normalize_column_name(candidate)
        if normalized_candidate in normalized_lookup:
            return normalized_lookup[normalized_candidate]
    raise ValueError(f"Не удалось найти ни один из вариантов колонок: {tuple(candidates)}")


def load_transform_csv(
    file_path: Union[str, Path],
    save_csv: Optional[Union[str, Path]] = None,
    save_encoding: str = "cp1251",
) -> pd.DataFrame:
    """Прочитать CSV и выполнить требуемые преобразования.

    Args:
        file_path: путь к исходному CSV. Ожидается кодировка cp1251 и разделитель ';'.
        save_csv: если указан, сохраняет результирующий DataFrame в этот путь (sep=';', кодировка как в save_encoding).
        save_encoding: кодировка для сохранения (по умолчанию 'cp1251').

Returns:
        pandas.DataFrame с унифицированными столбцами ['date_time', 'price', 'current_pos', 'position_profit']
        где 'date_time' в pd.Timestamp округлён до минуты; 'price' -> float; 'current_pos' -> int;
        'position_profit' -> float (реально сохранённые значения только в последней строке и строках с нулевой позицией, остальные обнуляются).

    Raises:
        ValueError: если в входном CSV отсутствуют необходимые столбцы.
    """
    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    # Читаем CSV с предустановленной кодировкой и разделителем
    # ????????? CSV, ?????? ????????? ????????? ? ???????? ??????? ???????
    df = None
    resolved_columns = {}
    last_missing_error = None
    for encoding in ("cp1251", "utf-8", "utf-8-sig"):
        try:
            candidate = pd.read_csv(src, encoding=encoding, sep=";")
        except UnicodeDecodeError:
            continue

        resolved_columns = {}
        missing_variants = {}
        for key, aliases in _COLUMN_ALIASES.items():
            try:
                resolved_columns[key] = _resolve_column(candidate.columns, aliases)
            except ValueError:
                missing_variants[key] = aliases

        if missing_variants:
            details = "; ".join(
                f"{key}: {', '.join(variants)}" for key, variants in missing_variants.items()
            )
            last_missing_error = ValueError(
                f"?? ??????? ???????????? ???????. ????????? -> {details}"
            )
            continue

        df = candidate
        break

    if df is None:
        if last_missing_error is not None:
            raise last_missing_error
        raise UnicodeDecodeError(
            "csv_transformer_profit",
            b"",
            0,
            1,
            "Failed to decode CSV using supported encodings",
        )

    date_col = resolved_columns["date_time"]
    price_col = resolved_columns["price"]
    current_pos_col = resolved_columns["current_pos"]
    profit_col = resolved_columns["position_profit"]

    # Оставляем только найденные столбцы (в указанном порядке)
    df = df[[date_col, price_col, current_pos_col, profit_col]].copy()

    # 1) Дата и время -> date_time; округление вниз до минуты (часто такой шаг нужнее всего)
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df[date_col] = df[date_col].dt.floor("min")  # 'T' == minute

    # 2) Цена -> float (2 знака)
    df[price_col] = (
        df[price_col]
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.replace(",", ".")
    )
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce").round(2)

    # 3) Доход позиции -> float
    df[profit_col] = (
        df[profit_col]
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.replace(",", ".")
    )
    df[profit_col] = pd.to_numeric(df[profit_col], errors="coerce").fillna(0.0)

    # 4) Текущее количество контрактов -> integer
    cleaned_current_pos = (
        df[current_pos_col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^\d\.\-]", "", regex=True)
    )
    numeric_current_pos = pd.to_numeric(cleaned_current_pos, errors="coerce").fillna(0.0)
    df[current_pos_col] = numeric_current_pos.round().astype(int)

    # Приводим к унифицированным именам столбцов
    df = df.rename(
        columns={
            date_col: "date_time",
            price_col: "price",
            current_pos_col: "current_pos",
            profit_col: "position_profit",
        }
    )

    result = df[["date_time", "price", "current_pos", "position_profit"]].copy()

    if not result.empty:
        # Определяем актуальный фрагмент данных: ищем скачок назад во времени и берём последнюю "партию"
        date_time_series = result["date_time"]
        new_chunk_mask = date_time_series < date_time_series.shift()
        test_ids = new_chunk_mask.cumsum() + 1
        last_test_id = int(test_ids.max())
        result = result.loc[test_ids == last_test_id].copy()

        if not result.empty:
            # Доход позиции сохраняем только для последней строки и строк с нулевой позицией
            last_index = result.index[-1]
            keep_profit_mask = (result["current_pos"] == 0) | (result.index == last_index)
            result.loc[~keep_profit_mask, "position_profit"] = 0.0

    if save_csv:
        out_path = Path(save_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False, encoding=save_encoding, sep=";")

    return result
