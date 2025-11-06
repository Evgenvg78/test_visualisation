"""Модуль для чтения и преобразования CSV с форматом, как в
``test_result_SP.csv``.

Функция:
        load_transform_csv(file_path, save_csv=None, save_encoding='cp1251') -> pandas.DataFrame

Поведение:
 - читает CSV с encoding='cp1251' и sep=';'
 - оставляет только столбцы: 'Дата и время', 'Цена последней сделки',
     'Текущее количество контрактов'
 - преобразует:
         'Дата и время' -> 'date_time' (pd.Timestamp), округлённая вниз до минуты
         'Цена последней сделки' -> float (2 знака после запятой)
         'Текущее количество контрактов' -> int
 - при указании ``save_csv`` сохраняет результат в CSV с указанной кодировкой
     и sep=';'.

Замечания/предположения:
 - Для парсинга даты используется ``dayfirst=True`` (обычный для России порядок
     d/m/Y). Если в ваших данных иной формат, передайте строку в ISO или
     измените код.
 - В числовых полях десятичный разделитель может быть запятой — модуль
     автоматически заменяет запятые на точки перед преобразованием.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

__all__ = ["load_transform_csv"]


def load_transform_csv(
    file_path: Union[str, Path],
    save_csv: Optional[Union[str, Path]] = None,
    save_encoding: str = "cp1251",
) -> pd.DataFrame:
    """Прочитать CSV и выполнить требуемые преобразования.

    Args:
        file_path: путь к исходному CSV. Ожидается кодировка cp1251 и разделитель ';'.
        save_csv: если указан, сохранить результирующий DataFrame в этот путь (sep=';', кодировка как в save_encoding).
        save_encoding: кодировка для сохранения (по умолчанию 'cp1251').

    Returns:
        pd.DataFrame с унифицированными колонками ['date_time', 'price', 'current_pos']
        где 'date_time' — pd.Timestamp округлённый вниз до минуты; 'price' -> float; 'current_pos' -> int.

    Raises:
        ValueError: если в исходном CSV отсутствуют необходимые столбцы.
    """
    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    # Читать CSV с указанной кодировкой и разделителем
    df = pd.read_csv(src, encoding="cp1251", sep=";")

    expected = [
        "Дата и время",
        "Цена последней сделки",
        "Текущее количество контрактов",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют необходимые столбцы в CSV: {missing}")

    # Оставляем только нужные колонки (в исходных именах)
    df = df[expected].copy()

    # 1) Дата и время -> date_time; парсим, затем округляем вниз до минуты (удаляем секунды)
    # Используем dayfirst=True (предположение: d.m.Y или d/m/Y). errors='coerce' для безопасного парсинга.
    df["Дата и время"] = pd.to_datetime(
        df["Дата и время"], dayfirst=True, errors="coerce"
    )
    df["Дата и время"] = df["Дата и время"].dt.floor("T")  # 'T' == minute

    # 2) Цена последней сделки -> float (2 знака)
    # Убираем пробелы, заменяем запятую на точку
    df["Цена последней сделки"] = (
        df["Цена последней сделки"]
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.replace(",", ".")
    )
    df["Цена последней сделки"] = pd.to_numeric(
        df["Цена последней сделки"], errors="coerce"
    ).round(2)

    # 3) Текущее количество контрактов -> integer
    # Убираем всё, кроме цифр и знака минус (на случай отрицательных значений)
    df["Текущее количество контрактов"] = (
        df["Текущее количество контрактов"]
        .astype(str)
        .str.replace(r"[^\d\-]", "", regex=True)
    )
    # Преобразуем в число; при ошибках используем NaN -> затем приводим к int (с заполнением 0)
    df["Текущее количество контрактов"] = (
        pd.to_numeric(df["Текущее количество контрактов"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # Создадим унифицированные имена колонок: date_time, price, current_pos
    df = df.rename(
        columns={
            "Дата и время": "date_time",
            "Цена последней сделки": "price",
            "Текущее количество контрактов": "current_pos",
        }
    )

    # Внутренне date_time уже pd.Timestamp округлённый до минуты
    result = df[["date_time", "price", "current_pos"]].copy()

    if not result.empty:
        # Определяем границы тестов: новый участок начинается, когда время откатывается назад
        date_time_series = result["date_time"]
        new_chunk_mask = date_time_series < date_time_series.shift()
        test_ids = new_chunk_mask.cumsum() + 1
        last_test_id = int(test_ids.max())
        result = result.loc[test_ids == last_test_id].copy()

    # Опционально сохраняем
    if save_csv:
        out_path = Path(save_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False, encoding=save_encoding, sep=";")

    return result
