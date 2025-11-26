from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import csv_transformer_profit as transformer


def _header(key: str, variant: int = 0) -> str:
    return transformer._COLUMN_ALIASES[key][variant]


def test_load_transform_csv_cp1251(tmp_path: Path):
    headers = [
        _header("date_time"),
        _header("price"),
        _header("current_pos"),
        _header("position_profit"),
    ]
    content = "\n".join(
        [
            ";".join(headers),
            ";".join(["01.02.2024 11:00", "1 234,50", "1", "15,7"]),
            ";".join(["01.02.2024 12:00", "1 230,00", "0", "-5,2"]),
        ]
    )
    csv_path = tmp_path / "log_cp1251.csv"
    csv_path.write_text(content, encoding="cp1251")

    df = transformer.load_transform_csv(csv_path)

    assert list(df.columns) == ["date_time", "price", "current_pos", "position_profit"]
    assert df["price"].tolist() == [1234.5, 1230.0]
    assert df["current_pos"].tolist() == [1, 0]
    assert df["position_profit"].tolist() == [0.0, -5.2]


def test_load_transform_csv_utf8_fallback(tmp_path: Path):
    headers = [
        _header("date_time"),
        _header("price"),
        _header("current_pos"),
        _header("position_profit"),
    ]
    content = "\n".join(
        [
            ";".join(headers),
            ";".join(["2024-03-01 09:00", "1 000,00", "0", "0"]),
        ]
    )
    csv_path = tmp_path / "log_utf8.csv"
    csv_path.write_text(content, encoding="utf-8")

    df = transformer.load_transform_csv(csv_path)

    assert df["date_time"].dt.hour.tolist() == [9]
    assert df["price"].tolist() == [1000.0]
    assert df["current_pos"].tolist() == [0]


def test_load_transform_csv_missing_column_raises(tmp_path: Path):
    headers = [
        _header("date_time"),
        _header("price"),
        _header("current_pos"),
    ]
    content = "\n".join(
        [
            ";".join(headers),
            ";".join(["2024-03-01 09:00", "1 000,00", "0"]),
        ]
    )
    csv_path = tmp_path / "broken.csv"
    csv_path.write_text(content, encoding="cp1251")

    with pytest.raises(ValueError):
        transformer.load_transform_csv(csv_path)
