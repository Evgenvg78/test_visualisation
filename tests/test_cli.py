from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts import combine_equity


def test_discover_log_paths_removes_duplicates(tmp_path: Path):
    folder = tmp_path / "logs"
    folder.mkdir()
    file_a = folder / "a.t.csv"
    file_b = folder / "b.t.csv"
    file_a.write_text("a")
    file_b.write_text("b")

    def _stub(folder_arg: Path, mask_arg: str):
        return [
            SimpleNamespace(path=file_a),
            SimpleNamespace(path=file_b),
            SimpleNamespace(path=file_a),
        ]

    paths = combine_equity.discover_log_paths(
        [str(folder)],
        mask_override="*.t.csv",
        discover_logs_fn=_stub,
        default_mask="*t.csv",
    )

    assert paths == [file_a.resolve(), file_b.resolve()]


def test_main_invokes_services_and_writes_outputs(tmp_path: Path, monkeypatch):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    log_path = logs_dir / "log1.t.csv"
    log_path.write_text("time;price\n", encoding="utf-8")

    combined_df = pd.DataFrame(
        {
            "date_time": pd.date_range("2024-01-01 00:00:00", periods=2, freq="1min"),
            "Equity": [0.0, 10.0],
        }
    )
    metrics = SimpleNamespace(
        go_requirement=1000.0,
        final_equity=10.0,
        return_percent=1.0,
        drawdown_currency=2.0,
        drawdown_percent_of_go=0.2,
        drawdown_start=pd.Timestamp("2024-01-01 00:00:00"),
        drawdown_end=pd.Timestamp("2024-01-01 00:01:00"),
        longest_drawdown_minutes=5.0,
        longest_drawdown_start=pd.Timestamp("2024-01-01 00:00:00"),
        longest_drawdown_end=pd.Timestamp("2024-01-01 00:05:00"),
        total_commission=0.5,
        start_time=pd.Timestamp("2024-01-01 00:00:00"),
        end_time=pd.Timestamp("2024-01-01 00:01:00"),
    )
    per_logs = [
        SimpleNamespace(
            log_path=log_path,
            ticker="LOG1",
            equity=combined_df.copy(),
        )
    ]
    result = SimpleNamespace(
        combined=combined_df,
        metrics=metrics,
        per_logs=per_logs,
        figure=None,
        plot_title="Combined Equity",
    )

    bundle_calls = {}

    def _fake_build(log_paths, **kwargs):
        bundle_calls["paths"] = list(log_paths)
        bundle_calls["kwargs"] = kwargs
        return result

    def _fake_discover(folder, mask):
        assert folder == logs_dir.resolve()
        assert mask == "*.t.csv"
        return [SimpleNamespace(path=log_path)]

    def _fake_load_services():
        return _fake_build, _fake_discover, "*.t.csv"

    export_calls = []

    def _fake_export(res, html_path, png_path):
        export_calls.append((html_path, png_path))
        if html_path:
            html_path.write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(combine_equity, "_load_services", _fake_load_services)
    monkeypatch.setattr(combine_equity, "export_plots", _fake_export)

    out_csv = tmp_path / "combined.csv"
    metrics_path = tmp_path / "combined.metrics.json"
    html_path = tmp_path / "combined.html"

    exit_code = combine_equity.main(
        [
            "--logs",
            str(logs_dir),
            "--mask",
            "*.t.csv",
            "--comis",
            "0.5",
            "--out",
            str(out_csv),
            "--html",
            str(html_path),
        ]
    )

    assert exit_code == 0
    assert out_csv.exists()
    assert metrics_path.exists()
    assert html_path.exists()
    assert export_calls == [(html_path, None)]
    assert [p.resolve() for p in bundle_calls["paths"]] == [log_path.resolve()]
    assert bundle_calls["kwargs"]["comis"] == 0.5
