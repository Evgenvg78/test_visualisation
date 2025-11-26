from __future__ import annotations

import io
from pathlib import Path

import app
from src.services.files import LogFileInfo


def test_merge_log_infos_prefers_latest_metadata(tmp_path: Path):
    base = tmp_path
    info_old = LogFileInfo(path=base / "a.csv", mtime=1.0, size=10, ticker="AAA")
    info_new = LogFileInfo(path=base / "a.csv", mtime=2.0, size=20, ticker="AAA")
    info_other = LogFileInfo(path=base / "b.csv", mtime=3.0, size=5, ticker="BBB")

    merged = app._merge_log_infos([info_old], [info_new, info_other])

    assert merged[0] == info_new
    assert merged[1] == info_other


def test_build_log_descriptors_returns_resolved_paths(tmp_path: Path):
    info = LogFileInfo(path=tmp_path / "relative" / "log.csv", mtime=1.23, size=1)
    info.path.parent.mkdir(parents=True, exist_ok=True)
    info.path.write_text("data", encoding="utf-8")

    descriptors = app._build_log_descriptors([info])

    assert descriptors == ((str(info.path.resolve()), info.mtime),)


def test_session_state_helpers_round_trip(monkeypatch):
    monkeypatch.setattr(app.st, "session_state", {})
    infos = [
        LogFileInfo(path=Path("a.csv"), mtime=1.0, size=1),
        LogFileInfo(path=Path("b.csv"), mtime=2.0, size=2),
    ]

    app._set_log_infos(infos)
    assert app._get_log_infos() == infos


def test_process_logs_invokes_build_equity_bundle(monkeypatch, tmp_path: Path):
    log_paths = [tmp_path / "a.t.csv", tmp_path / "b.t.csv"]
    for path in log_paths:
        path.write_text("time;price\n", encoding="utf-8")

    called = {}

    def _fake_bundle(paths, **kwargs):
        called["paths"] = list(paths)
        called["kwargs"] = kwargs
        return "result"

    monkeypatch.setattr(app, "build_equity_bundle", _fake_bundle)

    result = app._process_logs(log_paths, commission=0.25)

    assert result == "result"
    assert called["paths"] == [Path(p) for p in log_paths]
    assert called["kwargs"]["comis"] == 0.25
    assert called["kwargs"]["build_plot"] is True
    cache_key = ";".join(str(p.resolve()) for p in log_paths)
    assert called["kwargs"]["cache_key"] == cache_key


class _DummyUpload:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._buffer = io.BytesIO(data)

    def getbuffer(self):
        return self._buffer.getbuffer()


def test_persist_uploaded_files_writes_buffers(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(app, "UPLOAD_DIR", tmp_path / "uploads")
    uploads = [_DummyUpload("test.csv", b"col1\n1\n")]

    saved = app._persist_uploaded_files(uploads)

    assert len(saved) == 1
    assert saved[0].exists()
    assert saved[0].read_text(encoding="utf-8") == "col1\n1\n"
