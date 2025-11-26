"""Pytest fixtures and test-time shims for the equity toolkit."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, List, Optional

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _FakeTicker:
    """Deterministic stand-in for moexalgo.Ticker used in tests."""

    calls: List[dict] = []
    responses: List[pd.DataFrame | Callable[..., pd.DataFrame]] = []

    def __init__(self, ticker: str, board: Optional[str] = None) -> None:
        self.ticker = ticker
        self.board = board

    def candles(self, start: str, end: str, period: int = 1) -> pd.DataFrame:
        _FakeTicker.calls.append(
            {"ticker": self.ticker, "board": self.board, "start": start, "end": end, "period": period}
        )
        if not _FakeTicker.responses:
            return pd.DataFrame()
        response = _FakeTicker.responses.pop(0)
        frame = response(self, start, end, period) if callable(response) else response
        return frame.copy()

    @classmethod
    def queue_response(cls, frame_or_factory: pd.DataFrame | Callable[..., pd.DataFrame]) -> None:
        cls.responses.append(frame_or_factory)

    @classmethod
    def reset(cls) -> None:
        cls.calls.clear()
        cls.responses.clear()


fake_moexalgo = types.ModuleType("moexalgo")
fake_moexalgo.Ticker = _FakeTicker
sys.modules["moexalgo"] = fake_moexalgo

# --- Provide a lightweight Streamlit stub if the real package is missing ---
try:
    import streamlit as st  # type: ignore
except ModuleNotFoundError:
    streamlit_stub = types.ModuleType("streamlit")
    streamlit_stub.session_state = {}

    class _DummyProgress:
        def progress(self, value):
            return None

    class _DummyStatus:
        def info(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

        def success(self, *args, **kwargs):
            return None

    class _DummyExpander:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyColumn:
        def metric(self, *args, **kwargs):
            return None

        def plotly_chart(self, *args, **kwargs):
            return None

        def markdown(self, *args, **kwargs):
            return None

        def download_button(self, *args, **kwargs):
            return None

        def info(self, *args, **kwargs):
            return None

    def _cache_data_decorator(*d_args, **d_kwargs):
        def _wrap(func):
            def _inner(*args, **kwargs):
                return func(*args, **kwargs)

            return _inner

        return _wrap

    def _noop(*args, **kwargs):
        return None

    def _columns(spec, **kwargs):
        length = spec if isinstance(spec, int) else len(spec)
        return [_DummyColumn() for _ in range(length)]

    streamlit_stub.cache_data = _cache_data_decorator
    streamlit_stub.file_uploader = lambda *args, **kwargs: []
    streamlit_stub.button = lambda *args, **kwargs: False
    streamlit_stub.text_input = _noop
    streamlit_stub.number_input = _noop
    streamlit_stub.selectbox = lambda *args, **kwargs: kwargs.get("options", [None])[0]
    streamlit_stub.empty = lambda *args, **kwargs: _DummyStatus()
    streamlit_stub.progress = lambda *args, **kwargs: _DummyProgress()
    streamlit_stub.columns = _columns
    streamlit_stub.metric = _noop
    streamlit_stub.subheader = _noop
    streamlit_stub.title = _noop
    streamlit_stub.caption = _noop
    streamlit_stub.plotly_chart = _noop
    streamlit_stub.warning = _noop
    streamlit_stub.info = _noop
    streamlit_stub.success = _noop
    streamlit_stub.error = _noop
    streamlit_stub.download_button = _noop
    streamlit_stub.expander = lambda *args, **kwargs: _DummyExpander()
    streamlit_stub.write = _noop
    streamlit_stub.set_page_config = _noop
    streamlit_stub.header = _noop
    streamlit_stub.multiselect = lambda *args, **kwargs: []
    sys.modules["streamlit"] = streamlit_stub


@pytest.fixture(autouse=True)
def _reset_fake_ticker() -> Iterable[None]:
    _FakeTicker.reset()
    yield
    _FakeTicker.reset()


@pytest.fixture
def fake_ticker_class() -> Callable[..., _FakeTicker]:
    """Expose the fake ticker class to tests that need to customize responses."""

    return _FakeTicker
