"""Wrapper to launch the Streamlit dashboard without relying on the CLI binary."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    try:
        from streamlit.web import bootstrap
    except ImportError as exc:  # pragma: no cover - guarded by packaging checks
        raise SystemExit(
            "Streamlit is not installed. Install requirements-ui.txt before running the dashboard."
        ) from exc

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    # ``is_hello`` expects a bool (Streamlit 1.40 tightened this), so pass False explicitly.
    bootstrap.run(str(app_path), False, [], {})


if __name__ == "__main__":
    main()
