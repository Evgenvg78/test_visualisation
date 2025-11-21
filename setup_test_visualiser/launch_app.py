from __future__ import annotations

import os
import shutil
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _is_frozen() -> bool:
    return hasattr(sys, "_MEIPASS")


def _bundle_root() -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    candidate = base / "app_bundle"
    return candidate if candidate.exists() else Path(__file__).resolve().parents[1]


def _data_root() -> Path:
    if not _is_frozen():
        return Path(__file__).resolve().parents[1]
    base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    target = base / "test_visualiser"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _sync_help_data(bundle: Path, runtime: Path) -> None:
    src = bundle / "help_data"
    dst = runtime / "help_data"
    if not src.exists():
        dst.mkdir(parents=True, exist_ok=True)
        return
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)


def _prepare_workspace() -> tuple[Path, Path]:
    bundle = _bundle_root()
    runtime = _data_root()
    if _is_frozen():
        _sync_help_data(bundle, runtime)
    return bundle, runtime


def _launch_streamlit(script: Path, port: str) -> None:
    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(script),
        f"--server.port={port}",
        "--browser.serverAddress=127.0.0.1",
        "--server.headless=false",
        "--global.developmentMode=false",
    ]

    # def opener() -> None:
    #     time.sleep(1.5)
    #     try:
    #         webbrowser.open(f"http://localhost:{port}", new=2)
    #     except Exception:
    #         pass

    # threading.Thread(target=opener, daemon=True).start()
    stcli.main()


def main() -> None:
    bundle, data_root = _prepare_workspace()
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_FILEWATCHER_TYPE", "poll")
    os.environ.setdefault("STREAMLIT_SERVER_ENABLECORS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION", "false")

    os.chdir(data_root)
    sys.path.insert(0, str(bundle))
    sys.path.insert(0, str(bundle / "src"))
    script_path = bundle / "app.py"
    port = os.environ.get("TEST_VISUALISER_PORT", "8501")
    _launch_streamlit(script_path, port)


if __name__ == "__main__":
    main()
