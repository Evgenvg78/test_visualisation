"""Interactive file selector helpers for Jupyter notebooks.

Typical usage inside a notebook::

    from trend_indicator_module.examples.file_selector import (
        load_saved_selection,
        select_files,
    )

    DATA_DIR = r"G:\\My Drive\\data_fut"
    tickers = load_saved_selection()
    select_files(DATA_DIR)

The last chosen tickers are stored in JSON so you can restore them later.
"""
from __future__ import annotations

import json
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

try:
    import ipywidgets as widgets
    from IPython.display import clear_output, display
except Exception:  # pragma: no cover - widget import handled at runtime
    widgets = None
    clear_output = None
    display = None

STORAGE_FILENAME = "last_selected_tickers.json"


def _resolve_storage_path(storage_path: Optional[Union[str, Path]]) -> Path:
    if storage_path is None:
        return Path(__file__).with_name(STORAGE_FILENAME)
    return Path(storage_path)


def _normalize_pattern(pattern: str) -> str:
    stripped = pattern.strip().lower()
    if not stripped:
        return "*"
    if any(char in stripped for char in "*?[]"):
        return stripped
    if not stripped.startswith("*"):
        return f"*{stripped}"
    return stripped


def _list_files(data_dir: Union[str, Path], extensions: Sequence[str] = (".txt", ".csv")) -> List[Path]:
    root = Path(data_dir)
    if not root.exists():
        return []
    patterns = tuple(_normalize_pattern(ext) for ext in extensions)
    files = [
        item
        for item in root.iterdir()
        if item.is_file() and any(fnmatch(item.name.lower(), pattern) for pattern in patterns)
    ]
    files.sort()
    return files


def list_matching_files(data_dir: Union[str, Path], extensions: Sequence[str] = ("*t.csv",)) -> List[Path]:
    """Return a sorted list of files in ``data_dir`` matching the provided patterns."""
    return _list_files(data_dir, extensions=extensions)


def load_saved_selection(storage_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Return previously saved tickers (filenames without extensions)."""
    path = _resolve_storage_path(storage_path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        return [str(item) for item in data]
    return []


def get_saved_selection(storage_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Backward-compatible alias for loading the saved selection."""
    return load_saved_selection(storage_path)


def get_selected_files(storage_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Return the most recently saved selection."""
    return load_saved_selection(storage_path)


def save_selection(tickers: Iterable[str], storage_path: Optional[Union[str, Path]] = None) -> None:
    """Persist tickers to JSON so they can be reused on the next run."""
    path = _resolve_storage_path(storage_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(list(tickers), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # Saving is a convenience feature only; swallow errors silently.
        pass


def select_files(
    data_dir: Union[str, Path],
    title: str = "Select tickers",
    storage_path: Optional[Union[str, Path]] = None,
    extensions: Sequence[str] = (".txt", "*t.csv"),
) -> List[str]:
    """Render a checkbox selector for files in ``data_dir``.

    Returns the latest saved selection so that the notebook can proceed immediately,
    and exposes ``get_selected_files()`` for accessing the live selection after
    pressing the button inside the widget.
    """
    files = _list_files(data_dir, extensions=extensions)
    if not files:
        print(f"No files found in {data_dir}")
        return []

    saved_selection = load_saved_selection(storage_path)

    if widgets is None or display is None or clear_output is None:
        print("ipywidgets is not available. Fallback to text prompt.")
        for idx, file_path in enumerate(files, 1):
            print(f"{idx}: {file_path.name}")
        if saved_selection:
            print(f"Saved selection: {', '.join(saved_selection)}")
        selection = input("Enter comma-separated indices (leave blank to keep saved selection): ").strip()
        if not selection:
            return saved_selection
        try:
            indices = [int(value.strip()) - 1 for value in selection.split(",") if value.strip()]
        except Exception:
            return []
        chosen = [files[index] for index in indices if 0 <= index < len(files)]
        chosen_no_ext = [path.stem for path in chosen]
        save_selection(chosen_no_ext, storage_path)
        return chosen_no_ext

    saved_set = set(saved_selection)
    checkboxes = [widgets.Checkbox(value=file_path.stem in saved_set, description=file_path.name) for file_path in files]
    button = widgets.Button(description="Save selection", button_style="primary")
    output = widgets.Output()
    container = widgets.VBox([widgets.Label(title)] + checkboxes + [button, output])

    selected: dict[str, List[str]] = {"names": saved_selection}

    def on_click(_: widgets.Button) -> None:
        chosen = [cb.description for cb in checkboxes if cb.value]
        chosen_no_ext = [Path(name).stem for name in chosen]
        selected["names"] = chosen_no_ext
        if chosen_no_ext:
            save_selection(chosen_no_ext, storage_path)
        with output:
            clear_output()
            if chosen_no_ext:
                print("Saved tickers:")
                for ticker in chosen_no_ext:
                    print(f"- {ticker}")
            else:
                print("No tickers selected.")

    button.on_click(on_click)
    display(container)

    def get_selected() -> List[str]:
        return selected["names"]

    globals()["get_selected_files"] = get_selected

    print("Press 'Save selection' to persist your choice.")
    print("Call get_selected_files() after saving to retrieve the updated list.")

    return saved_selection


__all__ = [
    "get_saved_selection",
    "get_selected_files",
    "load_saved_selection",
    "save_selection",
    "select_files",
    "list_matching_files",
]
