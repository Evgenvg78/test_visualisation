"""Файловые сервисы: поиск логов сделок с кешем на диске."""

from __future__ import annotations

import fnmatch
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

CacheKey = Tuple[str, str, Tuple[Tuple[str, float], ...]]

CACHE_PATH = Path("help_data") / "discover_logs_cache.json"
DEFAULT_MASK = "*t.csv"


@dataclass(frozen=True)
class LogFileInfo:
    """Описание найденного файла лога."""

    path: Path
    mtime: float
    size: int
    ticker: str | None = None


def _guess_ticker(name: str) -> str | None:
    stem = Path(name).stem
    parts = [p for p in stem.replace("-", "_").split("_") if p]
    if not parts:
        return None
    # Пытаемся пропустить технические префиксы вроде T_
    first = parts[1] if len(parts) > 1 and parts[0].lower() in {"t", "log"} else parts[0]
    return first.upper()


def _list_files(folder: Path, mask: str) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    pattern = mask.strip() or "*"
    files = [
        p.resolve()
        for p in folder.iterdir()
        if p.is_file() and fnmatch.fnmatch(p.name, pattern)
    ]
    return sorted(set(files))


def _build_mtime_snapshot(files: Sequence[Path]) -> Tuple[Tuple[str, float], ...]:
    snapshot: list[Tuple[str, float]] = []
    for path in files:
        try:
            stat = path.stat()
            snapshot.append((str(path), stat.st_mtime))
        except OSError:
            continue
    return tuple(snapshot)


def _load_cache() -> tuple[CacheKey, tuple[LogFileInfo, ...]] | None:
    if not CACHE_PATH.exists():
        return None
    try:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        key = (
            raw["path"],
            raw["mask"],
            tuple(tuple(item) for item in raw.get("mtimes", [])),
        )
        files = tuple(
            LogFileInfo(
                path=Path(item["path"]),
                mtime=float(item["mtime"]),
                size=int(item["size"]),
                ticker=item.get("ticker"),
            )
            for item in raw.get("files", [])
        )
        return key, files
    except Exception:
        return None


def _save_cache(key: CacheKey, files: Iterable[LogFileInfo]) -> None:
    payload = {
        "path": key[0],
        "mask": key[1],
        "mtimes": list(key[2]),
        "files": [
            {
                "path": str(info.path),
                "mtime": info.mtime,
                "size": info.size,
                "ticker": info.ticker,
            }
            for info in files
        ],
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def discover_logs(path: Path | str, mask: str = DEFAULT_MASK) -> tuple[LogFileInfo, ...]:
    """
    Находит файлы логов по маске и кеширует результат.

    Ключ кеша: (path, mask, список (file, mtime)). Если путь/маска/mtime совпали
    с кешем, возвращается сохранённый результат без повторного обхода диска.
    """

    folder = Path(path).expanduser().resolve()
    files = _list_files(folder, mask)
    mtimes = _build_mtime_snapshot(files)
    key: CacheKey = (str(folder), mask, mtimes)

    cached = _load_cache()
    if cached is not None:
        cached_key, cached_files = cached
        if cached_key == key:
            return cached_files

    infos: list[LogFileInfo] = []
    for file_path in files:
        try:
            stat = file_path.stat()
        except OSError:
            continue
        infos.append(
            LogFileInfo(
                path=file_path,
                mtime=stat.st_mtime,
                size=stat.st_size,
                ticker=_guess_ticker(file_path.name),
            )
        )

    _save_cache(key, infos)
    return tuple(infos)


__all__ = ["LogFileInfo", "discover_logs"]
