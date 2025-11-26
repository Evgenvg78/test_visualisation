# Минимальный комплект для переноса визуализатора

Этот документ описывает, какие зависимости и файлы достаточно взять с собой, чтобы развернуть Streamlit-визуализатор на чистой Windows-машине.

## Необходимые программы и системные требования
- **Windows 10/11 x64** с правами на установку ПО.
- **Python 3.11+ (64-bit)** установленный в `PATH`. Проект собирался и тестировался с CPython, поэтому другой интерпретатор не требуется.
- **pip** входит в поставку Python; нужен доступ в интернет для загрузки колёс pandas/streamlit.
- **Браузер** (Edge/Chrome) — Streamlit открывает интерфейс по `http://localhost:8501`.
- (Опционально) **PowerShell 5+** либо классический `cmd.exe` для запуска `.bat` скриптов.

## Минимальные каталоги/файлы проекта
| Путь | Назначение |
| --- | --- |
| `app.py` | Главный Streamlit-дэшборд. |
| `src/` | Модули расчётов: комбинирование equity, загрузка логов, кэширование. |
| `src/services/` | Обёртки над файловой системой и подготовкой данных; без них UI не запустится. |
| `help_data/` | Настройки дэшборда и кэшированные выборки (`dashboard_state.json`). Папка создаётся автоматически, но для портирования удобно включить шаблон. |
| `scripts/run_streamlit_dashboard.py` | Обёртка, которую вызываем из `.bat`, чтобы не требовать `streamlit` CLI. |
| `requirements-base.txt`, `requirements-ui.txt` | Минимальный список пакетов. Для установки UI достаточно `pip install -r requirements-ui.txt`. |
| `README.md` (корневой) | Быстрые команды и структура проекта. |
| `pyinstaller/` (опционально) | Нужен только если планируется собирать самодостаточные EXE через PyInstaller. |

Каталоги `OHCL/`, `trade_data/`, `out/`, `build/`, `dist/`, `.pytest_cache/`, `__pycache__/` переносить не требуется — это данные пользователя и артефакты сборок.

## Процесс переноса
1. Запакуйте перечисленные каталоги/файлы в архив (либо git-репозиторий) и перенесите на целевой ПК.
2. Oткройте PowerShell в корне проекта и выполните:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements-ui.txt
   ```
3. Для запуска используйте:
   ```powershell
   .\.venv\Scripts\python.exe scripts\run_streamlit_dashboard.py
   ```
   или заверните эту команду в `visualisation.bat`.

## Заготовки для .bat файлов
`install_visualisation.bat`:
```bat
@echo off
python -m venv .venv || goto :error
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-ui.txt || goto :error
echo Зависимости установлены.
goto :eof
:error
echo Установка завершилась с ошибкой.
exit /b 1
```

`visualisation.bat`:
```bat
@echo off
if not exist .venv\Scripts\python.exe (
  echo Сначала запустите install_visualisation.bat
  exit /b 1
)
call .venv\Scripts\activate
python scripts\run_streamlit_dashboard.py
```

Эти два файла можно держать в корне проекта; они покрывают как установку зависимостей, так и запуск дэшборда в браузере.
