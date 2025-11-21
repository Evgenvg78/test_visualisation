# Runtime manifest

PyInstaller получает модульные и файловые ресурсы напрямую из корня проекта и раскладывает их внутри архива так, чтобы `launch_app.py` мог «развернуть» рабочее окружение в `%LOCALAPPDATA%\test_visualiser`.

## Python код
- `app.py` (основной Streamlit скрипт);
- вся папка `src/` (включая `data/*`, `equity_combination.py`, `moex_all_in_one_v2.py` и т.д.).

## Данные и шаблоны
- `help_data/` — используется как исходный шаблон; при запуске exe отсутствующие файлы копируются в пользовательскую папку, существующие не перезаписываются.

## Что создаётся у пользователя
- `%LOCALAPPDATA%\test_visualiser\help_data\` — рабочие данные (сюда сохраняется состояние), недостающие файлы подтягиваются из архива;
- остальные папки (`OHCL/`, `trade_data/`, экспортные CSV) также появляются внутри `%LOCALAPPDATA%\test_visualiser`.

## Зависимости
- `streamlit` (CLI и веб-сервер);
- `pandas`, `plotly`, `requests` и др. из `requirements.txt`.

При необходимости добавить новые файлы — пропиши их в `build_exe.ps1` через `--add-data` или `--collect-all`, тогда они автоматически попадут в `app_bundle`.
