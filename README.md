# Test Visualization Project

This project downloads and processes financial data from the Moscow Exchange (MOEX).

## Project Structure

```
.
├── src/
│   └── data/
│       ├── downloader_my.py      # MOEX data downloader
│       └── moex_tickers_steps.csv # Downloaded data (generated)
├── tests/
│   └── test_basic.py            # Unit tests
├── requirements.txt             # Python dependencies
└── init_project.ps1             # Project initialization script
```

## Installation

1. Run the initialization script:
   ```powershell
   .\init_project.ps1
   ```

2. Установите зависимости под нужный сценарий:
   ```bash
   # Минимальное ядро (CLI/сервисы)
   pip install -r requirements-base.txt

   # Streamlit UI + CLI
   pip install -r requirements-ui.txt

   # Разработка/линтинг/сборка (включает всё выше)
   pip install -r requirements-dev.txt
   ```

## Usage

### Download MOEX Data

To download the latest securities data from MOEX:

```bash
python src/data/downloader_my.py
```

This will create a CSV file at `src/data/moex_tickers_steps.csv` with the latest securities information.

### Run Tests

```bash
python -m pytest tests/ -v
```

### PyInstaller Bundles

Готовые спецификации лежат в каталоге `pyinstaller/`:

- `pyinstaller/cli.spec` — headless CLI (`scripts/combine_equity.py`).
- `pyinstaller/streamlit.spec` — Streamlit dashboard через раннер `scripts/run_streamlit_dashboard.py`.

Сборка выполняется командой:

```bash
pyinstaller pyinstaller/cli.spec
pyinstaller pyinstaller/streamlit.spec
```

Артефакты появятся в `dist/combine_equity/` и `dist/streamlit_dashboard/`.

## Dependencies

- `requirements-base.txt`: pandas, requests, moexalgo — всё, что нужно для расчётов и CLI.
- `requirements-ui.txt`: дополняет base Streamlit'ом и Plotly.
- `requirements-dev.txt`: добавляет инструменты разработки (pytest, black, isort, flake8, PyInstaller, pre-commit).
