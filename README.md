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

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
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

## Dependencies

- pandas: Data processing
- requests: HTTP requests to MOEX API
- pytest: Testing framework