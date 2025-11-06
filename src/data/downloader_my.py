from pathlib import Path

import pandas as pd
import requests


def download_moex_securities_data() -> str:
    """
    Download MOEX futures securities reference and store it as CSV.

    Returns:
        Path to the saved CSV file.
    """
    base = "https://iss.moex.com/iss/engines/futures/markets/forts/boards/" "RFUD/"
    endpoint = "securities.json"
    url = base + endpoint

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()

    columns = payload["securities"]["columns"]
    records = payload["securities"]["data"]
    securities = pd.DataFrame(records, columns=columns)

    securities["SEC_NAME"] = securities["SHORTNAME"].str.split("-").str[0]
    securities["STEPPRICE"] = pd.to_numeric(securities["STEPPRICE"], errors="coerce")
    securities["MINSTEP"] = pd.to_numeric(securities["MINSTEP"], errors="coerce")
    securities["one_step_price"] = securities["STEPPRICE"] / securities["MINSTEP"]

    data_dir = Path(__file__).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "moex_tickers_steps.csv"
    securities.to_csv(csv_path, index=False)

    print(f"Data successfully downloaded and saved to {csv_path}")
    return str(csv_path)


if __name__ == "__main__":
    download_moex_securities_data()
