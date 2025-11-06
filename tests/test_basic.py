import os

import pandas as pd

from src.data.downloader_my import download_moex_securities_data


def test_download_moex_securities_data():
    """Test that MOEX securities data can be downloaded and saved correctly."""
    # Run the download function
    csv_path = download_moex_securities_data()

    # Check that the file was created
    assert os.path.exists(csv_path), "CSV file was not created"

    # Check that the file contains data
    df = pd.read_csv(csv_path)
    assert len(df) > 0, "CSV file is empty"
    assert len(df.columns) > 0, "CSV file has no columns"

    # Check that expected columns exist
    expected_columns = ["SECID", "BOARDID"]
    for col in expected_columns:
        assert col in df.columns, f"Expected column {col} not found in CSV"

    # test assertion is enough; avoid printing in tests


def test_basic():
    assert True
