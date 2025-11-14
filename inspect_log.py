from pathlib import Path

import pandas as pd

from src.data.csv_transformer_profit import load_transform_csv


log_path = Path("trade_data/T_GLD_new_tr_1t.csv")
result = load_transform_csv(log_path)
print(result.shape)
print(result[["date_time", "position_profit"]].head(20))
print(result[["date_time", "position_profit"]].tail(20))
print("monotonic", result["date_time"].is_monotonic_increasing)
print("dtype", result["date_time"].dtype)
