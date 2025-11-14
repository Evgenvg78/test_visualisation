from pathlib import Path
import pandas as pd
from src.data.moex_ohcl_loader import download_moex_ohlc
from src.data.equity_builder import build_equity

pd.set_option("display.max_rows", 60)
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 120)

log_path = Path("trade_data/T_GLD_new_tr_1t.csv")
candles = download_moex_ohlc(log_path, save=True)
candles_path = Path("OHCL") / f"{log_path.stem}_ohcl.csv"
equity = build_equity(candles_csv=candles_path, trades_csv=log_path)
print(equity["date_time"].head(20))
print(equity["date_time"].tail(20))
print('monotonic', equity['date_time'].is_monotonic_increasing)
