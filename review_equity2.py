from pathlib import Path
import pandas as pd
from src.data.moex_ohcl_loader import download_moex_ohlc
from src.data.equity_builder import build_equity

log_path = Path('trade_data/T_GLD_new_tr_1t.csv')
download_moex_ohlc(log_path, save=True)
candles_path = Path('OHCL') / f"{log_path.stem}_ohcl.csv"
equity = build_equity(candles_csv=candles_path, trades_csv=log_path)
print(equity.columns.tolist())
print('DATE_TIME monotonic', equity['DATE_TIME'].is_monotonic_increasing)
print('sample date_time lower', equity[['date_time','DATE_TIME']].head(5))
