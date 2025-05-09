import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os

# Ensure the data folder exists
os.makedirs("../forex_data_downloader/data", exist_ok=True)

# Initialize MT5
if not mt5.initialize():
    print("Initialization failed:", mt5.last_error())
    quit()

symbol = "GBPUSDm"
timeframes = {"D1": mt5.TIMEFRAME_D1, "H1": mt5.TIMEFRAME_H1}
years_of_data = 4

end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
start_date = end_date - timedelta(days=365 * years_of_data)

for tf_name, tf in timeframes.items():
    print(f"\nFetching {tf_name} data for {symbol} from {start_date.date()} to {end_date.date()}...")
    
    rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
    if rates is None:
        print(f"Failed to retrieve {tf_name} data. Error: {mt5.last_error()}")
        continue
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread']]

    # Save into data/ folder
    filename = f"data/{symbol}_{tf_name}_4years.csv"
    df.to_csv(filename, index=False)
    print(f"{tf_name} data saved to: {filename}")

mt5.shutdown()
