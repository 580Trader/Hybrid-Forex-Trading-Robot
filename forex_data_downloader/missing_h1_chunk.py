import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import os

# Initialize connection
if not mt5.initialize():
    raise RuntimeError("MT5 initialization failed. Error code:", mt5.last_error())

# Symbol
symbol = "GBPUSDm"
timeframe = mt5.TIMEFRAME_H1

# Define time range (UTC assumed)
start_date = datetime(2021, 5, 10)
end_date = datetime(2021, 7, 18, 23)  # Include the full day

# Get rates
rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
if rates is None or len(rates) == 0:
    raise RuntimeError("No data returned. Error code:", mt5.last_error())

# Convert to DataFrame
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

# Save to CSV
output_folder = "data/missing_h1_chunk"
os.makedirs(output_folder, exist_ok=True)
csv_path = os.path.join(output_folder, f"{symbol}_H1_20210509_20210718.csv")
df.to_csv(csv_path, index=False)

print(f"Downloaded {len(df)} H1 bars from {start_date.date()} to {end_date.date()} and saved to {csv_path}")

# Shutdown MT5
mt5.shutdown()
