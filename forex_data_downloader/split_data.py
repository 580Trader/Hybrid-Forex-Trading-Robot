import pandas as pd
import os

# Folder setup
data_dir = "data"
output_dir = "data/splits"
os.makedirs(output_dir, exist_ok=True)

# List of files to split
files = [
    "GBPUSDm_H1_clean.csv",
    "GBPUSDm_D1_clean.csv"
]

# Split ratios
train_ratio = 0.65
val_ratio = 0.20
test_ratio = 0.15

for file in files:
    filepath = os.path.join(data_dir, file)
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'])

    total = len(df)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    base = file.replace("_clean.csv", "")  # e.g., GBPUSDm_H1

    # Save the splits
    train_df.to_csv(os.path.join(output_dir, f"{base}_train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, f"{base}_val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, f"{base}_test.csv"), index=False)

    print(f"✅ Split {file} → Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
