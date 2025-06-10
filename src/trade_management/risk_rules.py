import pandas as pd
import numpy as np

def calculate_sl_tp_levels(
    df_h1_with_signals: pd.DataFrame, #might be df_h1_aligned
    signal_col: str = 'rules_signal',
    atr_col: str = 'ATR_14',
    low_col: str = 'low',      # Low of the H1 signal candle
    high_col: str = 'high',    # High of the H1 signal candle
    close_col: str = 'close',  # Close of H1 signal candle (assumed entry for TP calc)
    sl_atr_offset_pips: float = 5.0, # The "5 fixed pips" part
    pip_value: float = 0.0001,       # For GBPUSD (5-decimal, pip is 4th). Adjust for JPY pairs (0.01)
    baseline_tp_rr_ratio: float = 2.0
) -> pd.DataFrame:
    """
    Calculates Stop Loss (SL) and Take Profit (TP) levels for rows with trade signals.
    SL Rule: Placed from the low/high of the signal candle. The distance is ATR + fixed_offset_pips.
    TP Rule: Based on baseline_tp_rr_ratio * actual_risk_from_entry.
    Assumes entry is at the close of the signal bar for TP calculation.
    """
    required_cols = [signal_col, atr_col, low_col, high_col, close_col]
    if df_h1_with_signals is None or not all(c in df_h1_with_signals.columns for c in required_cols):
        missing = [c for c in required_cols if c not in df_h1_with_signals.columns]
        print(f"Error: Input DataFrame for SL/TP missing required columns. Missing: {missing}")
        # Return a copy with initialized columns if input df is not None, else None
        if df_h1_with_signals is not None:
            df = df_h1_with_signals.copy()
            df['entry_price'] = np.nan
            df['stop_loss_price'] = np.nan
            df['take_profit_price'] = np.nan
            return df
        return None


    df = df_h1_with_signals.copy()
    df['entry_price'] = np.nan       # Assumed entry price (e.g., close of signal bar)
    df['stop_loss_price'] = np.nan
    df['take_profit_price'] = np.nan
    
    # Convert the fixed pip offset to actual price value
    sl_fixed_offset_in_price = sl_atr_offset_pips * pip_value

    for index, row in df.iterrows():
        signal = row[signal_col]
        atr_value = row[atr_col]
        entry_price_assumed = row[close_col] # Entry assumed at the close of the signal candle

        if pd.isna(atr_value): # Skip if ATR is NaN (e.g., early in data or no valid calculation)
            continue

        # This is the total distance to subtract from low (for buy) or add to high (for sell)
        sl_total_distance_from_wick = atr_value + sl_fixed_offset_in_price

        if signal == "BUY_RULE":
            df.loc[index, 'entry_price'] = entry_price_assumed
            # SL is placed 'sl_total_distance_from_wick' BELOW the LOW of the signal candle
            sl_price = row[low_col] - sl_total_distance_from_wick
            df.loc[index, 'stop_loss_price'] = sl_price
            
            # TP is based on risk taken from entry price relative to this SL
            actual_risk_per_trade = entry_price_assumed - sl_price
            if actual_risk_per_trade > 0: # Ensure risk is positive (entry > SL)
                tp_price = entry_price_assumed + (actual_risk_per_trade * baseline_tp_rr_ratio)
                df.loc[index, 'take_profit_price'] = tp_price
            else: 
                # This can happen if entry_price_assumed is already below or too close to the calculated sl_price
                print(f"Warning: For BUY_RULE at {index}, calculated SL {sl_price:.5f} is not sufficiently below assumed entry {entry_price_assumed:.5f} (Risk: {actual_risk_per_trade:.5f}). No TP set.")
                df.loc[index, 'take_profit_price'] = np.nan # Ensure TP is NaN if SL is problematic

        elif signal == "SELL_RULE":
            df.loc[index, 'entry_price'] = entry_price_assumed
            # SL is placed 'sl_total_distance_from_wick' ABOVE the HIGH of the signal candle
            sl_price = row[high_col] + sl_total_distance_from_wick
            df.loc[index, 'stop_loss_price'] = sl_price

            # TP is based on risk taken from entry price relative to this SL
            actual_risk_per_trade = sl_price - entry_price_assumed
            if actual_risk_per_trade > 0: # Ensure risk is positive (SL > entry)
                tp_price = entry_price_assumed - (actual_risk_per_trade * baseline_tp_rr_ratio)
                df.loc[index, 'take_profit_price'] = tp_price
            else:
                print(f"Warning: For SELL_RULE at {index}, calculated SL {sl_price:.5f} is not sufficiently above assumed entry {entry_price_assumed:.5f} (Risk: {actual_risk_per_trade:.5f}). No TP set.")
                df.loc[index, 'take_profit_price'] = np.nan # Ensure TP is NaN if SL is problematic
            
    print("SL/TP levels calculated for rule signals (if any).")
    return df