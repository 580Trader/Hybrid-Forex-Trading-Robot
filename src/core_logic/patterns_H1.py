import pandas as pd
import numpy as np

# Let's reuse and adapt the swing point logic for H1
def find_h1_swing_points(df, N=3, high_col='high', low_col='low'): # N=3 might be a reasonable start for H1
    """
    Identifies potential swing high and swing low points on H1 data.
    A swing high is a candle with a high greater than the N candles before and after.
    A swing low is a candle with a low lower than the N candles before and after.
    """
    if not isinstance(df, pd.DataFrame) or not all(c in df.columns for c in [high_col, low_col]):
        print("Error: Input must be a DataFrame with specified high and low columns for H1 swings.")
        return None

    df_copy = df.copy()
    df_copy['is_h1_sh'] = False
    df_copy['is_h1_sl'] = False

    for i in range(N, len(df_copy) - N):
        is_sh = True
        for j in range(1, N + 1):
            if not (df_copy[high_col].iloc[i] >= df_copy[high_col].iloc[i-j] and \
                    df_copy[high_col].iloc[i] > df_copy[high_col].iloc[i+j]): # Allow current high to be equal to left, but strictly greater than right
                is_sh = False
                break
        if is_sh: # Check if it's also higher than the right N if the condition above was modified
             is_sh_right = True
             for j in range(1, N + 1):
                 if not (df_copy[high_col].iloc[i] > df_copy[high_col].iloc[i+j]):
                     is_sh_right = False
                     break
             if is_sh_right: # Stricter: current high must be greater than all N on right
                df_copy.loc[df_copy.index[i], 'is_h1_sh'] = True


        is_sl = True
        for j in range(1, N + 1):
            if not (df_copy[low_col].iloc[i] <= df_copy[low_col].iloc[i-j] and \
                    df_copy[low_col].iloc[i] < df_copy[low_col].iloc[i+j]): # Allow current low to be equal to left, but strictly lower than right
                is_sl = False
                break
        if is_sl: # Stricter check for right side
            is_sl_right = True
            for j in range(1, N + 1):
                if not (df_copy[low_col].iloc[i] < df_copy[low_col].iloc[i+j]):
                    is_sl_right = False
                    break
            if is_sl_right:
                df_copy.loc[df_copy.index[i], 'is_h1_sl'] = True
            
    return df_copy


# src/core_logic/patterns_H1.py
import pandas as pd
import numpy as np

# Make sure your find_h1_swing_points function is also in this file, 
# or imported if it's in a different utility file.
# For this example, I'll include a slightly refined version of find_h1_swing_points here.

def find_h1_swing_points(df, N=3, high_col='high', low_col='low'):
    """
    Identifies potential swing high and swing low points on H1 data.
    A swing high is a candle with a high strictly greater than the N candles before and after.
    A swing low is a candle with a low strictly lower than the N candles before and after.
    Adds 'is_h1_sh' and 'is_h1_sl' boolean columns.
    """
    if not isinstance(df, pd.DataFrame) or not all(c in df.columns for c in [high_col, low_col]):
        print("Error: Input must be a DataFrame with specified high and low columns for H1 swings.")
        return None

    df_copy = df.copy()
    # Initialize with False
    df_copy['is_h1_sh'] = False
    df_copy['is_h1_sl'] = False

    for i in range(N, len(df_copy) - N):
        # Potential Swing High: High[i] > High[i-N:i] and High[i] > High[i+1:i+N+1]
        is_sh = True
        for j in range(1, N + 1):
            if not (df_copy[high_col].iloc[i] > df_copy[high_col].iloc[i-j] and \
                    df_copy[high_col].iloc[i] > df_copy[high_col].iloc[i+j]):
                is_sh = False
                break
        if is_sh:
            df_copy.loc[df_copy.index[i], 'is_h1_sh'] = True

        # Potential Swing Low: Low[i] < Low[i-N:i] and Low[i] < Low[i+1:i+N+1]
        is_sl = True
        for j in range(1, N + 1):
            if not (df_copy[low_col].iloc[i] < df_copy[low_col].iloc[i-j] and \
                    df_copy[low_col].iloc[i] < df_copy[low_col].iloc[i+j]):
                is_sl = False
                break
        if is_sl:
            df_copy.loc[df_copy.index[i], 'is_h1_sl'] = True
            
    return df_copy


def identify_w_pattern_with_divergence(
    h1_df_with_swings_rsi_atr, 
    h1_sl_col='is_h1_sl',    # Name of boolean col for H1 Swing Lows
    h1_sh_col='is_h1_sh',    # <<<< **** MAKE SURE THIS PARAMETER IS DEFINED HERE ****
    low_col='low', 
    high_col='high', 
    close_col='close', 
    open_col='open',
    rsi_col='RSI_10', 
    atr_col='ATR_14',
    price_tolerance_atr_multiplier=0.5,
    min_peak_prominence_atr_multiplier=0.3,
    max_bars_between_bottoms=40,
    confirmation_lookback=3
):
    """
    Identifies W-Patterns (Double Bottoms) with Bullish RSI Divergence on H1 data.
    Adds 'w_pattern_buy_signal' boolean column to the input DataFrame.
    """
    required_cols = [low_col, high_col, close_col, open_col, rsi_col, atr_col, h1_sl_col, h1_sh_col] # Include h1_sh_col
    if not isinstance(h1_df_with_swings_rsi_atr, pd.DataFrame) or \
       not all(c in h1_df_with_swings_rsi_atr.columns for c in required_cols):
        missing = [c for c in required_cols if c not in h1_df_with_swings_rsi_atr.columns]
        print(f"Error: Input DataFrame for W-Pattern missing required columns. Missing: {missing}")
        print(f"DataFrame actually has columns: {h1_df_with_swings_rsi_atr.columns.tolist()}")
        return None

    df = h1_df_with_swings_rsi_atr.copy()
    df['w_pattern_buy_signal'] = False 
    
    # Get DataFrames of H1 swing lows and highs for easier iteration
    # Using the passed column names for swing high/low booleans
    potential_b1s = df[df[h1_sl_col]]
    # Ensure h1_sh_col is used when needing swing high points
    # For example, if finding P1s:
    # potential_p1s_in_window = window_df[window_df[h1_sh_col]] # This was the line that caused NameError before

    if potential_b1s.empty:
        print("No H1 swing lows (potential B1s) found to start W-pattern search.")
        return df

    # Iterate through each potential B1 (first bottom)
    for b1_idx, b1_row in potential_b1s.iterrows():
        b1_price = b1_row[low_col]
        b1_rsi = b1_row[rsi_col]
        atr_at_b1 = b1_row[atr_col]
        
        b1_iloc = df.index.get_loc(b1_idx) 
        search_window_start_iloc = b1_iloc + 1
        search_window_end_iloc = min(search_window_start_iloc + max_bars_between_bottoms, len(df))
        
        window_df_after_b1 = df.iloc[search_window_start_iloc:search_window_end_iloc]
        
        if window_df_after_b1.empty:
            continue

        potential_p1s_in_window = window_df_after_b1[window_df_after_b1[h1_sh_col]] # Using h1_sh_col parameter
        if potential_p1s_in_window.empty:
            continue

        for p1_idx, p1_row in potential_p1s_in_window.iterrows():
            p1_price = p1_row[high_col]

            if not (p1_price > b1_price + min_peak_prominence_atr_multiplier * atr_at_b1):
                continue

            p1_iloc = df.index.get_loc(p1_idx) 
            search_window_start_for_b2_iloc = p1_iloc + 1
            search_window_end_for_b2_iloc = min(b1_iloc + 1 + max_bars_between_bottoms, len(df))
            
            if search_window_start_for_b2_iloc >= search_window_end_for_b2_iloc:
                continue

            window_for_b2 = df.iloc[search_window_start_for_b2_iloc:search_window_end_for_b2_iloc]
            if window_for_b2.empty:
                continue
            
            potential_b2s_in_window_after_p1 = window_for_b2[window_for_b2[h1_sl_col]] # Using h1_sl_col parameter
            if potential_b2s_in_window_after_p1.empty:
                continue

            for b2_idx, b2_row in potential_b2s_in_window_after_p1.iterrows():
                b2_price = b2_row[low_col]
                b2_rsi = b2_row[rsi_col]
                atr_at_b2 = b2_row[atr_col]

                price_diff_ok = abs(b1_price - b2_price) < (price_tolerance_atr_multiplier * ((atr_at_b1 + atr_at_b2) / 2))
                b2_holds_b1_level = b2_price >= b1_price - (price_tolerance_atr_multiplier * 0.25 * atr_at_b1) 

                if not (price_diff_ok and b2_holds_b1_level):
                    continue
                
                if not (p1_price > b1_price and p1_price > b2_price):
                    continue
                
                price_condition_for_divergence = b2_price <= b1_price + (price_tolerance_atr_multiplier * 0.25 * atr_at_b1) 
                rsi_divergence_present = b2_rsi > (b1_rsi + 1) 

                if not (price_condition_for_divergence and rsi_divergence_present):
                    continue

                b2_iloc_in_df = df.index.get_loc(b2_idx)
                confirmed = False
                confirmation_candle_idx = None
                for k_confirm in range(1, confirmation_lookback + 1):
                    if b2_iloc_in_df + k_confirm < len(df):
                        idx_confirm_candle = df.index[b2_iloc_in_df + k_confirm]
                        if df.loc[idx_confirm_candle, close_col] > df.loc[idx_confirm_candle, open_col]: 
                            confirmed = True
                            confirmation_candle_idx = idx_confirm_candle
                            df.loc[confirmation_candle_idx, 'w_pattern_buy_signal'] = True
                            break 
                
                if confirmed:
                    pass 
    
    if 'w_pattern_buy_signal' in df.columns and df['w_pattern_buy_signal'].any():
        print(f"W-Pattern Buy Signals identified: {df['w_pattern_buy_signal'].sum()}")
    else:
        print("No W-Pattern Buy Signals identified with current logic.")
        
    return df

def identify_m_pattern_with_divergence(h1_df_with_swings_rsi_atr, 
                                       h1_sh_col='is_h1_sh',    # Name of boolean col for H1 Swing Highs
                                       h1_sl_col='is_h1_sl',    # Name of boolean col for H1 Swing Lows
                                       high_col='high', 
                                       low_col='low', 
                                       close_col='close', 
                                       open_col='open',      # Added open_col for bearish confirmation candle
                                       rsi_col='RSI_10', 
                                       atr_col='ATR_14',
                                       price_tolerance_atr_multiplier=0.5, # T1 & T2 highs should be within 0.5*ATR
                                       min_valley_prominence_atr_multiplier=0.3, # Valley V1 should be at least 0.3*ATR below lower of T1/T2
                                       max_bars_between_tops=40,      # Max H1 bars from T1 to T2
                                       confirmation_lookback=3):       # Look for bearish confirmation candle within X bars after T2
    """
    Identifies M-Patterns (Double Tops) with Bearish RSI Divergence on H1 data.
    Adds 'm_pattern_sell_signal' boolean column to the input DataFrame.
    """
    required_cols = [high_col, low_col, close_col, open_col, rsi_col, atr_col, h1_sh_col, h1_sl_col]
    if not isinstance(h1_df_with_swings_rsi_atr, pd.DataFrame) or \
       not all(c in h1_df_with_swings_rsi_atr.columns for c in required_cols):
        missing = [c for c in required_cols if c not in h1_df_with_swings_rsi_atr.columns]
        print(f"Error: Input DataFrame missing required columns for M-Pattern. Missing: {missing}")
        return None

    df = h1_df_with_swings_rsi_atr.copy()
    df['m_pattern_sell_signal'] = False # Initialize the signal column
    
    potential_t1s = df[df[h1_sh_col]] # DataFrame of all potential first tops

    if potential_t1s.empty:
        print("No H1 swing highs (potential T1s) found to start M-pattern search.")
        return df

    for t1_idx, t1_row in potential_t1s.iterrows():
        t1_price = t1_row[high_col]
        t1_rsi = t1_row[rsi_col]
        atr_at_t1 = t1_row[atr_col]
        
        t1_iloc = df.index.get_loc(t1_idx)
        search_window_start_iloc = t1_iloc + 1
        search_window_end_iloc = min(search_window_start_iloc + max_bars_between_tops, len(df))
        
        window_df_after_t1 = df.iloc[search_window_start_iloc:search_window_end_iloc]
        
        if window_df_after_t1.empty:
            continue

        potential_v1s_in_window = window_df_after_t1[window_df_after_t1[h1_sl_col]] # Valley is a swing low
        if potential_v1s_in_window.empty:
            continue

        # Iterate through potential V1s (simplification: take the lowest V1 in window for now)
        v1_idx = potential_v1s_in_window[low_col].idxmin()
        v1_price = potential_v1s_in_window.loc[v1_idx, low_col]
        
        # Valley V1 must be sufficiently prominent below T1
        if not (v1_price < t1_price - min_valley_prominence_atr_multiplier * atr_at_t1):
            continue 

        v1_iloc_in_window = window_df_after_t1.index.get_loc(v1_idx)
        window_for_t2 = window_df_after_t1.iloc[v1_iloc_in_window + 1:]
        
        if window_for_t2.empty:
            continue
            
        potential_t2s_in_window_after_v1 = window_for_t2[window_for_t2[h1_sh_col]] # T2 is a swing high
        if potential_t2s_in_window_after_v1.empty:
            continue

        for t2_idx, t2_row in potential_t2s_in_window_after_v1.iterrows():
            t2_price = t2_row[high_col]
            t2_rsi = t2_row[rsi_col]
            atr_at_t2 = t2_row[atr_col]

            price_diff_ok = abs(t1_price - t2_price) < (price_tolerance_atr_multiplier * ((atr_at_t1 + atr_at_t2) / 2))
            t2_not_far_above_t1 = t2_price <= t1_price + (price_tolerance_atr_multiplier * 0.25 * atr_at_t1)

            if not (price_diff_ok and t2_not_far_above_t1):
                continue
            
            if not (v1_price < t1_price and v1_price < t2_price): # Valley must be lower than both tops
                continue
            
            # Condition: Bearish RSI Divergence
            # Price makes equal or slightly higher high (T2 vs T1), RSI makes lower high
            price_condition_for_divergence = t2_price >= t1_price - (price_tolerance_atr_multiplier * 0.1 * atr_at_t1)
            rsi_divergence_present = t2_rsi < (t1_rsi - 1) # RSI at T2 must be clearly lower

            if not (price_condition_for_divergence and rsi_divergence_present):
                continue

            t2_iloc_in_df = df.index.get_loc(t2_idx)
            confirmed = False
            confirmation_candle_idx = None
            for k_confirm in range(1, confirmation_lookback + 1):
                if t2_iloc_in_df + k_confirm < len(df):
                    idx_confirm_candle = df.index[t2_iloc_in_df + k_confirm]
                    if df.loc[idx_confirm_candle, close_col] < df.loc[idx_confirm_candle, open_col]: # Bearish candle
                        confirmed = True
                        confirmation_candle_idx = idx_confirm_candle
                        df.loc[confirmation_candle_idx, 'm_pattern_sell_signal'] = True
                        break 
            
            if confirmed:
                pass 
    
    if 'm_pattern_sell_signal' in df.columns and df['m_pattern_sell_signal'].any():
        print(f"M-Pattern Sell Signals identified: {df['m_pattern_sell_signal'].sum()}")
    else:
        print("No M-Pattern Sell Signals identified with current logic.")
        
    return df