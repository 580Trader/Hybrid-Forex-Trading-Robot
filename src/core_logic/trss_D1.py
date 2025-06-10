import pandas as pd
import numpy as np

def find_basic_swing_points(df, N=1, high_col='high', low_col='low'):
    """
    Identifies potential swing high and swing low points based on N neighboring candles.
    A swing high is a candle with a high greater than the N candles before and after.
    A swing low is a candle with a low lower than the N candles before and after.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with 'high' and 'low' price columns.
    - N (int): Number of neighboring candles to compare against on each side.
    - high_col (str): Name of the high price column.
    - low_col (str): Name of the low price column.

    Returns:
    - pd.DataFrame: Original DataFrame with two new boolean columns:
                    'is_potential_sh' (True for potential swing highs)
                    'is_potential_sl' (True for potential swing lows)
    """
    if not isinstance(df, pd.DataFrame) or not all(c in df.columns for c in [high_col, low_col]):
        print("Error: Input must be a DataFrame with specified high and low columns.")
        return None

    df_copy = df.copy()
    df_copy['is_potential_sh'] = False
    df_copy['is_potential_sl'] = False

    # Iterate through the DataFrame, avoiding edges where N neighbors don't exist
    for i in range(N, len(df_copy) - N):
        # Check for potential Swing High
        is_sh = True
        for j in range(1, N + 1):
            if not (df_copy[high_col].iloc[i] > df_copy[high_col].iloc[i-j] and \
                    df_copy[high_col].iloc[i] > df_copy[high_col].iloc[i+j]):
                is_sh = False
                break
        if is_sh:
            df_copy.loc[df_copy.index[i], 'is_potential_sh'] = True

        # Check for potential Swing Low
        is_sl = True
        for j in range(1, N + 1):
            if not (df_copy[low_col].iloc[i] < df_copy[low_col].iloc[i-j] and \
                    df_copy[low_col].iloc[i] < df_copy[low_col].iloc[i+j]):
                is_sl = False
                break
        if is_sl:
            df_copy.loc[df_copy.index[i], 'is_potential_sl'] = True
            
    return df_copy


def validate_swings_with_3_candle_pullback(df_with_potential_swings, 
                                           potential_sh_col='is_potential_sh',
                                           potential_sl_col='is_potential_sl',
                                           high_col='high', low_col='low', close_col='close'):
    """
    Validates potential swing points using the 3-candle pullback rule.
    - Major Swing Low: After a potential SL, 3 subsequent candles do not close below its low.
    - Major Swing High: After a potential SH, 3 subsequent candles do not close above its high.
    """
    if not isinstance(df_with_potential_swings, pd.DataFrame) or \
       not all(c in df_with_potential_swings.columns for c in [potential_sh_col, potential_sl_col, high_col, low_col, close_col]):
        print("Error: Invalid DataFrame or missing required columns.")
        return None

    df_copy = df_with_potential_swings.copy()
    df_copy['is_major_sh'] = False
    df_copy['is_major_sl'] = False

    # Iterate through rows where a potential swing was identified
    # We need at least 3 future candles to check the pullback rule.
    for i in range(len(df_copy) - 3): 
        current_index = df_copy.index[i]

        if df_copy.loc[current_index, potential_sl_col]: # If it's a potential Swing Low
            swing_low_price = df_copy.loc[current_index, low_col]
            valid_pullback = True
            for k in range(1, 4): # Check next 3 candles
                if df_copy[close_col].iloc[i+k] < swing_low_price: 
                    valid_pullback = False
                    break
            if valid_pullback:
                df_copy.loc[current_index, 'is_major_sl'] = True

        if df_copy.loc[current_index, potential_sh_col]: # If it's a potential Swing High
            swing_high_price = df_copy.loc[current_index, high_col]
            valid_pullback = True
            for k in range(1, 4): # Check next 3 candles
                if df_copy[close_col].iloc[i+k] > swing_high_price:
                    valid_pullback = False
                    break
            if valid_pullback:
                df_copy.loc[current_index, 'is_major_sh'] = True
                
    return df_copy

def determine_d1_market_structure(df_with_major_swings, 
                                   major_sh_col='is_major_sh', major_sl_col='is_major_sl',
                                   high_col='high', low_col='low', close_col='close'):
    """
    Determines D1 market structure based on major swing points and BoS logic.
    States: "Undetermined", "Uptrend", "Downtrend", 
            "Bearish BoS Target", "Bullish BoS Target"
    A confirmed BoS immediately transitions the trend.
    """
    if not isinstance(df_with_major_swings, pd.DataFrame) or \
       not all(c in df_with_major_swings.columns for c in 
               [major_sh_col, major_sl_col, high_col, low_col, close_col]):
        print("Error: Invalid DataFrame or missing required columns for structure determination.")
        return None

    df = df_with_major_swings.copy()
    df['d1_market_structure'] = "Undetermined"

    # Store actual prices and indices of swings
    last_sl_price, second_last_sl_price = np.nan, np.nan
    last_sh_price, second_last_sh_price = np.nan, np.nan
    
    # These are the key structural points that define the current trend
    # For an uptrend, structure_defining_low is the last confirmed Higher Low (HL)
    # For a downtrend, structure_defining_high is the last confirmed Lower High (LH)
    structure_defining_low = np.nan 
    structure_defining_high = np.nan

    current_trend = "Undetermined"

    for i in range(len(df)):
        # Carry forward previous state initially
        if i > 0:
            current_trend = df['d1_market_structure'].iloc[i-1]
            # If previous state was a BoS target, maintain it until resolved
            if "BoS Target" in current_trend:
                pass # Keep current_trend as "BoS Target" state
            # If prev state was confirmed trend, keep it
            elif current_trend in ["Uptrend", "Downtrend"]:
                 pass
            else: # If prev state was e.g. BoS Confirmed, reset to Undetermined to find new trend
                 current_trend = "Undetermined"


        is_new_sl = df[major_sl_col].iloc[i]
        new_sl_price = df[low_col].iloc[i] if is_new_sl else np.nan

        is_new_sh = df[major_sh_col].iloc[i]
        new_sh_price = df[high_col].iloc[i] if is_new_sh else np.nan

        current_close = df[close_col].iloc[i]

        # --- Process Swing Lows ---
        if is_new_sl:
            if pd.notna(last_sl_price): # Ensure there is a last_sl_price to shift
                second_last_sl_price = last_sl_price
            last_sl_price = new_sl_price
            
            if current_trend == "Bullish BoS Target":
                # We need a Higher Low after breaking the MSH (structure_defining_high for the old downtrend)
                if last_sl_price > structure_defining_low: # structure_defining_low was the LL before BoS
                    current_trend = "Uptrend" # Bullish BoS Confirmed
                    structure_defining_low = last_sl_price # This new HL defines the new uptrend start
                    # structure_defining_high needs to be formed next by a HH
                    structure_defining_high = np.nan # Reset, waiting for a HH
            elif pd.notna(second_last_sl_price) and pd.notna(last_sh_price) and pd.notna(second_last_sh_price):
                # Check for Uptrend: Higher Low (last_sl > second_last_sl) 
                # AND Higher High (last_sh > second_last_sh, where last_sh formed *before* current last_sl)
                if last_sl_price > second_last_sl_price and last_sh_price > second_last_sh_price:
                    current_trend = "Uptrend"
                    structure_defining_low = last_sl_price
                    # structure_defining_high was the last_sh_price that made the HH.

        # --- Process Swing Highs ---
        if is_new_sh:
            if pd.notna(last_sh_price):
                second_last_sh_price = last_sh_price
            last_sh_price = new_sh_price

            if current_trend == "Bearish BoS Target":
                # We need a Lower High after breaking the MSL (structure_defining_low for the old uptrend)
                if last_sh_price < structure_defining_high: # structure_defining_high was the HH before BoS
                    current_trend = "Downtrend" # Bearish BoS Confirmed
                    structure_defining_high = last_sh_price # This new LH defines the new downtrend start
                    # structure_defining_low needs to be formed next by a LL
                    structure_defining_low = np.nan # Reset, waiting for a LL
            elif pd.notna(second_last_sh_price) and pd.notna(last_sl_price) and pd.notna(second_last_sl_price):
                # Check for Downtrend: Lower High (last_sh < second_last_sh)
                # AND Lower Low (last_sl < second_last_sl, where last_sl formed *before* current last_sh)
                if last_sh_price < second_last_sh_price and last_sl_price < second_last_sl_price:
                    current_trend = "Downtrend"
                    structure_defining_high = last_sh_price
                    # structure_defining_low was the last_sl_price that made the LL.

        # --- Update structure_defining_high/low based on confirmed trends from swings ---
        # This logic is complex because the last_sh and last_sl used for trend confirmation
        # might not be the *immediately* preceding ones. This needs a more robust way to track
        # the sequence (MSL1 -> MSH1 -> MSL2(HL) -> MSH2(HH)).
        # For now, the BoS confirmation above is the primary way a trend flips.
        # If current_trend is Uptrend, ensure structure_defining_low is the latest valid HL.
        # If current_trend is Downtrend, ensure structure_defining_high is the latest valid LH.

        if current_trend == "Uptrend":
            if is_new_sl and (pd.isna(structure_defining_low) or new_sl_price > structure_defining_low):
                 # This condition is too simple, needs context of previous HH.
                 # Let's rely on BoS confirmation for now or a full HH-HL sequence.
                 # If new_sl_price forms a HL and previous SH was a HH, then this new_sl_price is a good candidate for structure_defining_low.
                 # This part is tricky. The BoS confirmation logic handles trend changes more explicitly.
                 # For trend continuation, if a new MSL forms and it's a HL, update structure_defining_low.
                 if is_new_sl and (pd.isna(structure_defining_low) or new_sl_price > structure_defining_low):
                     if pd.notna(last_sh_price) and (pd.isna(structure_defining_high) or last_sh_price > structure_defining_high): # Ensure we also made a HH
                        structure_defining_low = new_sl_price
                        # structure_defining_high would be the last_sh_price that confirmed the HH
                        if pd.notna(last_sh_price) : structure_defining_high = last_sh_price

        elif current_trend == "Downtrend":
            if is_new_sh and (pd.isna(structure_defining_high) or new_sh_price < structure_defining_high):
                 if pd.notna(last_sl_price) and (pd.isna(structure_defining_low) or last_sl_price < structure_defining_low): # Ensure we also made a LL
                    structure_defining_high = new_sh_price
                    if pd.notna(last_sl_price) : structure_defining_low = last_sl_price


        # --- Initial Break of Structure (BoS Target) ---
        # Only check for BoS if we are in a confirmed trend and not already in a BoS target state
        if current_trend == "Uptrend" and pd.notna(structure_defining_low):
            if current_close < structure_defining_low:
                current_trend = "Bearish BoS Target"
                # structure_defining_high (the HH before this BoS) remains important for LH confirmation
        elif current_trend == "Downtrend" and pd.notna(structure_defining_high):
            if current_close > structure_defining_high:
                current_trend = "Bullish BoS Target"
                # structure_defining_low (the LL before this BoS) remains important for HL confirmation
        
        df.loc[df.index[i], 'd1_market_structure'] = current_trend
        
    return df