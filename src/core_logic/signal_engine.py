# src/core_logic/signal_engine.py
import pandas as pd
import numpy as np

def generate_rules_based_signals(
    df_d1_structure: pd.DataFrame, 
    df_h1_with_patterns: pd.DataFrame,
    d1_close_col: str = 'close',                 # Original name in df_d1_structure
    d1_sma_col: str = 'SMA_50',                  # Original name in df_d1_structure
    d1_structure_col: str = 'd1_market_structure', # Original name in df_d1_structure
    h1_w_pattern_col: str = 'w_pattern_buy_signal',
    h1_m_pattern_col: str = 'm_pattern_sell_signal',
    h1_hs_buy_col: str = 'ihs_buy_limit_setup', 
    h1_hs_sell_col: str = 'hs_sell_limit_setup'
) -> pd.DataFrame:

    if df_d1_structure is None or df_h1_with_patterns is None:
        print("ERROR (signal_engine): Input DataFrames cannot be None.")
        return df_h1_with_patterns # Or return None if df_h1_with_patterns is also None

    df_h1_aligned = df_h1_with_patterns.copy()

    # --- 1. Prepare D1 Context for Merge ---
    # Select only the D1 columns needed for context, using the provided original names
    d1_context_cols_original = [d1_close_col, d1_sma_col, d1_structure_col]
    missing_d1_cols = [col for col in d1_context_cols_original if col not in df_d1_structure.columns]
    if missing_d1_cols:
        print(f"ERROR (signal_engine): df_d1_structure is missing required D1 context columns: {missing_d1_cols}")
        return df_h1_aligned # Return H1 df without signals if D1 context is incomplete

    d1_context_to_merge = df_d1_structure[d1_context_cols_original].copy()
    
    if not isinstance(d1_context_to_merge.index, pd.DatetimeIndex):
        print("ERROR (signal_engine): df_d1_structure index is not DatetimeIndex.")
        return df_h1_aligned 
    d1_context_to_merge.index = d1_context_to_merge.index.normalize() # Ensure D1 index is just date

    # --- Debug print for d1_context_to_merge (checking our specific date) ---
    date_to_check_d1_debug = pd.Timestamp('2023-12-25', tz='UTC').normalize()
    print(f"\nDEBUG (signal_engine): d1_context_to_merge (checking for {date_to_check_d1_debug.date()}):")
    if date_to_check_d1_debug in d1_context_to_merge.index:
        row_to_debug = d1_context_to_merge.loc[[date_to_check_d1_debug]]
        print(row_to_debug)
        # Check specific values and types if needed, as done previously
    else:
        print(f"Date {date_to_check_d1_debug} NOT in d1_context_to_merge.index.")

    # --- 2. Prepare H1 Merge Key ---
    if not isinstance(df_h1_aligned.index, pd.DatetimeIndex):
        print("ERROR (signal_engine): df_h1_aligned index is not DatetimeIndex.")
        return df_h1_aligned
    df_h1_aligned['date_for_d1_merge'] = df_h1_aligned.index.normalize()

    # --- Debug print for H1 merge keys ---
    debug_h1_merge_key_target_date = pd.Timestamp('2023-12-25', tz='UTC').normalize()
    h1_rows_for_debug_date = df_h1_aligned[df_h1_aligned['date_for_d1_merge'] == debug_h1_merge_key_target_date]
    print(f"\nDEBUG (signal_engine): H1 'date_for_d1_merge' values for date {debug_h1_merge_key_target_date.date()}: Count = {len(h1_rows_for_debug_date)}")
    if not h1_rows_for_debug_date.empty:
        print(h1_rows_for_debug_date[['date_for_d1_merge']].head())

    # --- 3. Perform Merge ---
    df_h1_aligned = pd.merge(df_h1_aligned, d1_context_to_merge, 
                             left_on='date_for_d1_merge', right_index=True, 
                             how='left', suffixes=('', '_d1_ctx')) 
    
    print(f"DEBUG (signal_engine): Columns in df_h1_aligned IMMEDIATELY AFTER MERGE: {df_h1_aligned.columns.tolist()}")
    
    # Define the names of the newly merged D1 columns (with suffix)
    d1_close_col_merged = f'{d1_close_col}_d1_ctx'
    d1_sma_col_merged = f'SMA_50'
    d1_structure_col_merged = f'd1_market_structure'
    merged_d1_cols_to_inspect = [d1_close_col_merged, d1_sma_col_merged, d1_structure_col_merged]

    print(f"\nDEBUG (signal_engine): df_h1_aligned AFTER MERGE, BEFORE FFILL (for H1 bars on {debug_h1_merge_key_target_date.date()}):")
    # Slice using the original H1 index, normalized, to get the relevant H1 bars
    h1_slice_after_merge = df_h1_aligned[df_h1_aligned.index.normalize() == debug_h1_merge_key_target_date]
    
    if not h1_slice_after_merge.empty:
         print(f"  Slice shape after merge for {debug_h1_merge_key_target_date.date()}: {h1_slice_after_merge.shape}")
         print(f"  Columns available in this slice: {h1_slice_after_merge.columns.tolist()}") # List all columns in the slice

         # Explicitly try to access and print each D1 context column
         for col_name_to_check in merged_d1_cols_to_inspect: # merged_d1_cols_to_inspect = [d1_close_col_merged, d1_sma_col_merged, d1_structure_col_merged]
             if col_name_to_check in h1_slice_after_merge.columns:
                 print(f"    Values for '{col_name_to_check}' in slice:\n{h1_slice_after_merge[col_name_to_check].to_string()}")
                 if h1_slice_after_merge[col_name_to_check].isnull().all():
                     print(f"    WARNING: Column '{col_name_to_check}' is ALL NaN for this slice after merge.")
                 elif h1_slice_after_merge[col_name_to_check].notnull().any():
                     print(f"    SUCCESS: Column '{col_name_to_check}' has non-NaN values after merge.")
                 else: # Should be caught by isnull().all() but good for completeness
                     print(f"    Column '{col_name_to_check}' exists but its state is unclear (possibly empty string or other non-NaN but 'empty' value).")
             else:
                 print(f"    CRITICAL: Column '{col_name_to_check}' IS MISSING from slice after merge.")
         
         # Also print a key H1 column for context
         if h1_w_pattern_col in h1_slice_after_merge.columns:
             print(f"    Context H1 values for '{h1_w_pattern_col}':\n{h1_slice_after_merge[h1_w_pattern_col].to_string()}")
    else:
         print(f"  No H1 bars found for date {debug_h1_merge_key_target_date.date()} in df_h1_aligned to check post-merge state.")
    # --- END MODIFIED DEBUG PRINT ---

    # --- 4. Forward Fill ---
    cols_to_ffill = [d1_close_col_merged, d1_sma_col_merged, d1_structure_col_merged]
    existing_cols_to_ffill = [col for col in cols_to_ffill if col in df_h1_aligned.columns] # Check if suffixed cols exist
    if existing_cols_to_ffill: 
        df_h1_aligned[existing_cols_to_ffill] = df_h1_aligned[existing_cols_to_ffill].ffill()
    else:
        print(f"WARNING (signal_engine): Merged D1 context columns ({cols_to_ffill}) not found after merge. Cannot forward-fill.")

    # --- DEBUG: Inspect merged columns for 2023-12-25 H1 bars AFTER ffill ---
    print(f"\nDEBUG (signal_engine): df_h1_aligned AFTER FFILL (for H1 bars on {debug_h1_merge_key_target_date.date()}):")
    # Slice using the original H1 index, normalized
    h1_slice_after_ffill = df_h1_aligned[df_h1_aligned.index.normalize() == debug_h1_merge_key_target_date]
    if not h1_slice_after_ffill.empty:
         cols_exist_post_ffill = [col for col in merged_d1_cols_to_inspect if col in h1_slice_after_ffill.columns]
         if cols_exist_post_ffill:
             cols_to_print_post_ffill = cols_exist_post_ffill + ([h1_w_pattern_col] if h1_w_pattern_col in h1_slice_after_ffill else [])
             print(h1_slice_after_ffill[cols_to_print_post_ffill].head())
         else:
             print(f"  None of D1 context columns ({merged_d1_cols_to_inspect}) found post-ffill for this slice.")
    else:
         print(f"  No H1 bars found for date {debug_h1_merge_key_target_date.date()} in df_h1_aligned after ffill (checking by index).")
    
    if 'date_for_d1_merge' in df_h1_aligned.columns:
        df_h1_aligned.drop(columns=['date_for_d1_merge'], inplace=True, errors='ignore')
    
    df_h1_aligned['rules_signal'] = "NO_RULE_SIGNAL"
    
    # --- 5. Main Loop & Targeted Debug for Specific H1 Timestamp ---
    test_timestamp = pd.Timestamp('2023-12-25 23:00:00+00:00', tz='UTC') 
    
    for index, row in df_h1_aligned.iterrows():
        current_d1_structure = row.get(d1_structure_col_merged, "Undetermined") # Uses suffixed column name
        d1_bar_close_price = row.get(d1_close_col_merged, np.nan)        # Uses suffixed column name
        d1_bar_sma = row.get(d1_sma_col_merged, np.nan)                # Uses suffixed column name
        
        is_w_buy = row.get(h1_w_pattern_col, False)
        is_m_sell = row.get(h1_m_pattern_col, False)
        is_ihs_buy = row.get(h1_hs_buy_col, False) 
        is_hs_sell = row.get(h1_hs_sell_col, False)

        if index == test_timestamp: 
            print(f"\nDEBUGGING H1 BAR AT: {index} (inside generate_rules_based_signals loop)")
            print(f"  H1 W-Pattern Flag: {is_w_buy}")
            print(f"  H1 M-Pattern Flag: {is_m_sell}")
            print(f"  D1 Structure from Merged Data ('{d1_structure_col_merged}'): {current_d1_structure}")
            print(f"  D1 Close from Merged Data ('{d1_close_col_merged}'): {d1_bar_close_price}")
            print(f"  D1 SMA_50 from Merged Data ('{d1_sma_col_merged}'): {d1_bar_sma}")

            d1_price_above_sma_debug, d1_price_below_sma_debug = False, False
            if pd.notna(d1_bar_close_price) and pd.notna(d1_bar_sma):
                d1_price_above_sma_debug = d1_bar_close_price > d1_bar_sma
                d1_price_below_sma_debug = d1_bar_close_price < d1_bar_sma
                print(f"  Is D1 Close > D1 SMA_50? {d1_price_above_sma_debug}")
                print(f"  Is D1 Close < D1 SMA_50? {d1_price_below_sma_debug}")
            else:
                print("  D1 Close or D1 SMA is NaN for this H1 bar's day for comparison.")

            d1_struct_bullish_ok_debug = current_d1_structure in ["Uptrend", "Bullish BoS Confirmed", "Bullish BoS Target"]
            d1_sma_bullish_ok_debug = d1_price_above_sma_debug
            print(f"  D1 Structure OK for Buy? {d1_struct_bullish_ok_debug}")
            print(f"  D1 SMA OK for Buy? {d1_sma_bullish_ok_debug}")
            if d1_struct_bullish_ok_debug and d1_sma_bullish_ok_debug and is_w_buy:
                print("  ----> DEBUG: CONDITIONS FOR BUY_RULE MET FOR THIS BAR <----")
            else:
                print("  ----> DEBUG: Conditions for BUY_RULE NOT MET <----")
                if not d1_struct_bullish_ok_debug: print("        DEBUG Reason: D1 Structure not bullish enough.")
                if not d1_sma_bullish_ok_debug: print("        DEBUG Reason: D1 Price not above D1 SMA (or SMA was NaN).")
                if not is_w_buy: print("        DEBUG Reason: H1 W-Pattern not active.")
            # (Similar detailed debug for sell can be added if needed)

        # Main Logic
        if pd.isna(d1_bar_close_price) or pd.isna(d1_bar_sma):
            continue 

        d1_bullish_context = (current_d1_structure in ["Uptrend", "Bullish BoS Confirmed", "Bullish BoS Target"]) and \
                             (d1_bar_close_price > d1_bar_sma)
        h1_buy_pattern_active = is_w_buy or is_ihs_buy
        
        d1_bearish_context = (current_d1_structure in ["Downtrend", "Bearish BoS Confirmed", "Bearish BoS Target"]) and \
                             (d1_bar_close_price < d1_bar_sma)
        h1_sell_pattern_active = is_m_sell or is_hs_sell
        
        if d1_bullish_context and h1_buy_pattern_active:
            df_h1_aligned.loc[index, 'rules_signal'] = "BUY_RULE"
        elif d1_bearish_context and h1_sell_pattern_active:
            df_h1_aligned.loc[index, 'rules_signal'] = "SELL_RULE"
            
    print(f"\nRule-based signals generated (end of function): BUY_RULE count = {(df_h1_aligned['rules_signal'] == 'BUY_RULE').sum()}, SELL_RULE count = {(df_h1_aligned['rules_signal'] == 'SELL_RULE').sum()}")
    return df_h1_aligned