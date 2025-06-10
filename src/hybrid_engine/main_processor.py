# src/hybrid_engine/main_processor.py
import pandas as pd
import numpy as np

# We will need to import the SL/TP calculation function
# Ensure this path is correct based on your structure if you run this file directly
# For notebook imports, the sys.path setup in the notebook will handle it.
try:
    from src.trade_management.risk_rules import calculate_sl_tp_levels
except ModuleNotFoundError: # Fallback for direct run if path not set
    print("Warning (main_processor.py): Could not import calculate_sl_tp_levels directly. Ensure PYTHONPATH or sys.path is set if running standalone.")
    # Define a dummy if needed for standalone testing of this file, though not recommended
    def calculate_sl_tp_levels(df, **kwargs): print("Dummy SL/TP called"); return df


# --- 1. AI Component Placeholder Functions ---
# You (Sakhe) will replace these with calls to your actual K-Means and LSTM model prediction functions
# once you have developed those AI components.

def get_d1_kmeans_regime(d1_bar_features: pd.Series) -> str:
    """
    Placeholder: Simulates K-Means D1 market regime classification.
    INPUT: A Pandas Series containing D1 features for a single day.
    OUTPUT: A string representing the regime (e.g., "Bullish Trend", "Bearish Trend", "Ranging").
    """
    # In reality, this function (likely in src/ai_components/kmeans_filter.py) would:
    # 1. Load your trained K-Means model and feature scaler.
    # 2. Take D1 features for the given day (d1_bar_features).
    # 3. Scale the features.
    # 4. Predict the cluster.
    # 5. Map cluster to a meaningful regime label.
    print(f"DEBUG (K-Means Placeholder): Received D1 features for regime check. (Features not shown here for brevity)")
    # Simulate some logic for now
    if 'SMA_50' in d1_bar_features and 'close' in d1_bar_features:
        if pd.notna(d1_bar_features['SMA_50']) and pd.notna(d1_bar_features['close']):
            if d1_bar_features['close'] > d1_bar_features['SMA_50'] + (0.005 * d1_bar_features['SMA_50']): # Price significantly above SMA
                return "Bullish Trend Regime" # Placeholder output
            elif d1_bar_features['close'] < d1_bar_features['SMA_50'] - (0.005 * d1_bar_features['SMA_50']): # Price significantly below SMA
                return "Bearish Trend Regime" # Placeholder output
    return "Ranging/Unfavorable Regime" # Default placeholder

def get_h1_lstm_confidence(h1_sequence_features: np.ndarray | None, # e.g., (sequence_length, num_features)
                           d1_regime_for_lstm: str,
                           h1_pattern_type_for_lstm: str) -> float:
    """
    Placeholder: Simulates LSTM H1 entry confirmation confidence.
    INPUT: 
        h1_sequence_features: A NumPy array of H1 features for the LSTM.
        d1_regime_for_lstm: The D1 K-Means regime.
        h1_pattern_type_for_lstm: The type of H1 pattern detected.
    OUTPUT: A confidence score (e.g., probability between 0.0 and 1.0).
    """
    # In reality, this function (likely in src/ai_components/lstm_handler.py) would:
    # 1. Load your trained LSTM model and feature scaler for H1 sequences.
    # 2. Take the H1 sequence features, D1 regime, H1 pattern type.
    # 3. Scale/prepare features as needed for the LSTM.
    # 4. Get a prediction (probability) from the LSTM.
    print(f"DEBUG (LSTM Placeholder): Received data for confidence check. D1 Regime='{d1_regime_for_lstm}', H1 Pattern='{h1_pattern_type_for_lstm}'")
    # Simulate some basic confidence based on D1 regime for now
    if "Bullish Trend Regime" in d1_regime_for_lstm and "W_PATTERN" in h1_pattern_type_for_lstm:
        return 0.78 # Higher confidence placeholder
    elif "Bearish Trend Regime" in d1_regime_for_lstm and "M_PATTERN" in h1_pattern_type_for_lstm:
        return 0.72 # Higher confidence placeholder
    elif "Ranging" in d1_regime_for_lstm:
        return 0.40 # Lower confidence in ranging
    return 0.60 # Default placeholder confidence

# --- 2. Main Hybrid Engine Processing Function ---

def process_hybrid_signals(
    df_h1_with_rules_signals: pd.DataFrame, # Output from signal_engine.py
    df_d1_full_structure: pd.DataFrame,     # Full D1 data with structure and SMA
    ai_assisted_tp_sl_enabled: bool = True,
    # Configurable parameters (these would ideally come from your config.json)
    # D1 Features for K-Means (example, actual features will be defined by K-Means model)
    d1_kmeans_feature_cols: list = ['close', 'SMA_50', 'd1_market_structure'], # Example only
    # LSTM Confidence Tiers
    lstm_no_trade_threshold: float = 0.55,
    lstm_tier1_min_score: float = 0.55, lstm_tier1_rr: float = 1.5,
    lstm_tier2_min_score: float = 0.65, lstm_tier2_rr: float = 2.0,
    lstm_tier3_min_score: float = 0.75, lstm_tier3_rr: float = 3.0,
    # Risk Rules Parameters
    atr_col_h1: str = 'ATR_14', low_col_h1: str = 'low', high_col_h1: str = 'high',
    close_col_h1: str = 'close', open_col_h1 = 'open', # Added open for SL/TP if needed by its logic
    sl_atr_offset_pips: float = 5.0, pip_value: float = 0.0001,
    baseline_tp_rr_ratio: float = 2.0
) -> pd.DataFrame:
    """
    Orchestrates the hybrid decision-making process.
    Takes H1 DataFrame with rule-based signals, applies AI filters/enhancements,
    and calculates final SL/TP.
    """
    if df_h1_with_rules_signals is None or df_d1_full_structure is None:
        print("ERROR (Hybrid Processor): Input DataFrames cannot be None.")
        return df_h1_with_rules_signals # Or None

    df_processed = df_h1_with_rules_signals.copy()

    # Initialize new columns for AI insights and final decisions
    df_processed['kmeans_regime'] = "N/A"
    df_processed['lstm_confidence'] = np.nan
    df_processed['final_signal'] = "NO_TRADE" # "HYBRID_BUY", "HYBRID_SELL", "NO_TRADE"
    df_processed['final_entry_price'] = np.nan
    df_processed['final_stop_loss'] = np.nan
    df_processed['final_take_profit'] = np.nan
    df_processed['decision_reason'] = ""

    # Ensure D1 index is normalized for lookup
    # This should ideally be done once before passing df_d1_full_structure if it's always used this way
    if not isinstance(df_d1_full_structure.index, pd.DatetimeIndex) or \
       not df_d1_full_structure.index.is_normalized:
        try:
            temp_d1_idx = pd.to_datetime(df_d1_full_structure.index).normalize()
            df_d1_for_lookup = df_d1_full_structure.set_index(temp_d1_idx)
        except Exception as e:
            print(f"Error normalizing D1 index: {e}. Cannot proceed with K-Means linking.")
            # Fill reason and return if D1 lookup fails
            df_processed['decision_reason'] = "D1 index normalization error for K-Means"
            return df_processed
    else:
        df_d1_for_lookup = df_d1_full_structure


    print("Processing signals with Hybrid Engine...")
    # Iterate through H1 bars that have a preliminary rule-based signal
    for index, row in df_processed[df_processed['rules_signal'] != "NO_RULE_SIGNAL"].iterrows():
        rule_signal = row['rules_signal']
        decision_reason_parts = [f"RuleSignal:{rule_signal}"]

        # --- Step 1: K-Means D1 Regime Filter ---
        h1_bar_date_normalized = index.normalize() # Normalized date of the H1 signal bar
        
        d1_bar_features_for_kmeans = None
        if h1_bar_date_normalized in df_d1_for_lookup.index:
            # Extract relevant D1 features for K-Means for that day
            # This assumes d1_kmeans_feature_cols are present in df_d1_for_lookup
            try:
                d1_bar_features_for_kmeans = df_d1_for_lookup.loc[h1_bar_date_normalized, d1_kmeans_feature_cols]
            except KeyError as e:
                print(f"KeyError accessing K-Means features for {h1_bar_date_normalized}: {e}. Available D1 cols: {df_d1_for_lookup.columns.tolist()}")
                decision_reason_parts.append("KMeansFeatureError")
                df_processed.loc[index, 'decision_reason'] = "; ".join(decision_reason_parts)
                continue # Skip to next signal
        else:
            # If D1 data for the H1 bar's day isn't found (e.g. after ffill in signal_engine, it might be stale)
            # This scenario needs careful handling or should be prevented by signal_engine.
            # For now, we'll log and potentially skip.
            # Or, we can use the D1 context already merged by signal_engine
            # Let's use the merged D1 context columns if available, assuming they were correctly ffilled.
            # The column names from signal_engine will be like 'close_d1_ctx', 'SMA_50_d1_ctx', 'd1_market_structure_d1_ctx'
            # We need to map d1_kmeans_feature_cols to these potentially suffixed names.
            # This is getting complex if we re-fetch features. Let's assume K-Means uses the already merged values.
            # For simplicity, the placeholder get_kmeans_d1_regime can use row['d1_market_structure_d1_ctx'] etc.
            # For this example, let's just pass the relevant part of the H1 row (which has D1 context merged)
            
            # Construct a series for the K-Means placeholder from the H1 row's D1 context
            # The actual K-Means function will need proper D1 features for THAT day.
            # This is a simplification for the placeholder.
            d1_context_data = {}
            # These are the merged D1 column names used in signal_engine loop
            d1_close_col_merged_in_h1 = 'close_d1_ctx' # From signal_engine
            d1_sma_col_merged_in_h1 = 'SMA_50' if 'SMA_50' in row else 'SMA_50_d1_ctx' # Check which name merged
            d1_structure_col_merged_in_h1 = 'd1_market_structure' if 'd1_market_structure' in row else 'd1_market_structure_d1_ctx'

            d1_context_data['close'] = row.get(d1_close_col_merged_in_h1)
            d1_context_data['SMA_50'] = row.get(d1_sma_col_merged_in_h1)
            d1_context_data['d1_market_structure'] = row.get(d1_structure_col_merged_in_h1)
            d1_bar_features_for_kmeans = pd.Series(d1_context_data)


        kmeans_regime = get_d1_kmeans_regime(d1_bar_features_for_kmeans)
        df_processed.loc[index, 'kmeans_regime'] = kmeans_regime
        decision_reason_parts.append(f"Regime:{kmeans_regime}")

        # Example Filtering Logic for K-Means (refine this!)
        is_regime_favorable_for_trade = True 
        if "Unfavorable" in kmeans_regime:
            is_regime_favorable_for_trade = False
        elif rule_signal == "BUY_RULE" and "Bearish" in kmeans_regime:
            is_regime_favorable_for_trade = False
        elif rule_signal == "SELL_RULE" and "Bullish" in kmeans_regime:
            is_regime_favorable_for_trade = False
            
        if not is_regime_favorable_for_trade:
            df_processed.loc[index, 'final_signal'] = "NO_TRADE"
            decision_reason_parts.append("KMeansFilter")
            df_processed.loc[index, 'decision_reason'] = "; ".join(decision_reason_parts)
            continue # Skip to next H1 rule signal

        # --- Step 2: LSTM H1 Confirmation ---
        # Placeholder: Extract H1 sequence features ending at 'index' from df_h1_with_rules_signals
        # This needs proper implementation based on how Mthabisi's LSTM expects features.
        # For example, taking the last 20 H1 bars of 'close', 'rsi', 'atr' etc.
        h1_sequence_features_placeholder = None # Actual feature extraction needed
        h1_pattern_type = "W_PATTERN" if rule_signal == "BUY_RULE" else "M_PATTERN" # Simplified

        lstm_confidence = get_h1_lstm_confidence(h1_sequence_features_placeholder, kmeans_regime, h1_pattern_type)
        df_processed.loc[index, 'lstm_confidence'] = lstm_confidence
        decision_reason_parts.append(f"LSTMConf:{lstm_confidence:.2f}")

        if lstm_confidence < lstm_no_trade_threshold:
            df_processed.loc[index, 'final_signal'] = "NO_TRADE"
            decision_reason_parts.append("LSTMLowConf")
            df_processed.loc[index, 'decision_reason'] = "; ".join(decision_reason_parts)
            continue # Skip to next H1 rule signal

        # --- Step 3: Trade is Confirmed by AI - Set final signal and SL/TP ---
        final_trade_signal = "HYBRID_BUY" if rule_signal == "BUY_RULE" else "HYBRID_SELL"
        df_processed.loc[index, 'final_signal'] = final_trade_signal
        
        current_entry_price = row[close_col_h1] # Assumed entry at close of H1 signal bar
        df_processed.loc[index, 'final_entry_price'] = current_entry_price

        # Determine TP R:R based on LSTM confidence if AI TP/SL is enabled
        current_tp_rr_to_use = baseline_tp_rr_ratio
        if ai_assisted_tp_sl_enabled:
            if lstm_confidence >= lstm_tier3_min_score:
                current_tp_rr_to_use = lstm_tier3_rr
            elif lstm_confidence >= lstm_tier2_min_score:
                current_tp_rr_to_use = lstm_tier2_rr
            elif lstm_confidence >= lstm_tier1_min_score:
                current_tp_rr_to_use = lstm_tier1_rr
            decision_reason_parts.append(f"AITP_RR:{current_tp_rr_to_use}")
        else:
            decision_reason_parts.append(f"BaselineTP_RR:{current_tp_rr_to_use}")


        # Create a single-row DataFrame for calculate_sl_tp_levels
        # It needs the 'rules_signal' to be "BUY_RULE" or "SELL_RULE"
        # and H1 ohlc, ATR for that specific bar.
        single_trade_df_data = {
            'rules_signal': rule_signal, # Use the original rule_signal
            atr_col_h1: row[atr_col_h1],
            low_col_h1: row[low_col_h1],
            high_col_h1: row[high_col_h1],
            close_col_h1: current_entry_price, # Use current_entry_price
            open_col_h1: row[open_col_h1]
        }
        single_trade_df = pd.DataFrame([single_trade_df_data], index=[index])

        # Add other columns if calculate_sl_tp_levels expects them from df_h1_with_signals
        for col in df_h1_with_rules_signals.columns:
            if col not in single_trade_df.columns and col in row:
                single_trade_df[col] = row[col]


        sl_tp_df = calculate_sl_tp_levels(
            single_trade_df, # Pass the single row DataFrame
            signal_col='rules_signal', # It looks for this column
            atr_col=atr_col_h1,
            low_col=low_col_h1,
            high_col=high_col_h1,
            close_col=close_col_h1, # This is 'entry_price_assumed' inside calculate_sl_tp_levels
            sl_atr_offset_pips=sl_atr_offset_pips,
            pip_value=pip_value,
            baseline_tp_rr_ratio=current_tp_rr_to_use 
        )
        
        if sl_tp_df is not None and not sl_tp_df.empty:
            df_processed.loc[index, 'final_stop_loss'] = sl_tp_df['stop_loss_price'].iloc[0]
            df_processed.loc[index, 'final_take_profit'] = sl_tp_df['take_profit_price'].iloc[0]
        
        df_processed.loc[index, 'decision_reason'] = "; ".join(decision_reason_parts)

    print("Hybrid signal processing complete.")
    return df_processed