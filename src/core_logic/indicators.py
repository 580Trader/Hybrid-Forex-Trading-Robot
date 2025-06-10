import pandas as pd

#Function to calculate 50 SMA
def calculate_sma(dataframe, price_column='close', period=50, sma_column_name=None):
    if not isinstance(dataframe, pd.DataFrame) or price_column not in dataframe.columns:
        print(f"Error: Invalid DataFrame or price column '{price_column}' not found.")
        return None

    if sma_column_name is None:
        sma_column_name = f'SMA_{period}' 

    try:
        df_copy = dataframe.copy()
        df_copy[sma_column_name] = df_copy[price_column].rolling(window=period).mean()
        return df_copy
    except Exception as e:
        print(f"Error calculating SMA: {e}")
        return None
    
#Function to calculate ATR
def calculate_atr(dataframe, high_col='high', low_col='low', close_col='close', period=14, atr_column_name=None):
    if not isinstance(dataframe, pd.DataFrame) or \
       not all(col in dataframe.columns for col in [high_col, low_col, close_col]):
        print(f"Error: Invalid DataFrame or missing HLC columns ('{high_col}', '{low_col}', '{close_col}') for ATR.")
        return None

    if atr_column_name is None:
        atr_column_name = f'ATR_{period}'

    try:
        df_copy = dataframe.copy()

        # Calculate True Range (TR)
        df_copy['h_minus_l'] = df_copy[high_col] - df_copy[low_col]
        df_copy['h_minus_pc'] = abs(df_copy[high_col] - df_copy[close_col].shift(1))
        df_copy['l_minus_pc'] = abs(df_copy[low_col] - df_copy[close_col].shift(1))

        df_copy['true_range'] = df_copy[['h_minus_l', 'h_minus_pc', 'l_minus_pc']].max(axis=1)

        # Calculate ATR using Wilder's smoothing (an EMA can approximate this)
        # For the first ATR value, it's a simple average of the first 'period' TR values.
        # Subsequent values use smoothing: ATR = (Previous_ATR * (period - 1) + Current_TR) / period
        # Pandas ewm can do this with alpha = 1/period
        df_copy[atr_column_name] = df_copy['true_range'].ewm(alpha=1/period, adjust=False, min_periods=period).mean()

        # Clean up temporary columns
        df_copy.drop(columns=['h_minus_l', 'h_minus_pc', 'l_minus_pc', 'true_range'], inplace=True)

        return df_copy
    except Exception as e:
        print(f"Error calculating ATR: {e}")
        return None
    

#Function to calculate the RSI
def calculate_rsi(dataframe, price_column='close', period=10, rsi_column_name=None):
    if not isinstance(dataframe, pd.DataFrame) or price_column not in dataframe.columns:
        print(f"Error: Invalid DataFrame or price column '{price_column}' not found for RSI.")
        return None

    if rsi_column_name is None:
        rsi_column_name = f'RSI_{period}'

    try:
        df_copy = dataframe.copy()
        delta = df_copy[price_column].diff(1) # Price change from previous period

        gain = delta.copy()
        gain[gain < 0] = 0 # Gains are positive changes or zero

        loss = delta.copy()
        loss[loss > 0] = 0 # Losses are negative changes or zero
        loss = abs(loss) # Make losses positive for calculation

        # Calculate Average Gain and Average Loss using Wilder's smoothing 
        # For first avg_gain/avg_loss, it's a simple average of first 'period' gains/losses.
        # Subsequent values use smoothing. 
        avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss

        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))

        # Handle cases where avg_loss is zero (to prevent division by zero and inf RSI)
        rsi[avg_loss == 0] = 100 # If no losses, RSI is 100
        rsi[avg_gain == 0] = 0   # If no gains (and some losses), RSI is 0 (this is implicitly handled by rs being 0)


        df_copy[rsi_column_name] = rsi
        return df_copy
    except Exception as e:
        print(f"Error calculating RSI: {e}")
        return None

