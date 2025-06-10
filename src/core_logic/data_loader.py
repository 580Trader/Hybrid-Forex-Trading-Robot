# src/core_logic/data_loader.py
import pandas as pd

def load_price_data_from_csv(file_path, datetime_col_name='time', 
                             open_col='open', high_col='high', 
                             low_col='low', close_col='close', 
                             volume_col='tick_volume', desired_timezone='UTC'):
    """
    Loads historical price data from a CSV file into a Pandas DataFrame.

    Parameters:
    - file_path (str): The full path to the CSV file.
    - datetime_col_name (str): The name of the column containing datetime information.
                               This could be a combined datetime or just a date for D1 data.
    - open_col, high_col, low_col, close_col, volume_col (str): Column names for OHLCV data.
    - desired_timezone (str): The timezone to localize the datetime index to (e.g., 'UTC').
                               Pass None if the data is already timezone-aware or you want naive.

    Returns:
    - pd.DataFrame: DataFrame with a datetime index and standardized column names 
                      ('open', 'high', 'low', 'close', 'tick_volume'), or None if loading fails.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Convert the datetime column to pandas datetime objects
        df[datetime_col_name] = pd.to_datetime(df[datetime_col_name])

        # Set the datetime column as the index
        df = df.set_index(datetime_col_name)

        column_map = {
            open_col: 'open',
            high_col: 'high',
            low_col: 'low',
            close_col: 'close',
            volume_col: 'tick_volume'
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        
        # Select only the standard columns we need, if they exist
        standard_cols = ['open', 'high', 'low', 'close']
        if 'tick_volume' in df.columns: 
            standard_cols.append('tick_volume')
        
        # Filter to keep only essential columns if they were successfully renamed
        existing_standard_cols = [col for col in standard_cols if col in df.columns]
        if len(existing_standard_cols) < 4: # Check if we have at least OHLC
            print(f"Warning: Could not find all standard OHLC columns in {file_path} after renaming.")
            # Decide how to handle this: return None, or df with available columns? For now, proceed with available.

        df = df[existing_standard_cols]


        # Ensure OHLC data is numeric
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') # errors='coerce' turns non-numeric into NaN

        if 'tick_volume' in df.columns:
             df['tick_volume'] = pd.to_numeric(df['tick_volume'], errors='coerce')


        # Sort the DataFrame by index (time)
        df = df.sort_index()

        # Handle timezone (important for consistency)
        if desired_timezone:
            if df.index.tz is None:
                df = df.tz_localize(desired_timezone)
            else:
                df = df.tz_convert(desired_timezone)
        
        print(f"Successfully loaded and processed data from: {file_path}")
        return df

    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading data from {file_path}: {e}")
        return None

if __name__ == '__main__':
    sample_d1_file = '../../data/splits/GBPUSDm_D1_train.csv' 
    
    # Assuming your D1 CSV has a 'Date' column for datetime, and standard OHLCV names
    df_d1_test = load_price_data_from_csv(
        file_path=sample_d1_file,
        datetime_col_name='time',
        open_col='open', 
        high_col='high',
        low_col='low',
        close_col='close',
        volume_col='tick_volume'
    )

    if df_d1_test is not None:
        print("\nSample D1 Data Loaded:")
        print(df_d1_test.head())
        print("\nData Info:")
        df_d1_test.info()