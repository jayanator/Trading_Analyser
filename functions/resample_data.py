import pandas as pd

def resample_data(df, resample_intervals):
    """
    Resample the given DataFrame to multiple specified time intervals,
    excluding incomplete intervals.

    Parameters:
    - df: Original DataFrame with OHLCV data.
    - resample_intervals: List of intervals to resample data to (e.g., ['5min', '15min', '1H']).

    Returns:
    - Dictionary where keys are intervals and values are resampled DataFrames.
    """
    print("Resampling data...")

    # Validate input DataFrame
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not set(required_columns).issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    if not isinstance(resample_intervals, list) or not all(isinstance(i, str) for i in resample_intervals):
        raise ValueError("resample_intervals must be a list of strings, e.g., ['5T', '15T', '1H'].")

    resampled_dfs = {}

    try:
        # Ensure 'open_time' is set as the index
        if 'open_time' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('open_time')
            print("Set 'open_time' as the index.")

        # Ensure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame index must be a DatetimeIndex for resampling.")

        for interval in resample_intervals:
            # Replace 'T' with 'min' only for resampling, leave the key unchanged
            resample_interval_clean = interval.replace('T', 'min') if 'T' in interval else interval

            # Calculate the truncation point for the last complete interval
            last_complete_time = df.index[-1].floor(resample_interval_clean)

            # Truncate the data to exclude incomplete intervals
            truncated_df = df[df.index < last_complete_time]

            print(f"Resampling to interval: {interval}")
            resampled = truncated_df.resample(resample_interval_clean).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })

            # Drop any intervals with missing values
            resampled.dropna(inplace=True)

            if resampled.empty:
                print(f"Warning: Resampled data for interval {interval} is empty.")

            # Save to CSV with interval in the file name
            resampled.to_csv(f"resampled_{interval}.csv", index=True)

            # Store in the dictionary using the original key
            resampled_dfs[interval] = resampled

        print("Data resampling completed successfully.")
    except Exception as e:
        raise RuntimeError(f"Error during resampling: {e}")

    return resampled_dfs



if __name__ == "__main__":
    # Example usage (test case)
    mock_data = {
        'open_time': pd.date_range(start='2023-01-01', periods=100, freq='min'),
        'open': [100 + i for i in range(100)],
        'high': [105 + i for i in range(100)],
        'low': [95 + i for i in range(100)],
        'close': [102 + i for i in range(100)],
        'volume': [1000 + i for i in range(100)]
    }
    mock_df = pd.DataFrame(mock_data)

    resample_intervals = ['5min', '15min']
    resampled = resample_data(mock_df, resample_intervals)

    for interval, df in resampled.items():
        print(f"Resampled DataFrame for interval {interval}:")
        print(df.head())
