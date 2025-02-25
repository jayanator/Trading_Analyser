import numpy as np
import pandas as pd

def calculate_swing_points(df, rolling_window, interval):
    """
    Calculate swing points and associated metrics (e.g., distances to swing points, Fibonacci ratios).

    Parameters:
    - df: DataFrame containing OHLCV data.
    - swing_windows: Dictionary mapping intervals to rolling window sizes.
    - interval: The interval for which the DataFrame corresponds (e.g., '1m', '5m').

    Returns:
    - DataFrame with new columns for swing points and related metrics.
    """
    print("Calculating Swing Points...")
    try:
        # Determine rolling window size for the given interval
        window = rolling_window.get(interval)

        # Validate the rolling window size
        if not isinstance(window, int) or window <= 0 or window > len(df):
            raise ValueError(f"Invalid rolling window size: {window} for interval {interval}.")

        print(f"Using rolling window size: {window} for interval: {interval}")

        # Identify Swing Highs
        swing_high = np.where(
            (df['high'] == df['high'].rolling(window, center=True).max()) &
            (df['high'] > df['high'].shift(1)) &
            (df['high'] > df['high'].shift(-1)),
            df['high'], np.nan
        )

        # Identify Swing Lows
        swing_low = np.where(
            (df['low'] == df['low'].rolling(window, center=True).min()) &
            (df['low'] < df['low'].shift(1)) &
            (df['low'] < df['low'].shift(-1)),
            df['low'], np.nan
        )

        # Forward Fill Swing Points
        swing_high_ffill = pd.Series(swing_high, index=df.index).ffill().to_numpy()
        swing_low_ffill = pd.Series(swing_low, index=df.index).ffill().to_numpy()

        # Calculate Distances and Fibonacci Ratios
        dist_to_sh = df['close'] - swing_high_ffill
        dist_to_sl = df['close'] - swing_low_ffill
        fib_ratio = np.where(
            (swing_high_ffill - swing_low_ffill) != 0,
            (df['close'] - swing_low_ffill) / (swing_high_ffill - swing_low_ffill),
            np.nan
        )

        # Add to DataFrame
        df['Swing_High'] = swing_high_ffill
        df['Swing_Low'] = swing_low_ffill
        df['Dist_to_SH'] = dist_to_sh
        df['Dist_to_SL'] = dist_to_sl
        df['Fib_Ratio'] = fib_ratio

        print("Swing Points calculated...")

        return df

    except Exception as e:
        raise RuntimeError(f"Error calculating swing points: {e}")


def detect_consolidation(df, key_level, tolerance_factor=0.01, min_duration=3):
    """
    Detect consolidation zones around a key price level.

    Parameters:
    - df: DataFrame containing OHLCV data.
    - key_level: The price level to check for consolidation.
    - tolerance_factor: Percentage tolerance around the key level for defining consolidation.
    - min_duration: Minimum number of consecutive candles required to classify as consolidation.

    Returns:
    - DataFrame with a new column for consolidation zones.
    """
    try:
        tolerance = key_level * tolerance_factor
        consolidation_zone = ((df['close'] > key_level - tolerance) &
                              (df['close'] < key_level + tolerance)).astype(int)

        # Consolidation Zone
        consolidation_zone = (
            pd.Series(consolidation_zone)
            .rolling(min_duration, center=False)
            .sum()
            .fillna(0)
            .astype(int)
        )
        df['Consolidation_Zone'] = (consolidation_zone >= min_duration).astype(int)

        return df
    except Exception as e:
        raise RuntimeError(f"Error detecting consolidation zones: {e}")


def analyse_consolidation(df, key_level):
    """
    Perform an analysis on consolidation zones and calculate detailed metrics.

    Parameters:
    - df: DataFrame containing OHLCV data and consolidation zones.
    - key_level: The price level around which to analyze consolidation.

    Returns:
    - df: Updated DataFrame with consolidation-related metrics.
    - metrics: A dictionary of aggregate consolidation metrics.
    """
    try:
        # Filter for consolidation zone rows
        consolidation_rows = df[df['Consolidation_Zone'] == 1]

        # Aggregate Metrics
        bias = (consolidation_rows['close'] - key_level).mean()
        volume = consolidation_rows['volume'].sum()
        duration = len(consolidation_rows)

        # New Features
        df['Close_Volume_Corr'] = df['close'].rolling(20).corr(df['volume'])
        df['High_Low_Diff'] = df['high'] - df['low']
        df['Open_Close_Diff'] = df['open'] - df['close']
        df.loc[consolidation_rows.index, 'Consolidation_Width'] = (
            consolidation_rows['high'] - consolidation_rows['low']
        )

        # Add bias, volume, and duration to the DataFrame
        df['Bias'] = np.nan
        df['Total_Volume'] = np.nan
        df['Zone_Duration'] = np.nan

        # Populate these columns only for rows in the consolidation zone
        df.loc[consolidation_rows.index, 'Bias'] = bias
        df.loc[consolidation_rows.index, 'Total_Volume'] = volume
        df.loc[consolidation_rows.index, 'Zone_Duration'] = duration

        return df

    except Exception as e:
        raise RuntimeError(f"Error analyzing consolidation zones: {e}")



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
    mock_df = pd.DataFrame(mock_data).set_index('open_time')

    swing_windows = {'1m': 5, '5m': 10, '15m': 20, '1h': 30}
    interval = '1m'  # Set the specific interval
    mock_df.attrs['interval'] = interval

    swing_df = calculate_swing_points(mock_df, swing_windows, interval)
    consolidation_df = detect_consolidation(swing_df, key_level=105)
    consolidation_df, metrics = analyse_consolidation(consolidation_df, key_level=105)

    print(swing_df[['close', 'Swing_High', 'Swing_Low', 'Dist_to_SH', 'Dist_to_SL', 'Fib_Ratio']].head())
    print(metrics)
