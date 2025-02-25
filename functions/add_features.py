# `add_features.py`: Add Calculated Features to Resampled Data
import numpy as np
import pandas as pd

def add_advanced_features(df, equity_indices, rolling_window, interval):
    """
    Add advanced features such as volatility, cumulative volume, and seasonality.

    Parameters:
    - df: DataFrame containing price and volume data.
    - equity_data: Dictionary containing equity market data.
    - rolling_window: Window size for rolling calculations.
    - price_target_period: Period to calculate target price movement.

    Returns:
    - DataFrame with additional features.
    """
    print("Adding advanced features...")

    window = rolling_window.get(interval)

    # Validate input columns
    required_columns = ['high', 'low', 'open', 'close', 'volume']
    if not set(required_columns).issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    try:
        # Basic calculations
        df['Close_Volume_Corr'] = df['close'].rolling(20).corr(df['volume'])
        df['High_Low_Diff'] = df['high'] - df['low']
        df['Open_Close_Diff'] = df['open'] - df['close']
        df['Volatility'] = df['high'] - df['low']
        df['Cumulative_Delta_Volume'] = df['volume'].cumsum()
        df['Relative_Volume'] = df['volume'] / df['volume'].rolling(window=window).mean()
        df['Volume_Shock'] = (df['volume'] > df['volume'].rolling(window=window).mean() * 1.5).astype(int)
        df['Volume_Delta'] = df['volume'].diff()
        df['Z_Score_Volume'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()

        #Identify Key Reversal High/Low
        df['Key_Reversal_High'] = ((df['high'] > df['high'].shift(1)) &
                                   (df['close'] < df['low'].shift(1))).astype(int)
        df['Key_Reversal_Low'] = ((df['low'] < df['low'].shift(1)) &
                                  (df['close'] > df['high'].shift(1))).astype(int)

        # Add seasonality features
        df['Day_of_Week'] = df.index.dayofweek
        df['Hour_of_Day'] = df.index.hour
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter

        # Price movement target
        df['Price_Target'] = df['close'].shift(-window) - df['close']
        df['Price_Target_Label'] = np.where(df['Price_Target'] > 0, 1, 0)
        df['Price_target_Percent'] = ((df['close'].shift(-window) - df['close']) / df['close']) * 100

        # Calculate target and stop-loss levels using rolling window
        df['Target_Long'] = df['close'].rolling(window=window, min_periods=1).max().shift(-window + 1)
        df['Stop_Loss_Long'] = df['close'] - (df['close'] * 0.03)  # 2% risk for long trades

        df['Target_Short'] = df['close'].rolling(window=window, min_periods=1).min().shift(-window + 1)
        df['Stop_Loss_Short'] = df['close'] + (df['close'] * 0.03)  # 2% risk for short trades

        # Calculate risk and reward for long and short trades
        df['Reward_Long'] = df['Target_Long'] - df['close']
        df['Risk_Long'] = df['close'] - df['Stop_Loss_Long']

        df['Reward_Short'] = df['close'] - df['Target_Short']
        df['Risk_Short'] = df['Stop_Loss_Short'] - df['close']

        # Calculate Risk:Reward Ratios
        df['RR_Long'] = df['Reward_Long'] / df['Risk_Long']
        df['RR_Short'] = df['Reward_Short'] / df['Risk_Short']

        # Debug output for Risk:Reward ratios
        #print(df[['RR_Long', 'RR_Short']].describe())

        # Define thresholds using quantiles
        long_threshold = df['RR_Long'].quantile(0.90)  # Top 5% for long trades
        short_threshold = df['RR_Short'].quantile(0.90)  # Top 5% for short trades

        # Debugging: Print thresholds
        #print(f"Dynamic Long Threshold (95th Percentile): {long_threshold}")
        #print(f"Dynamic Short Threshold (95th Percentile): {short_threshold}")

        # Define labels based on dynamic thresholds
        df['Trade_Label'] = np.where(
            df['RR_Long'] >= long_threshold,  # Long trades meeting threshold
            1,
            np.where(df['RR_Short'] >= short_threshold,  # Short trades meeting threshold
                     -1,
                     0  # No trade
                     )
        )

        # Debugging: Check how many trades meet criteria
        print(f"Number of Long Trades: {len(df[df['Trade_Label'] == 1])}")
        print(f"Number of Short Trades: {len(df[df['Trade_Label'] == -1])}")
        print(f"Number of No Trades: {len(df[df['Trade_Label'] == 0])}")

        # Drop unnecessary fields
        fields_to_drop = ['Target_Long', 'Stop_Loss_Long', 'Reward_Long', 'Risk_Long',
                          'Target_Short', 'Stop_Loss_Short', 'Reward_Short', 'Risk_Short',
                          'RR_Long', 'RR_Short']
        df.drop(columns=fields_to_drop, inplace=True)

        # Binary Fractal Indicators
        df['Fractal_High_Binary'] = ((df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))).astype(int)
        df['Fractal_Low_Binary'] = ((df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))).astype(int)

        # Rolling Count of Fractals
        df['Fractal_High_Count'] = df['Fractal_High_Binary'].rolling(window=window).sum()
        df['Fractal_Low_Count'] = df['Fractal_Low_Binary'].rolling(window=window).sum()

        # Time Since Last Fractal
        df['Time_Since_Last_Fractal_High'] = (df['Fractal_High_Binary'][::-1].cumsum()[::-1] > 0).astype(int).cumsum()
        df['Time_Since_Last_Fractal_Low'] = (df['Fractal_Low_Binary'][::-1].cumsum()[::-1] > 0).astype(int).cumsum()

        # Relative Distance to Last Fractal
        df['Distance_To_Last_Fractal_High'] = df['close'] - df['high'].where(df['Fractal_High_Binary'] == 1).ffill()
        df['Distance_To_Last_Fractal_Low'] = df['close'] - df['low'].where(df['Fractal_Low_Binary'] == 1).ffill()


        print("Advanced features added successfully.")
    except Exception as e:
        raise RuntimeError(f"Error adding advanced features: {e}")

    return df

def add_multi_timeframe_features(df):
    """
    Add multi-timeframe features to the DataFrame using a resampled dataset.

    Parameters:
    - df: Base DataFrame with lower timeframe data.
    - df: Resampled DataFrame for the current interval.

    Returns:
    - DataFrame with additional multi-timeframe features.
    """
    print("Adding multi-timeframe features...")

    try:
        df['Higher_Timeframe_Close'] = df['close'].reindex(df.index, method='ffill')
        df['Higher_Timeframe_High'] = df['high'].reindex(df.index, method='ffill')
        df['Higher_Timeframe_Low'] = df['low'].reindex(df.index, method='ffill')

        print("Multi-timeframe features added successfully.")
    except Exception as e:
        raise RuntimeError(f"Error adding multi-timeframe features: {e}")

    return df

def add_lagged_features(df, max_lag):
    """
    Add lagged features to the DataFrame for historical data analysis.

    Parameters:
    - df: DataFrame with base features.
    - max_lag: Maximum number of lags to generate.

    Returns:
    - DataFrame with additional lagged features.
    """
    print(f"Adding lagged features up to {max_lag} periods...")

    try:
        for lag in range(1, max_lag + 1):
            df[f'Lag_{lag}_Close'] = df['close'].shift(lag)
            df[f'Lag_{lag}_Volume'] = df['volume'].shift(lag)

        print("Lagged features added successfully.")
    except Exception as e:
        raise RuntimeError(f"Error adding lagged features: {e}")

    return df

def detect_levels_with_trend_alignment(combined_df, TIMEFRAME_MAPPING, rolling_window, atr_multiplier=1.5, atr_lookback=20, prominence_percent=0.01):
    """Detect HH, HL, LH, LL levels with dynamic volatility and weighted scores."""

    def calculate_turning_points(df, interval, prominence_percent):
        """Calculate 'is_high' and 'is_low' for turning points with prominence."""
        try:
            window_size = rolling_window.get(interval, 3)
        except KeyError:
            print(f"Error: Interval '{interval}' not found in rolling_window.")
            return df

        df['is_high'] = False
        df['is_low'] = False

        for i in range(window_size // 2, len(df) - window_size // 2):
            window = df.iloc[i - window_size // 2:i + window_size // 2 + 1]
            current_high = df.loc[df.index[i], 'high']
            current_low = df.loc[df.index[i], 'low']

            if current_high == window['high'].max() and len(
                    window['high'].unique()) > 1:  # Added check for different high values.
                higher_than_neighbors = True
                for j in range(len(window)):
                    if j != window_size // 2 and current_high <= window.iloc[j]['high']:
                        higher_than_neighbors = False
                        break
                if higher_than_neighbors:
                    peak_value = window['high'].max()
                    try:
                        peak_to_trough = (peak_value - window['low'].min()) / peak_value
                        if peak_to_trough >= prominence_percent:
                            df.loc[df.index[i], 'is_high'] = True
                    except ZeroDivisionError:
                        pass

            if current_low == window['low'].min() and len(
                    window['low'].unique()) > 1:  # Added check for different low values.
                lower_than_neighbors = True
                for j in range(len(window)):
                    if j != window_size // 2 and current_low >= window.iloc[j]['low']:
                        lower_than_neighbors = False
                        break
                if lower_than_neighbors:
                    trough_value = window['low'].min()
                    try:
                        trough_to_peak = (window['high'].max() - trough_value) / trough_value
                        if trough_to_peak >= prominence_percent:
                            df.loc[df.index[i], 'is_low'] = True
                    except ZeroDivisionError:
                        pass

        return df

    def determine_higher_timeframe_trend(df, TIMEFRAME_MAPPING, ROLLING_WINDOW):  # Now takes the entire DataFrame as input
        """Determine higher timeframe trend for the DataFrame."""

        df['interval'] = df['interval'].astype(str)
        df.dropna(subset=['interval'], inplace=True)
        df.loc[df['interval'] == '', 'interval'] = '1m' # Corrected line using .loc
        df['interval'] = df['interval'].str.strip()

        unique_intervals = df['interval'].unique()

        for interval in unique_intervals:  # Iterate over each unique interval
            df_group = df[df['interval'] == interval]  # Create a group for the current interval

            try:
                higher_timeframe = TIMEFRAME_MAPPING[interval]
            except KeyError:
                print(
                    f"Error: No higher timeframe defined for interval '{interval}'. Skipping higher timeframe trend calculation for this interval.")
                df.loc[df['interval'] == interval, 'higher_timeframe_ma'] = None  # Update the original df
                df.loc[df['interval'] == interval, 'higher_timeframe_trend'] = None  # Update the original df
                continue  # Go to the next interval

            try:
                ma_period = ROLLING_WINDOW[higher_timeframe]
            except KeyError:
                print(
                    f"Error: Rolling window not defined for higher timeframe '{higher_timeframe}'. Skipping higher timeframe trend calculation for this interval.")
                df.loc[df['interval'] == interval, 'higher_timeframe_ma'] = None  # Update the original df
                df.loc[df['interval'] == interval, 'higher_timeframe_trend'] = None  # Update the original df
                continue  # Go to the next interval

            df_group['open_time'] = pd.to_datetime(df_group['open_time'], errors='coerce')
            df_group = df_group.set_index('open_time')

            higher_tf_df = df_group['close'].resample(higher_timeframe).agg(['first', 'last', 'max', 'min', 'mean'])

            if higher_tf_df.empty:
                print(
                    f"Warning: No data available for higher timeframe '{higher_timeframe}'. Skipping higher timeframe trend calculation for this interval.")
                df.loc[df['interval'] == interval, 'higher_timeframe_ma'] = None  # Update the original df
                df.loc[df['interval'] == interval, 'higher_timeframe_trend'] = None  # Update the original df
                continue  # Go to the next interval

            higher_tf_df['higher_timeframe_ma'] = higher_tf_df['mean'].rolling(window=ma_period).mean()
            higher_tf_df['higher_timeframe_trend'] = higher_tf_df['mean'] > higher_tf_df['higher_timeframe_ma']

            df_group = df_group.merge(higher_tf_df[['higher_timeframe_ma', 'higher_timeframe_trend']],
                                      left_index=True, right_index=True, how='left')

            df.loc[df['interval'] == interval, 'higher_timeframe_ma'] = df_group[
                'higher_timeframe_ma']  # Update the original df
            df.loc[df['interval'] == interval, 'higher_timeframe_trend'] = df_group[
                'higher_timeframe_trend']  # Update the original df

        df = df.reset_index()

        return df

    def calculate_weighted_score(row, avg_volume, higher_timeframe_trend, df):
        """Calculate weighted score."""

        score = 0

        if row['volume'] > avg_volume * 1.2:
            score += 1

        if row['logic_used'] == 'ATR':
            score += 1

        if row['level_type'] in ['HH', 'HL']:
            try:
                previous_high = df['high'].rolling(window=10).max().shift(1).iloc[-1]
                if row['low'] > previous_high * 0.95:
                    score += 1
            except (IndexError, KeyError, TypeError):
                pass

        if row['level_type'] in ['LL', 'LH']:
            try:
                previous_low = df['low'].rolling(window=10).min().shift(1).iloc[-1]
                if row['high'] < previous_low * 1.05:
                    score += 1
            except (IndexError, KeyError, TypeError):
                pass

        if row['level_type'] in ['HH', 'HL'] and higher_timeframe_trend:
            score += 2
        if row['level_type'] in ['LH', 'LL'] and not higher_timeframe_trend:
            score += 2

        return score

    def classify_levels(df, interval, atr_lookback, atr_multiplier):

        print("Before Classify Levels execution (ALL Columns and dtypes):")

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # Show all rows and columns
            print(df.dtypes)
            print(df.columns)  # Print all column names

        """Classify turning points with dynamic volatility."""

        print("Classifying levels...")

        df['level_type'] = None
        df['logic_used'] = None
        df['weighted_score'] = None

        df['dynamic_volatility_threshold'] = df['ATR'].rolling(window=atr_lookback).mean() * atr_multiplier

        high_volatility = df['ATR'] > df['dynamic_volatility_threshold']

        hh_conditions_atr = (df['is_high']) & (high_volatility) & (df['high'] > df['high'].shift(1) + df['ATR'].shift(1) * atr_multiplier)
        df.loc[hh_conditions_atr, 'level_type'] = 'HH'
        df.loc[hh_conditions_atr, 'logic_used'] = 'ATR'

        hh_conditions_candle = (df['is_high']) & (~high_volatility) & (df['close'] > df['high'].shift(1))
        df.loc[hh_conditions_candle, 'level_type'] = 'HH'
        df.loc[hh_conditions_candle, 'logic_used'] = 'Candle'

        ll_conditions_atr = (df['is_low']) & (high_volatility) & (df['low'] < df['low'].shift(1) - df['ATR'].shift(1) * atr_multiplier)
        df.loc[ll_conditions_atr, 'level_type'] = 'LL'
        df.loc[ll_conditions_atr, 'logic_used'] = 'ATR'

        ll_conditions_candle = (df['is_low']) & (~high_volatility) & (df['close'] < df['low'].shift(1))
        df.loc[ll_conditions_candle, 'level_type'] = 'LL'
        df.loc[ll_conditions_candle, 'logic_used'] = 'Candle'

        lh_conditions = (df['is_high']) & (df['high'] < df['high'].shift(1)) & (df['level_type'].shift(1) == 'HH')
        df.loc[lh_conditions, 'level_type'] = 'LH'
        df.loc[lh_conditions, 'logic_used'] = 'Candle'

        hl_conditions = (df['is_low']) & (df['low'] > df['low'].shift(1)) & (df['level_type'].shift(1) == 'LL')
        df.loc[hl_conditions, 'level_type'] = 'HL'
        df.loc[hl_conditions, 'logic_used'] = 'Candle'

        avg_volume = df['volume'].mean()
        df['weighted_score'] = df.apply(lambda row: calculate_weighted_score(row, avg_volume, row['higher_timeframe_trend'], df), axis=1)

        print("Completed classifying levels.")

        return df


    # Ensure 'interval' and 'symbol' columns exist
    required_columns = ['interval', 'symbol']
    missing_columns = [col for col in required_columns if col not in combined_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    # Group and process (including 'symbol' and 'interval' in groupby)
    processed_df = combined_df.groupby(['symbol', 'interval'], group_keys=False).apply(
        lambda group: (
            determine_higher_timeframe_trend(group, rolling_window, TIMEFRAME_MAPPING),
            calculate_turning_points(group, group['interval'].iloc[0], prominence_percent),
            classify_levels(group, group['interval'].iloc[0], atr_lookback, atr_multiplier),
        )[-1]
    )

    processed_df = processed_df.reset_index()

    print("After Detect Levels with Trend execution execution (ALL Columns and dtypes):")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # Show all rows and columns
        print(processed_df.dtypes)
        print(processed_df.columns)  # Print all column names

    return processed_df

if __name__ == "__main__":
    import pandas as pd
    from add_features import add_advanced_features, add_multi_timeframe_features, add_lagged_features

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

    # Resample the entire DataFrame
    df = mock_df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # Add features
    df = add_advanced_features(df, {}, 10, 5)
    df = add_lagged_features(df, 3)

    # Display the results
    print(df.head(20))
    print('\n'.join(df.columns))


