# `add_indicators_and_patterns.py`: Add Technical Indicators and Patterns
import pandas as pd
import numpy as np
import pandas_ta as ta

def add_indicators(df):
    """
    Add technical indicators to the DataFrame, including RSI, EMA, Bollinger Bands, ATR, Ichimoku, PSAR, OBV, and MFI.

    Parameters:
    - df: DataFrame containing OHLCV data.

    Returns:
    - DataFrame with additional technical indicators.
    """
    print("Adding technical indicators...")

    # Validate input columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not set(required_columns).issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    try:
        # Add Relative Strength Index (RSI)
        df['RSI'] = ta.rsi(df['close'], length=14)

        # Add Exponential Moving Averages (EMA)
        df['EMA_20'] = ta.ema(df['close'], length=20)
        df['EMA_50'] = ta.ema(df['close'], length=50)

        # Add Bollinger Bands
        bb = ta.bbands(df['close'], length=20, std=2)
        if isinstance(bb, pd.DataFrame):
            df['BB_upper'] = bb.iloc[:, 0]
            df['BB_middle'] = bb.iloc[:, 1]
            df['BB_lower'] = bb.iloc[:, 2]

        # Fisher Transform
        epsilon = 1e-9  # Small constant to handle precision errors
        df['Fisher_Transform'] = (
                2 * ((df['close'] - df['low'].rolling(10).min()) /
                     (df['high'].rolling(10).max() - df['low'].rolling(10).min() + epsilon)) - 1
        ).clip(-1 + epsilon, 1 - epsilon).apply(np.arctanh)

        # Add Average True Range (ATR)
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        #Normalised ATR
        df['nATR'] = df['ATR'] / df['close']

        #Calculate Chandelier Exit
        df['Chandelier_Exit'] = df['high'].rolling(22).max() - (3 * df['ATR'])

        # Add Ichimoku Cloud
        ichimoku = ta.ichimoku(df['high'], df['low'], df['close'], tenkan=9, kijun=26,
                               senkou=52)

        if isinstance(ichimoku, tuple) and ichimoku[0] is not None:
            ichimoku_df = ichimoku[0]
            df['Ichimoku_Tenkan'] = ichimoku_df.get('ITS_9', np.nan)
            df['Ichimoku_Kijun'] = ichimoku_df.get('IKS_26', np.nan)
            df['Ichimoku_Senkou_A'] = ichimoku_df.get('ISA_9', np.nan)
            df['Ichimoku_Senkou_B'] = ichimoku_df.get('ISB_26', np.nan)
        else:
            print("Ichimoku calculation returned an unexpected output. Columns will be filled with NaN.")
            df['Ichimoku_Tenkan'] = np.nan
            df['Ichimoku_Kijun'] = np.nan
            df['Ichimoku_Senkou_A'] = np.nan
            df['Ichimoku_Senkou_B'] = np.nan

        # Add Parabolic SAR (PSAR)
        psar_df = ta.psar(df['high'], df['low'], df['close'], step=0.02, max_step=0.2)
        if isinstance(psar_df, pd.DataFrame):
            # print(psar_df.columns)  # to see the actual column names
            # Often you'll see something like:
            # Index(['PSARl_0.02_0.2', 'PSARs_0.02_0.2', 'PSAR_0.02_0.2'], dtype='object')

            # Combine the long and short columns into one continuous PSAR
            df['PSAR'] = psar_df['PSARl_0.02_0.2'].combine_first(psar_df['PSARs_0.02_0.2'])

        # Add On-Balance Volume (OBV)
        df['OBV'] = ta.obv(df['close'], df['volume'])

        # Add Money Flow Index (MFI)
        df['MFI'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)

        # Add Stochastic Oscillator (K% and D%)
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        if isinstance(stoch, pd.DataFrame):
            df['Stoch_K'] = stoch.iloc[:, 0]  # %K line
            df['Stoch_D'] = stoch.iloc[:, 1]  # %D line

        # Add Williams %R
        df['Williams_R'] = ta.willr(df['high'], df['low'], df['close'], length=14)

        # Add Commodity Channel Index (CCI)
        df['CCI'] = ta.cci(df['high'], df['low'], df['close'], length=20)

        # Add MACD Histogram
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if isinstance(macd, pd.DataFrame):
            df['MACD_Line'] = macd.iloc[:, 0]
            df['Signal_Line'] = macd.iloc[:, 1]
            df['MACD_Histogram'] = macd.iloc[:, 2]

        # Add Average Directional Index (ADX)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if isinstance(adx, pd.DataFrame):
            df['ADX'] = adx.iloc[:, 0]  # ADX value
            df['DIpos'] = adx.iloc[:, 1]  # Positive Directional Indicator
            df['DIneg'] = adx.iloc[:, 2]  # Negative Directional Indicator

        # Add Keltner Channels
        kc = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=2)
        if isinstance(kc, pd.DataFrame):
            df['KC_upper'] = kc.iloc[:, 0]
            df['KC_middle'] = kc.iloc[:, 1]
            df['KC_lower'] = kc.iloc[:, 2]

        # Add Supertrend Indicator
        supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
        if isinstance(supertrend, pd.DataFrame):
            df['Supertrend'] = supertrend.iloc[:, 0]  # Trend (1 or -1)

        # Add Volume Weighted Average Price (VWAP)
        df['VWAP'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])

        # Add Chaikin Money Flow (CMF)
        df['CMF'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)

        # Simplified Volume Profile (e.g., cumulative sum of volume over price levels)
        df['Cumulative_Volume'] = df['volume'].cumsum()

        # Add Donchian Channels
        donchian = ta.donchian(df['high'], df['low'], lower_length=20, upper_length=20)
        if isinstance(donchian, pd.DataFrame):
            df['Donchian_Upper'] = donchian.iloc[:, 0]
            df['Donchian_Lower'] = donchian.iloc[:, 1]

        # Add Choppiness Index
        df['Choppiness_Index'] = ta.chop(df['high'], df['low'], df['close'], length=14)

        # Add Historical Volatility (calculated manually)
        df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))
        df['Historical_Volatility'] = df['Log_Returns'].rolling(window=14).std() * np.sqrt(252)  # Annualized volatility
        df.drop(columns=['Log_Returns'], inplace=True)

        # Add Pivot Points (Classic formula)
        df['Pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['R1'] = 2 * df['Pivot'] - df['low']
        df['S1'] = 2 * df['Pivot'] - df['high']
        df['R2'] = df['Pivot'] + (df['high'] - df['low'])
        df['S2'] = df['Pivot'] - (df['high'] - df['low'])

        fractal_high = np.where(
            (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1)),
            df['high'],
            np.nan
        )
        fractal_low = np.where(
            (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1)),
            df['low'],
            np.nan
        )

        # 2) Forward fill the arrays
        fractal_high_ffill = pd.Series(fractal_high, index=df.index).ffill().to_numpy()
        fractal_low_ffill = pd.Series(fractal_low, index=df.index).ffill().to_numpy()

        # 3) Assign them back to the DataFrame
        df['Fractal_High'] = fractal_high_ffill
        df['Fractal_Low'] = fractal_low_ffill

        # Add Gann Levels (Support and Resistance)
        gann_levels = [1 / 8, 2 / 8, 3 / 8, 4 / 8, 5 / 8, 6 / 8, 7 / 8]
        for level in gann_levels:
            df[f'Gann_Level_Res_{int(level * 8)}'] = df['close'] * (1 + level)
            df[f'Gann_Level_Sup_{int(level * 8)}'] = df['close'] * (1 - level)

        # Add Rate of Change (ROC)
        df['ROC_14'] = ta.roc(df['close'], length=14)

        # Add Mean-Reverting Features (z-scores)
        rolling_mean = df['close'].rolling(window=20).mean()
        rolling_std = df['close'].rolling(window=20).std()
        df['Z_Score'] = (df['close'] - rolling_mean) / rolling_std

        # Add Rolling Sharpe Ratio
        rolling_return = df['close'].pct_change().rolling(window=14).mean()
        rolling_volatility = df['close'].pct_change().rolling(window=14).std()
        df['Sharpe_Ratio'] = rolling_return / rolling_volatility

        # Add Sortino Ratio (Downside risk focus)
        epsilon = 1e-9  # Small constant to prevent division by zero

        downside_returns = df['close'].pct_change()
        downside_returns[downside_returns > 0] = 0  # Only consider negative returns

        rolling_downside_std = downside_returns.rolling(window=14).std()
        rolling_return = df['close'].pct_change().rolling(window=14).mean()

        # Compute Sortino Ratio with a safeguard for zero denominator
        df['Sortino_Ratio'] = rolling_return / (rolling_downside_std + epsilon)

        # Optionally, replace invalid ratios with NaN or a default value
        df.loc[rolling_return.isna() | rolling_downside_std.isna(), 'Sortino_Ratio'] = np.nan

        # Add Max Drawdown
        rolling_max = df['close'].rolling(window=14).max()
        rolling_drawdown = (df['close'] - rolling_max) / rolling_max  # Percentage drawdown
        df['Max_Drawdown'] = rolling_drawdown.rolling(window=14).min()  # Maximum drawdown within the window

        # Add Liquidity Ratio
        df['Liquidity_Ratio'] = df['volume'] / (df['high'] - df['low']).replace(
            0, np.nan)  # Avoid division by zero

        # Add Expanded Correlation Features
        correlation_pairs = [
            ('close', 'volume'),  # Close price and volume correlation
            ('high', 'low'),  # High and low price correlation
            ('close', 'open'),  # Close price and open price correlation
        ]

        # Calculate rolling correlations for each pair
        for col1, col2 in correlation_pairs:
            correlation_name = f'Corr_{col1}_{col2}'
            df[correlation_name] = df[col1].rolling(window=14).corr(df[col2])

            # Handle invalid correlation values
            df[correlation_name] = df[correlation_name].fillna(0)  # Replace NaN with 0
            df[correlation_name] = df[correlation_name].clip(-1, 1)  # Clip values to [-1, 1]

        print("Technical indicators added successfully.")
    except Exception as e:
        raise RuntimeError(f"Error adding technical indicators: {e}")

    return df

def add_patterns(df):
    """
    Add candlestick patterns to the DataFrame.

    Parameters:
    - df: DataFrame containing OHLCV data.

    Returns:
    - DataFrame with additional candlestick pattern columns.
    """
    print("Adding candlestick patterns...")

    # Validate input columns
    required_columns = ['open', 'high', 'low', 'close']
    if not set(required_columns).issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    try:
        # Example pattern: Engulfing Pattern
        df['Engulfing'] = np.where(
            ((df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))) |
            ((df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1))),
            1, 0
        )

        # Example pattern: Doji
        df['Doji'] = np.where(
            (abs(df['close'] - df['open']) / (df['high'] - df['low']) < 0.1),
            1, 0
        )

        df['Marubozu'] = ((df['high'] - df['low']) == (df['close'] - df['open'])).astype(int)

        #Identify Three Line Strike Candle Pattern
        df['Three_Line_Strike'] = ((df['close'] > df['open']) &
                                   (df['close'].shift(1) > df['open'].shift(1)) &
                                   (df['close'].shift(2) > df['open'].shift(2)) &
                                   (df['close'].shift(-1) < df['open'].shift(-1))).astype(int)

        print("Candlestick patterns added successfully.")
    except Exception as e:
        raise RuntimeError(f"Error adding candlestick patterns: {e}")

    return df

def calculate_rolling_vwap(df, rolling_window, interval):
    """
    Calculate a rolling VWAP over a specified window.

    Parameters:
    - df: DataFrame containing 'close' and 'volume' columns.
    - window: Rolling window size (e.g., 20 for 20 periods).

    Returns:
    - Series representing the rolling VWAP.
    """

    print("Calculating Rolling VWAP...")

    window = rolling_window.get(interval)

    if 'close' not in df.columns or 'volume' not in df.columns:
        raise ValueError("The DataFrame must contain 'close' and 'volume' columns.")

    # Calculate the cumulative Price × Volume and Volume
    rolling_price_volume = (df['close'] * df['volume']).rolling(window=window).sum()
    rolling_volume = df['volume'].rolling(window=window).sum()

    # Calculate the rolling VWAP
    df['Rolling_VWAP'] = rolling_price_volume / rolling_volume

    print("Rolling VWAP Calculated...")

    return df



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

    df_with_indicators = add_indicators(mock_df)
    df_with_patterns = add_patterns(df_with_indicators)

    print(df_with_patterns.head())
    print('\n'.join(df_with_patterns.columns))