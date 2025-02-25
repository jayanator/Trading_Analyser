import pandas as pd
import numpy as np

def add_multi_class_labels(df, interval):
    """
    Add multi-class labels to the DataFrame for trading predictions.

    Parameters:
    - df: DataFrame containing OHLCV and feature data.
    - interval: Time interval for classifying the labels (e.g., '1m', '5m').

    Returns:
    - DataFrame with added multi-class label columns.
    """
    print(f"Adding multi-class labels for interval: {interval}...")

    # Validate input DataFrame
    required_columns = ['close']
    if not set(required_columns).issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    try:
        # 1) Define ATR-based thresholds
        atr_multiplier = 0.5  # tweak this value as you like
        ATR_up_threshold = (df['ATR'] / df['close']) * atr_multiplier
        ATR_down_threshold = -(df['ATR'] / df['close']) * atr_multiplier

        # 2) Calculate future price change
        ATR_Future_Close = df['close'].shift(-1)
        ATR_price_change = (ATR_Future_Close - df['close']) / df['close']

        # 3) Create the multi-class label
        multi_class_label = np.where(
            ATR_price_change > ATR_up_threshold,
            1,
            np.where(
                ATR_price_change < ATR_down_threshold,
                -1,
                0
            )
        )

        # Add all new columns in one operation
        new_columns = pd.DataFrame({
            'ATR_up_threshold': ATR_up_threshold,
            'ATR_down_threshold': ATR_down_threshold,
            'ATR_Future_Close': ATR_Future_Close,
            'ATR_price_change': ATR_price_change,
            'Multi_Class_Label': multi_class_label
        })

        # Concatenate the new columns to the original DataFrame
        df = pd.concat([df, new_columns], axis=1)

        # Drop intermediate columns if necessary
        #df.drop(columns=['ATR_Future_Close', 'ATR_price_change'], inplace=True)

        print("Multi-class labels added successfully.")
    except Exception as e:
        raise RuntimeError(f"Error adding multi-class labels: {e}")

    return df

if __name__ == "__main__":
    # Example usage (test case)
    mock_data = {
        'open_time': pd.date_range(start='2023-01-01', periods=100, freq='T'),
        'open': [100 + i for i in range(100)],
        'high': [105 + i for i in range(100)],
        'low': [95 + i for i in range(100)],
        'close': [102 + i for i in range(100)],
        'volume': [1000 + i for i in range(100)]
    }
    mock_df = pd.DataFrame(mock_data).set_index('open_time')

    labeled_df = add_multi_class_labels(mock_df, '1m')
    print(labeled_df.head())
    print('\n'.join(labeled_df.columns))
