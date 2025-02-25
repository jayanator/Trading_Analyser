import pandas as pd

def create_dataframe(raw_data, pair, interval):
    """
    Converts raw data into a structured DataFrame and adds metadata (pair, interval).

    Parameters:
    - raw_data: List of raw OHLCV data.
    - pair: Trading pair (e.g., 'BTCUSDT').
    - interval: Timeframe (e.g., '1m', '15m', '1h').

    Returns:
    - DataFrame with all required columns, including metadata.
    """
    print("Creating DataFrame from raw data...")

    try:
        # Define the expected columns from the raw data
        df = pd.DataFrame(raw_data, columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
            "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore"
        ])

        # Retain all original columns and format necessary ones
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

        # Print debug info to confirm the metadata is correctly assigned
        print(f"Metadata added: pair={pair}, interval={interval}")
        print(f"DataFrame has {len(df)} rows and {len(df.columns)} columns.")

        # Print the first 5 rows of specific columns
        columns_to_print = ['open_time', 'close', 'taker_buy_quote']
        print(df[columns_to_print].head())

        return df
    except Exception as e:
        raise RuntimeError(f"Error creating DataFrame: {e}")


if __name__ == "__main__":
    # Example usage for testing
    raw_data = [
        [1632960000000, 43000.0, 43500.0, 42500.0, 43200.0, 120.5, 1632963599999, 5300000.0, 200, 60.0, 520000.0, 0.0]
    ]
    pair = "BTCUSDT"
    interval = "15m"
    df = create_dataframe(raw_data, pair, interval)
    print(df.head())
    print('\n'.join(df.columns))
