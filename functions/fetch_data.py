# Updated `fetch_data.py`
import requests
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd  # Ensure pandas is imported
import parameters


def fetch_historical_data(pair, intervals, lookback_days):
    """
    Fetch all available fields of historical data from Binance for multiple intervals.

    Parameters:
    - pair: The trading pair (e.g., BTCUSDT).
    - intervals: List of timeframes to fetch (e.g., ['1m', '5m', '1h']).
    - lookback_days: How many days of historical data to fetch.

    Returns:
    - A dictionary of DataFrames, one for each interval.
    """
    now = datetime.now()
    end_time = now.replace(second=0, microsecond=0)  # Align to the last minute
    start_time = end_time - timedelta(days=lookback_days)  # Exactly the number of lookback days back

    # Convert to milliseconds for the Binance API
    end_time_ms = int(end_time.timestamp() * 1000)
    start_time_ms = int(start_time.timestamp() * 1000)

    data_frames = {}

    for interval in intervals:
        print(f"Fetching {lookback_days} day(s) of historical data for {pair} at interval {interval}...")
        all_data = []
        start = start_time_ms

        while start < end_time_ms:
            params = {
                "symbol": pair,
                "interval": interval,
                "startTime": start,
                "endTime": end_time_ms,
                "limit": 1000
            }
            try:
                response = requests.get(f"{parameters.BASE_URL}{parameters.KLINES_ENDPOINT}", params=params)
                response.raise_for_status()
                data = response.json()
                if not data:
                    break

                all_data.extend(data)
                start = data[-1][6] + 1  # Move start_time to the next batch
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for {pair} at interval {interval}: {e}")
                break

        if all_data:
            # Convert to DataFrame with all available fields
            df = pd.DataFrame(all_data, columns=[
                "open_time", "open", "high", "low", "close", "volume", "close_time",
                "quote_asset_volume", "number_of_trades", "taker_buy_base",
                "taker_buy_quote", "ignore"
            ])

            # Add Symbol and Interval columns
            df.insert(0, "symbol", pair)
            df.insert(1, "interval", interval)

            # Convert timestamp columns to datetime and ensure numeric data types
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
            numeric_columns = [
                "open", "high", "low", "close", "volume",
                "quote_asset_volume", "taker_buy_base", "taker_buy_quote"
            ]
            df[numeric_columns] = df[numeric_columns].astype(float)
            df.set_index('open_time', inplace=True)

            # Store DataFrame in the dictionary
            data_frames[interval] = df
            print(f"Fetched {len(df)} rows for interval {interval}.")
        else:
            print(f"No data returned for {pair} at interval {interval}.")


    if not data_frames:
        raise ValueError(f"No data returned for any intervals for {pair} within the {lookback_days}-day period.")

    print("All data successfully fetched.")
    return data_frames


if __name__ == "__main__":
    # Example usage
    pair = "BTCUSDT"
    intervals = ['1m', '5m', '1h', '1d']  # Add the intervals you want to fetch
    lookback_days = 7  # Number of days to look back

    try:
        # Fetch historical data for multiple intervals
        interval_data = fetch_historical_data(pair, intervals, lookback_days)

        # Example: Access and print the first 5 rows of the 5-minute interval data
        if '5m' in interval_data:
            print(interval_data['5m'].head())

        # Save each DataFrame to a CSV file (optional)
        for interval, df in interval_data.items():
            df.to_csv(f"{pair}_{interval}_data.csv", index=False)
            print(f"Saved {interval} data to {pair}_{interval}_data.csv")
    except Exception as e:
        print(f"Error: {e}")

def fetch_equity_data(EQUITY_INDICES):
    """
    Fetch equity market data (e.g., S&P 500, NASDAQ) from Yahoo Finance.

    Parameters:
    - symbols: List of equity indices (e.g., S&P 500, NASDAQ).

    Returns:
    - A dictionary with equity data.
    """
    print("Fetching equity market data (S&P 500, NASDAQ)...")
    symbols = EQUITY_INDICES
    indices_data = {}
    for symbol in symbols:
        try:
            index = yf.Ticker(symbol)
            history = index.history(period="1y")
            indices_data[symbol] = history["Close"]
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            indices_data[symbol] = pd.Series(dtype='float64')  # Empty Series for missing data

    if not indices_data:
        raise ValueError("No equity data fetched. Check the symbols or the data source.")

    print("Equity market data fetched successfully.")
    return indices_data
