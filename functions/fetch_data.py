import requests
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import parameters
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_historical_data(pair, interval, lookback_days):
    """
    Fetch historical data for a given trading pair and interval from Binance.

    Parameters:
    - pair: Trading pair (e.g., 'BTCUSDT').
    - interval: Timeframe to fetch (e.g., '5m').
    - lookback_days: Number of days to look back.

    Returns:
    - A DataFrame of historical data with a datetime index.
    """
    now = datetime.now()
    end_time = now.replace(second=0, microsecond=0)
    start_time = end_time - timedelta(days=lookback_days)

    end_time_ms = int(end_time.timestamp() * 1000)
    start_time_ms = int(start_time.timestamp() * 1000)

    all_data = []
    start = start_time_ms

    logger.info(f"Fetching {lookback_days} day(s) of historical data for {pair} at interval {interval}...")

    while start < end_time_ms:
        params_req = {
            "symbol": pair,
            "interval": interval,
            "startTime": start,
            "endTime": end_time_ms,
            "limit": 1000
        }
        try:
            response = requests.get(f"{parameters.BASE_URL}{parameters.KLINES_ENDPOINT}", params=params_req)
            response.raise_for_status()
            data = response.json()
            if not data:
                break

            all_data.extend(data)
            # Advance start time to one millisecond after the last candle's close_time (index 6)
            start = data[-1][6] + 1
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data for {pair} at interval {interval}: {e}")
            break

    if not all_data:
        raise ValueError(f"No data returned for {pair} at interval {interval}.")

    # Create DataFrame with all available fields
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    # Insert symbol and interval columns
    df.insert(0, "symbol", pair)
    df.insert(1, "interval", interval)

    # Convert timestamps to datetime
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    # Ensure numeric columns are floats
    numeric_columns = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "taker_buy_base", "taker_buy_quote"
    ]
    df[numeric_columns] = df[numeric_columns].astype(float)
    df.set_index("open_time", inplace=True)

    logger.info(f"Fetched {len(df)} rows for interval {interval}.")
    return df


def add_equity_data_to_df(df, equity_indices):
    """
    Fetch equity market data from Yahoo Finance and add the equity 'Close' prices as new columns
    in the trading pair data. Equity data is up-sampled to match the frequency of the pair data.

    For each equity index in equity_indices, the function fetches historical 'Close' prices over
    the date range of df, removes the timezone from the equity data (making it timezone-naive),
    and reindexes the series to match the trading data.

    Parameters:
    - df: DataFrame with a datetime index.
    - equity_indices: List of equity symbols (e.g., ['^GSPC', '^IXIC']).

    Returns:
    - The updated DataFrame with new equity price columns.
    """
    df = df.copy()
    start_date = df.index.min().strftime("%Y-%m-%d")
    end_date = df.index.max().strftime("%Y-%m-%d")
    logger.info(f"Equity fetch date range: {start_date} to {end_date}")

    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq is None:
        inferred_freq = '1min'
        logger.warning("Could not infer frequency from pair data; defaulting to 1min.")
    else:
        logger.info(f"Inferred pair data frequency: {inferred_freq}")

    for symbol in equity_indices:
        try:
            ticker = yf.Ticker(symbol)
            history = ticker.history(start=start_date, end=end_date)
            equity_series = history["Close"]
            logger.info(f"Equity series for {symbol} (raw): {equity_series.head()}")
            logger.info(f"Equity series length: {len(equity_series)}")

            if len(equity_series) < 3:
                logger.error(f"Not enough data for {symbol} to infer frequency. Filling column with NaN values.")
                equity_series = pd.Series([float('nan')] * len(df), index=df.index)
            else:
                equity_series.index = pd.to_datetime(equity_series.index)
                equity_series = equity_series.sort_index()
                # Remove timezone from equity series
                equity_series.index = equity_series.index.tz_localize(None)

                equity_freq = pd.infer_freq(equity_series.index)
                # If frequency is "B" (business day), convert to "1D"
                if equity_freq == "B":
                    logger.info(f"Equity frequency for {symbol} is 'B'. Converting to '1D'.")
                    equity_freq = "1D"

                logger.info(f"Inferred frequency for {symbol}: {equity_freq}")

                if equity_freq is None:
                    logger.warning(
                        f"Could not infer frequency for equity {symbol}; using direct reindex with forward-fill.")
                    equity_series = equity_series.reindex(df.index, method='ffill')
                else:
                    try:
                        equity_timedelta = pd.Timedelta(equity_freq)
                    except Exception as ex:
                        logger.error(f"Error converting equity frequency {equity_freq} to Timedelta for {symbol}: {ex}")
                        equity_timedelta = pd.Timedelta("1D")
                    pair_timedelta = pd.Timedelta(inferred_freq)
                    if pair_timedelta < equity_timedelta:
                        logger.info(f"Resampling equity {symbol} from {equity_freq} to {inferred_freq} frequency.")
                        equity_series = equity_series.resample(inferred_freq).ffill()
                    equity_series = equity_series.reindex(df.index, method='ffill')

            col_name = symbol.replace("^", "").replace(" ", "_")
            df[col_name] = equity_series
            logger.info(f"Added equity data for {symbol} as column '{col_name}'.")
        except Exception as e:
            logger.error(f"Error fetching equity data for {symbol}: {e}")
    return df


def fetch_data_pipeline(pair, interval, lookback_days, equity_indices):
    """
    Combined pipeline: fetch historical trading data for a given pair, then add equity data.

    Parameters:
    - pair: Trading pair (e.g., 'BTCUSDT').
    - interval: Data interval (e.g., '5m').
    - lookback_days: Number of days to look back.
    - equity_indices: List of equity symbols (e.g., ['^GSPC', '^IXIC']).

    Returns:
    - A DataFrame with trading data and corresponding equity prices.
    """
    try:
        df = fetch_historical_data(pair, interval, lookback_days)
        logger.info(f"Historical data sample for {pair} at interval {interval}:\n{df.head()}")

        df = add_equity_data_to_df(df, equity_indices)
        logger.info(f"Data after adding equity columns:\n{df.head()}")
        return df
    except Exception as e:
        logger.error(f"Error in fetch_data_pipeline: {e}")
        raise


if __name__ == "__main__":
    pair = "BTCUSDT"
    interval = "5m"
    lookback_days = 7
    equity_indices = parameters.EQUITY_INDICES if hasattr(parameters, 'EQUITY_INDICES') else ['^GSPC', '^IXIC']

    try:
        final_df = fetch_data_pipeline(pair, interval, lookback_days, equity_indices)
        output_csv = f"{pair}_{interval}_with_equity.csv"
        final_df.to_csv(output_csv)
        logger.info(f"Saved final data with equity to {output_csv}.")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
