# `main.py`: Pipeline Execution for Data Processing
import sqlite3
import pandas as pd

from functions.process_df import preprocess_dataframe_for_db
from parameters import DB_NAME, intervals, pairs, lookback_days, EQUITY_INDICES, rolling_window, MAX_LAG, SCHEMA_FIELDS, TIMEFRAME_MAPPING
from functions.fetch_data import fetch_historical_data, fetch_equity_data
from functions.add_features import add_advanced_features, add_lagged_features, detect_levels_with_trend_alignment
from functions.add_indicators_and_patterns import add_indicators, add_patterns, calculate_rolling_vwap
from functions.add_multi_class_labels import add_multi_class_labels
from functions.key_levels import calculate_swing_points, detect_consolidation, analyse_consolidation
from functions.save_to_db import initialise_and_save_to_db


def run_pipeline():
    """
    Execute the data processing pipeline step by step.
    """
    print("Starting data processing pipeline...")

    try:
        for pair in pairs:
            print(f"Processing pair: {pair}")

            combined_df = pd.DataFrame()

            # Fetch raw data
            data_frames = fetch_historical_data(pair, intervals, lookback_days)

            for key, df in data_frames.items():
                print(f"DataFrame for key: {key}")
                print(df.head())  # First 5 rows
                print("-" * 50)

            for interval, df in data_frames.items():
                print(f"Processing interval: {interval}")

                df['interval'] = interval
                df['symbol'] = pair

                # df = fetch_equity_data(EQUITY_INDICES)

                # Add features
                df = add_advanced_features(df, EQUITY_INDICES, rolling_window, interval)
                df = add_indicators(df)
                df = add_patterns(df)

                # Add multi-timeframe features and lagged features
                df = add_lagged_features(df, MAX_LAG)

                # Add multi-class labels
                df = add_multi_class_labels(df, interval)

                # Calculate key levels
                df = calculate_swing_points(df, rolling_window, interval)

                # Calculate the rolling VWAP
                df = calculate_rolling_vwap(df, rolling_window, interval)

                # Drop the first 77 rows with NULL values present.
                df = df.iloc[77:]
                df.reset_index(inplace=True)

                combined_df = pd.concat([combined_df, df], ignore_index=True)

                # Detect and analyze consolidation
                #key_level = df['Rolling_VWAP'].iloc[-1]  # Use the latest value as the key level
                #df = detect_consolidation(df, key_level)
                #df = analyse_consolidation(df, key_level)

                #print("\n".join(df.columns))
                #file_name = f"{pair.replace('/', '_').replace('-', '_')}_{interval}.csv"
                #df.to_csv(file_name, index=False)

            print(combined_df.iloc[:5, :5])

            combined_df = detect_levels_with_trend_alignment(combined_df, rolling_window, TIMEFRAME_MAPPING)

            # Select the first 5 rows
            first_5_rows = combined_df.head(5)

            # Output to CSV
            first_5_rows.to_csv("first_5_rows_1.csv")

            combined_df = preprocess_dataframe_for_db(combined_df, rolling_window, SCHEMA_FIELDS)

            # Select the first 5 rows
            first_5_rows = combined_df.head(5)

            # Output to CSV
            first_5_rows.to_csv("first_5_rows_2.csv")

            # Save all data into the unified_data table instead of creating individual tables
            initialise_and_save_to_db(combined_df, DB_NAME, pair, SCHEMA_FIELDS)

        print("Data processing pipeline completed successfully.")
    except Exception as e:
        print(f"Error in pipeline: {e}")
    finally:
        print("Database connection closed.")

if __name__ == "__main__":
    run_pipeline()
