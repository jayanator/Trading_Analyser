import sqlite3
import pandas as pd
import numpy as np

def initialise_and_save_to_db(combined_df, db_name, pair, SCHEMA_FIELDS):
    """Initializes the database and saves the DataFrame using proper type handling and ISO 8601 datetime format."""

    table_name = f"{pair}_combined".replace("-", "_").replace("/", "_")
    print(f"Initializing and saving to table '{table_name}'...")

    sqlite3.register_adapter(np.int64, lambda x: int(x))
    sqlite3.register_adapter(np.int32, lambda x: int(x))

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS "{table_name}" (
        {", ".join(SCHEMA_FIELDS)}
    );
    """
    cursor.execute(create_table_query)

    # CRITICAL: Convert open_time and close_time to ISO 8601 format *before* any other processing
    if 'open_time' in combined_df.columns:
        combined_df['open_time'] = pd.to_datetime(combined_df['open_time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

    if 'close_time' in combined_df.columns:
        combined_df['close_time'] = pd.to_datetime(combined_df['close_time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

    # CRITICAL: Ensure correct data types (after datetime conversion):
    for col in combined_df.columns:
        if pd.api.types.is_integer_dtype(combined_df[col]):
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0).astype(int)
        elif pd.api.types.is_numeric_dtype(combined_df[col]):
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').astype(float) # Or int if needed
        elif pd.api.types.is_datetime64_any_dtype(combined_df[col]):  # Handle datetime (CORRECTED)
            combined_df[col] = pd.to_datetime(combined_df[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
        elif combined_df[col].dtype == 'bool': # Handle boolean
            combined_df[col] = combined_df[col].astype(int) # Convert bool to int (0 or 1)
        else:
            combined_df[col] = combined_df[col].astype(str) # Convert everything else to string

    # CRITICAL: Save data using parameterized query (prevents SQL injection and handles data types correctly)
    placeholders = ", ".join(["?" for _ in combined_df.columns])
    columns = ", ".join(combined_df.columns)
    query = f"""
    INSERT OR REPLACE INTO "{table_name}" ({columns})
    VALUES ({placeholders})
    """
    data = [tuple(row) for row in combined_df.values]  # Convert to list of tuples

    try:
        cursor.executemany(query, data)
        conn.commit()
        print(f"Data successfully saved to table '{table_name}'.")
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        conn.rollback()  # Rollback in case of error
        # Print debugging information
        #print(combined_df.dtypes)
        #print(combined_df.head())
        #print(combined_df.tail())
        # Check for duplicates (after converting to correct types)
        print(combined_df.duplicated(subset=['interval', 'open_time']).sum())
    finally:
        conn.close()


# Example usage
if __name__ == "__main__":
    # Mock DataFrame
    mock_data = {
        "open_time": pd.date_range(start="2023-01-01", periods=3, freq="D"),
        "open": [100, 105, 110],
        "high": [110, 115, 120],
        "low": [95, 100, 105],
        "close": [105, 110, 115],
        "volume": [1000, 1500, 2000],
        "close_time": pd.date_range(start="2023-01-01", periods=3, freq="D") + pd.Timedelta(hours=1),
        "number_of_trades": [10, 15, 20],
        "Day_of_Week": [0, 1, 2],
        "Hour_of_Day": [10, 11, 12],
        "Month": [1, 1, 1],
        "Quarter": [1, 1, 1]
    }
    combined_df = pd.DataFrame(mock_data)
    print(combined_df.index.name)
    print(combined_df.columns)

    initialise_and_save_to_db(combined_df, "trading_data.db", "BTCUSDT")