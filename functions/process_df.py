import pandas as pd
import numpy as np

def preprocess_dataframe_for_db(combined_df, rolling_window, SCHEMA_FIELDS):
    """Preprocesses DataFrame for SQLite, handling data types, missing values, outliers, and rolling metrics."""

    print("Preprocessing DataFrame for SQLite storage...")

    if 'index' in combined_df.columns:
        combined_df.drop('index', axis=1, inplace=True)
        print("Removed 'index' column.")
    else:
        print("'index' column not found.")

    if combined_df.columns[0] == "":  # Check if the first column is unnamed
        combined_df.drop(combined_df.columns[0], axis=1, inplace=True)  # Drop the unnamed first column

    # 1. Data Type Conversion and Normalization
    type_mapping = {
        "INTEGER": int,
        "REAL": float,
        "TEXT": str,
        "DATETIME": str  # Keep datetime as string for now. Convert later if needed
    }

    for column in combined_df.columns:
        if column in [field.split()[0] for field in SCHEMA_FIELDS]:  # Only convert columns in schema
            schema_type = [field.split()[1] for field in SCHEMA_FIELDS if field.split()[0] == column][0]
            try:
                combined_df[column] = combined_df[column].astype(type_mapping.get(schema_type, str), errors="ignore")  # Use mapping, default to str
            except Exception as e:
                print(f"Error converting column '{column}': {e}")
                # Handle the error, e.g., set the column to string type as a fallback
                combined_df[column] = combined_df[column].astype(str)

    # Clean column names (replace non-alphanumeric with underscores)
    combined_df.columns = combined_df.columns.str.replace(r"[^\w]", "_", regex=True)

    # 2. Handle Missing Values (before outlier normalization)
    print("Handling missing values...")
    combined_df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)  # Replace inf with NaN
    combined_df.dropna(inplace=True)  # Drop rows with NaN

    # 3. Normalize Outliers (AFTER handling missing values)
    print("Normalizing outliers...")
    numeric_cols = combined_df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        if not combined_df[col].empty:  # Check if the column is not empty before calculating quantiles
            try:
                upper_limit = combined_df[col].quantile(0.99)
                lower_limit = combined_df[col].quantile(0.01)
                combined_df[col] = combined_df[col].clip(lower=lower_limit, upper=upper_limit)
            except KeyError:
                print(f"Column '{col}' not found during outlier normalization.")
        else:
            print(f"Skipping outlier normalization for empty column: {col}")

    # 4. Reset Index (BEFORE rolling metrics)
    combined_df.reset_index(drop=True, inplace=True)  # Moved UP

    # 5. Apply Rolling Window Metrics (AFTER resetting the index)
    print("Applying rolling window metrics...")
    try:
        interval = combined_df["interval"].iloc[0]
        window = rolling_window.get(interval, 5)

        rolling_metrics = pd.DataFrame({
            "Rolling_Mean_Close": combined_df["close"].rolling(window=window).mean(),
            "Rolling_Std_Close": combined_df["close"].rolling(window=window).std(),
            "Rolling_Volume": combined_df["volume"].rolling(window=window).mean(),
            "Future_Close": combined_df["close"].shift(-window)
        }, index=combined_df.index)

        combined_df = pd.concat([combined_df, rolling_metrics], axis=1)
    except KeyError as e:
        print(f"Error: 'interval' or 'close'/'volume' columns not found for rolling metrics: {e}")
    except Exception as e:
        print(f"An error occurred during rolling metrics calculation: {e}")

    combined_df.dropna(inplace=True)  # Drop any remaining NaNs after rolling calculations.

    print("\nColumn types after preprocessing:")
    print(combined_df.dtypes)

    print("Pre-save processing complete.")

    return combined_df