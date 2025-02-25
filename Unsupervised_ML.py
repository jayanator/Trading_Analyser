# Analyses the dataset to determine clusters and their associated
# parameter values for the best potential trade positions.

import sqlite3
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DB_FILE = "trading_data.db"
TABLE_NAME = "BTCUSDT_combined"


# Load and preprocess data
def load_data(db_file, table_name):
    conn = sqlite3.connect(db_file)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def preprocess_data(df):
    # Select relevant columns
    features = df[[
        "close", "high", "low", "VWAP", "Swing_High", "Swing_Low",
        "Dist_to_SH", "Dist_to_SL", "ATR", "RSI", "MACD_Line", "Signal_Line",
        "MACD_Histogram", "ADX", "Supertrend", "Relative_Volume", "Volume_Delta",
        "Cumulative_Delta_Volume", "Cumulative_Volume", "Rolling_Volume", "Rolling_Mean_Close",
        "Rolling_Std_Close", "Fib_Ratio", "BB_upper", "BB_lower"
    ]]

    # Fill missing values and scale features
    features.fillna(features.mean(), inplace=True)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, df


# Cluster data
def cluster_data(features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    return clusters


# Analyze clusters for optimal parameters
def analyze_clusters(df, clusters):
    df["Cluster"] = clusters
    trade_rules = []

    for cluster_id in df["Cluster"].unique():
        cluster_data = df[df["Cluster"] == cluster_id]
        print(f"Analyzing Cluster {cluster_id}...")

        # Calculate optimal thresholds and parameters
        entry_conditions = {
            "RSI_mean": cluster_data["RSI"].mean(),
            "VWAP_divergence_mean": (cluster_data["VWAP"] - cluster_data["close"]).mean(),
            "Supertrend_mean": cluster_data["Supertrend"].mean(),
        }
        sl_tp_levels = {
            "ATR_based_SL": cluster_data["ATR"].mean() * 1.5,  # Example: 1.5x ATR for SL
            "ATR_based_TP": cluster_data["ATR"].mean() * 10.0,  # Example: 3x ATR for TP
        }

        trade_rules.append({
            "Cluster": cluster_id,
            "Entry_Conditions": entry_conditions,
            "SL_TP_Levels": sl_tp_levels
        })

    return trade_rules


# Save trade rules to CSV
def save_trade_rules(trade_rules, file_path):
    """
    Save trade rules to a CSV file with structured columns.

    Parameters:
    - trade_rules: List of dictionaries containing trade rules.
    - file_path: Path to save the CSV file.
    """
    # Normalize the trade rules for structured saving
    structured_rules = []
    for rule in trade_rules:
        entry_conditions = rule.get("Entry_Conditions", {})
        sl_tp_levels = rule.get("SL_TP_Levels", {})

        # Combine cluster info, entry conditions, and SL/TP levels into a single flat dictionary
        structured_rule = {"Cluster": rule["Cluster"]}
        structured_rule.update({f"Entry_Conditions_{k}": v for k, v in entry_conditions.items()})
        structured_rule.update({f"SL_TP_Levels_{k}": v for k, v in sl_tp_levels.items()})

        structured_rules.append(structured_rule)

    # Convert to DataFrame and save as CSV
    df_rules = pd.DataFrame(structured_rules)
    df_rules.to_csv(file_path, index=False)
    print(f"Trade rules saved to {file_path}")


# Main function
def main():
    db_file = "trading_data.db"
    table_name = "BTCUSDT_combined"

    print("Loading data...")
    df = load_data(db_file, table_name)

    print("Preprocessing data...")
    features, df = preprocess_data(df)

    print("Clustering data...")
    clusters = cluster_data(features, n_clusters=5)

    print("Analyzing clusters...")
    trade_rules = analyze_clusters(df, clusters)

    print("Optimal Trade Rules:")
    for rule in trade_rules:
        print(rule)

    print("Saving trade rules to file...")
    save_trade_rules(trade_rules, "trade_rules.csv")


if __name__ == "__main__":
    main()
