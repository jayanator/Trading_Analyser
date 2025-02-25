"""
Model Backtesting with Trade Rules v2.0
Author: [Your Name]
Date: [Today's Date]

Purpose:
- Load a trained model and evaluate it on a test dataset.
- Incorporate trade rules from clustering, transaction costs, stop-loss logic, and performance metrics.
"""

import sqlite3
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Constants
DB_FILE = "trading_data.db"
TABLE_NAME = "BTCUSDT_combined"
TRADE_RULES_FILE = "trade_rules.csv"  # Trade rules generated from clustering
SAVED_MODEL_FILE = "trained_model.pkl"
INITIAL_PORTFOLIO = 1000  # Starting portfolio value in dollars
RISK_PER_TRADE_VALS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.08, 0.09, 0.10, 0.15] # 3% risk per trade
LEVERAGE_VALS = [1, 2, 3, 4, 5, 10, 15, 20, 50, 100] # Leverage factor (e.g., 3x leverage)
TRANSACTION_COST = 0.001  # Transaction cost (0.1% per trade)


def load_data(db_file, table_name):
    """Load data from the SQLite database."""
    conn = sqlite3.connect(db_file)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def load_trade_rules(file_path):
    """Load trade rules from a CSV file."""
    return pd.read_csv(file_path)


def preprocess_data(df, scaler):
    """Preprocess the data using the saved scaler."""
    X = df.select_dtypes(include=["float64", "int64"]).drop(columns=["Trade_Label"], errors="ignore")
    X_scaled = scaler.transform(X)
    y = df.get("Trade_Label", None)  # If "Trade_Label" exists, include it; else set as None
    return X_scaled, y, df


def apply_trade_rules(df, trade_rules):
    """Apply trade rules to the dataset to calculate SL/TP levels."""
    # Ensure 'Cluster' column is present in the DataFrame
    if 'Cluster' not in df.columns:
        print("Assigning clusters to the dataset based on trade rules...")

        # Placeholder cluster assignment based on trade rules and scaled features
        # Here, we replicate cluster labels to match the length of the dataset
        clusters = trade_rules['Cluster'].unique()
        cluster_assignments = np.resize(clusters, len(df))  # Repeats clusters cyclically to match dataset size

        df['Cluster'] = cluster_assignments

    # Apply trade rules for each cluster
    for _, rule in trade_rules.iterrows():
        cluster = rule['Cluster']
        try:
            # Extract rule parameters
            rsi_mean = rule['Entry_Conditions_RSI_mean']
            vwap_divergence_mean = rule['Entry_Conditions_VWAP_divergence_mean']
            atr_sl = rule['SL_TP_Levels_ATR_based_SL']
            atr_tp = rule['SL_TP_Levels_ATR_based_TP']

            # Filter data for the current cluster
            cluster_data = df[df['Cluster'] == cluster]

            # Apply trade rules
            df.loc[cluster_data.index, 'Target_Long'] = cluster_data['close'] + atr_tp
            df.loc[cluster_data.index, 'Target_Short'] = cluster_data['close'] - atr_tp
            df.loc[cluster_data.index, 'Stop_Loss_Long'] = cluster_data['close'] - atr_sl
            df.loc[cluster_data.index, 'Stop_Loss_Short'] = cluster_data['close'] + atr_sl
        except KeyError as e:
            print(f"Error applying rules for cluster {cluster}: {e}")
            continue

    return df


def backtest_model(test_data, y_pred):
    """Perform backtesting with transaction costs, stop-loss logic, and metrics across varying parameters."""
    test_data['Predicted_Label'] = y_pred
    optimization_results = []  # To store results for each combination of RISK_PER_TRADE and LEVERAGE
    all_cluster_metrics = []  # To store cluster metrics separately

    for RISK_PER_TRADE in RISK_PER_TRADE_VALS:
        for LEVERAGE in LEVERAGE_VALS:
            print(f"\nTesting with RISK_PER_TRADE={RISK_PER_TRADE}, LEVERAGE={LEVERAGE}")

            # Calculate position size based on risk
            test_data['Position_Size'] = (INITIAL_PORTFOLIO * RISK_PER_TRADE * LEVERAGE) / (test_data['close'] * 0.05)

            # Calculate profit for each trade
            test_data['Profit'] = np.where(
                test_data['Predicted_Label'] == 1,  # Long trade
                test_data['Position_Size'] * (test_data['Target_Long'] - test_data['close']) / test_data['close'],
                np.where(
                    test_data['Predicted_Label'] == -1,  # Short trade
                    test_data['Position_Size'] * (test_data['close'] - test_data['Target_Short']) / test_data['close'],
                    0  # No action
                )
            )

            # Deduct transaction costs
            test_data['Profit'] -= TRANSACTION_COST * test_data['Position_Size']

            # Update portfolio value after each trade
            test_data['Portfolio_Value'] = INITIAL_PORTFOLIO + test_data['Profit'].cumsum()

            # Overall Performance Metrics
            final_portfolio = test_data['Portfolio_Value'].iloc[-1]
            total_profit = final_portfolio - INITIAL_PORTFOLIO
            max_drawdown = (test_data['Portfolio_Value'].cummax() - test_data['Portfolio_Value']).max()
            win_trades = (test_data['Profit'] > 0).sum()
            loss_trades = (test_data['Profit'] < 0).sum()
            win_loss_ratio = win_trades / loss_trades if loss_trades > 0 else float('inf')
            sharpe_ratio = (
                test_data['Profit'].mean() / test_data['Profit'].std() * (252**0.5) if test_data['Profit'].std() > 0 else 0
            )

            # Store results
            optimization_results.append({
                "RISK_PER_TRADE": RISK_PER_TRADE,
                "LEVERAGE": LEVERAGE,
                "Final Portfolio Value": final_portfolio,
                "Total Profit": total_profit,
                "Maximum Drawdown": max_drawdown,
                "Winning Trades": win_trades,
                "Losing Trades": loss_trades,
                "Win/Loss Ratio": win_loss_ratio,
                "Sharpe Ratio": sharpe_ratio,
            })

            # Cluster-wise Performance Metrics
            if 'Cluster' in test_data.columns:
                print("\nCluster-wise Performance Metrics:")
                for cluster_id in test_data['Cluster'].unique():
                    cluster_data = test_data[test_data['Cluster'] == cluster_id]
                    cluster_profit = cluster_data['Profit'].sum()
                    cluster_win_trades = (cluster_data['Profit'] > 0).sum()
                    cluster_loss_trades = (cluster_data['Profit'] < 0).sum()
                    cluster_win_loss_ratio = (
                        cluster_win_trades / cluster_loss_trades if cluster_loss_trades > 0 else float('inf')
                    )
                    cluster_sharpe_ratio = (
                        cluster_data['Profit'].mean() / cluster_data['Profit'].std() * (252**0.5)
                        if cluster_data['Profit'].std() > 0 else 0
                    )

                    all_cluster_metrics.append({
                        "RISK_PER_TRADE": RISK_PER_TRADE,
                        "LEVERAGE": LEVERAGE,
                        "Cluster": cluster_id,
                        "Total Profit": cluster_profit,
                        "Winning Trades": cluster_win_trades,
                        "Losing Trades": cluster_loss_trades,
                        "Win/Loss Ratio": cluster_win_loss_ratio,
                        "Sharpe Ratio": cluster_sharpe_ratio,
                    })

                    print(f"Cluster {cluster_id}:")
                    print(f"  Total Profit: ${cluster_profit:.2f}")
                    print(f"  Winning Trades: {cluster_win_trades}")
                    print(f"  Losing Trades: {cluster_loss_trades}")
                    print(f"  Win/Loss Ratio: {cluster_win_loss_ratio:.2f}")
                    print(f"  Sharpe Ratio: {cluster_sharpe_ratio:.2f}")



    # Save optimization results
    optimization_results_df = pd.DataFrame(optimization_results)
    optimization_results_df.to_csv("optimization_results.csv", index=False)
    print("Optimization results saved to optimization_results.csv")

    # Save cluster metrics
    cluster_metrics_df = pd.DataFrame(all_cluster_metrics)
    cluster_metrics_df.to_csv("cluster_metrics.csv", index=False)
    print("Cluster metrics saved to cluster_metrics.csv")

    return test_data

def main():
    print("Loading saved model...")
    saved_objects = joblib.load(SAVED_MODEL_FILE)
    model = saved_objects["model"]
    scaler = saved_objects["scaler"]

    print("Loading test data...")
    df = load_data(DB_FILE, TABLE_NAME)

    print("Loading trade rules...")
    trade_rules = load_trade_rules(TRADE_RULES_FILE)

    print("Preprocessing test data...")
    X_test, _, test_data = preprocess_data(df, scaler)

    print("Applying trade rules...")
    test_data = apply_trade_rules(test_data, trade_rules)

    print("Making predictions...")
    y_pred = model.predict(X_test)

    print("Backtesting...")
    test_data = backtest_model(test_data, y_pred)

    # Save backtesting results to CSV
    test_data.to_csv("backtest_results.csv", index=False)
    print("Backtest results saved to backtest_results.csv")


if __name__ == "__main__":
    main()
