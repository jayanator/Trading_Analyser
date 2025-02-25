"""
Prediction System Development v1.0
Author: [Your Name]
Date: [Today's Date]

Purpose:
- Prepare cleaned data for machine learning.
- Train a predictive model using the Target column.
- Evaluate the model's performance.

Version Notes:
- v1.0: Initial implementation with a Random Forest classifier and evaluation metrics.
"""

import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

# Constants
DB_FILE = "trading_data.db"
TABLE_NAME = "BTCUSDT_combined"
OUTPUT_MODEL_FILE = "trained_model.pkl"


def load_data(db_file, table_name):
    """Load data from the SQLite database."""
    conn = sqlite3.connect(db_file)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def preprocess_data(df):
    """Preprocess the data for machine learning."""
    # Separate features and target
    X = df.drop(columns=["Trade_Label"]).select_dtypes(include=["float64", "int64"])
    y = df["Trade_Label"]

    # Drop non-numeric columns (e.g., timestamps or categorical data)
    X = X.select_dtypes(include=["float64", "int64"])

    print("Feature statistics:")
    print(X.describe())

    print("Checking for columns with values exceeding float64 limits:")
    large_values = (X.abs() > 1e308).sum()
    print(large_values[large_values > 0])

    print("Checking for extremely large values (above 1e12):")
    extreme_values = (X.abs() > 1e12).sum()
    print(extreme_values[extreme_values > 0])

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def train_model(X_train, y_train):
    """Train a Random Forest model with class balancing and dynamic thresholds."""
    # Convert data to float32 for reduced memory usage
    X_train = X_train.astype("float32")

    # Define Random Forest Classifier with class weights for balancing
    rf = RandomForestClassifier(
        random_state=42,
        max_samples=0.8,
        class_weight="balanced",  # Automatically adjust weights for class imbalance
        oob_score=False
    )

    # Define a hyperparameter grid for tuning
    param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [10, 20],
        "min_samples_split": [2],
        "min_samples_leaf": [1]
    }

    # Grid Search with multi-class scoring
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        scoring="roc_auc_ovr",  # Multi-class ROC-AUC
        n_jobs=2
    )

    print("Training the model with grid search...")
    grid_search.fit(X_train, y_train)

    # Best model and parameters
    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")

    # Feature importance for transparency
    if hasattr(best_model, "feature_importances_"):
        print("Feature importances:")
        for i, score in enumerate(best_model.feature_importances_):
            print(f"Feature {i}: {score:.4f}")

    return best_model


def evaluate_model(model, X_val, y_val):
    """Evaluate the model on the validation set."""
    y_pred = model.predict(X_val)

    # Predict probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val)
    else:
        y_prob = None

    print("\n--- Model Evaluation ---")
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    # Compute multi-class ROC-AUC if probabilities are available
    if y_prob is not None:
        try:
            print(f"ROC-AUC Score: {roc_auc_score(y_val, y_prob, multi_class='ovr'):.4f}")
        except ValueError as e:
            print(f"ROC-AUC calculation error: {e}")


def save_model(model, scaler, output_file):
    """Save the trained model and scaler."""
    joblib.dump({"model": model, "scaler": scaler}, output_file)
    print(f"Model saved to {output_file}")


def main():
    print("Loading data...")
    df = load_data(DB_FILE, TABLE_NAME)

    print("Preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_data(df)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    evaluate_model(model, X_val, y_val)

    print("Saving model...")
    save_model(model, scaler, OUTPUT_MODEL_FILE)


if __name__ == "__main__":
    main()
