# src/model.py

import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor          # our ML model
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_featured_data(ticker: str = "SPY") -> pd.DataFrame:
    """
    Loads the processed CSV and runs feature engineering on it.
    This is a convenience function so model.py is self-contained.
    """
    # Import our own module from the same src/ package
    from features import add_features

    path = os.path.join("data", f"{ticker}_processed.csv")

    df = pd.read_csv(
        path,
        skiprows=3,
        header=None,
        names=["Date", "Close", "High", "Low", "Open", "Volume",
               "log_return", "realized_vol", "vix"],
        index_col="Date",
        parse_dates=True
    )

    df = add_features(df)
    return df


def train_test_split_time(df: pd.DataFrame, test_size: float = 0.2):
    """
    Splits data into train and test sets CHRONOLOGICALLY.

    IMPORTANT: We never use random splitting for time series data.
    If we trained on 2020 data and tested on 2018 data, the model
    would be cheating — it would have 'seen the future' during training.
    Instead we train on the first 80% of dates, test on the last 20%.

    test_size: fraction of data to hold out for testing (0.2 = 20%)
    """
    # These are the columns the model learns FROM (inputs)
    feature_cols = [
        "vol_lag_1", "vol_lag_5", "vol_lag_10", "vol_lag_21",
        "vol_of_vol_5", "vol_of_vol_21",
        "mean_return_5", "mean_return_21",
        "vol_5d", "vol_63d", "vol_ratio",
        "abs_return",
        "vix_normalized", "vol_risk_premium", "vix_lag_1", "vix_rolling_5"
    ]

    # This is the column the model learns TO predict (output)
    target_col = "target_vol"

    # Calculate where to split — e.g. row 1929 out of 2411
    split_idx = int(len(df) * (1 - test_size))

    # Everything before split_idx = training data
    X_train = df[feature_cols].iloc[:split_idx]
    y_train = df[target_col].iloc[:split_idx]

    # Everything after split_idx = test data (unseen during training)
    X_test  = df[feature_cols].iloc[split_idx:]
    y_test  = df[target_col].iloc[split_idx:]

    print(f"Training on {len(X_train)} rows "
          f"({df.index[0].date()} to {df.index[split_idx].date()})")
    print(f"Testing  on {len(X_test)} rows  "
          f"({df.index[split_idx].date()} to {df.index[-1].date()})")

    return X_train, X_test, y_train, y_test, feature_cols


def train_model(X_train, y_train) -> XGBRegressor:
    """
    Trains an XGBoost regression model.

    XGBoost builds an ensemble of decision trees sequentially —
    each tree learns to correct the errors of the previous one.
    It's the most widely used model in quant finance and ML competitions.

    Key hyperparameters explained:
        n_estimators   : number of trees to build (100 is a safe start)
        max_depth      : how deep each tree can grow (3 = shallow = less overfit)
        learning_rate  : how much each tree contributes (smaller = more careful)
        subsample      : fraction of rows used per tree (0.8 = some randomness)
        random_state   : seed for reproducibility — same result every run
    """
    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbosity=0           # suppress XGBoost's own print output
    )

    # .fit() is where the actual learning happens
    # The model sees (features, target) pairs and learns the relationship
    model.fit(X_train, y_train)
    print("Model trained.")
    return model


def evaluate_model(model, X_test, y_test, feature_cols):
    """
    Evaluates model performance on the held-out test set.

    Metrics explained:
        MAE  (Mean Absolute Error)       : average prediction error
        RMSE (Root Mean Squared Error)   : penalizes large errors more
        Dir. Accuracy                    : did we predict the direction
                                           of vol change correctly?
    """
    # Generate predictions on test data
    y_pred = model.predict(X_test)

    # MAE: average of |predicted - actual|
    mae = mean_absolute_error(y_test, y_pred)

    # RMSE: sqrt of average squared errors
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    # Directional accuracy: did vol go up when we said it would?
    # Compare direction of change vs the previous day's vol
    actual_direction    = np.sign(y_test.values[1:] - y_test.values[:-1])
    predicted_direction = np.sign(y_pred[1:]        - y_pred[:-1])
    dir_accuracy = (actual_direction == predicted_direction).mean()

    print(f"\n--- Model Evaluation ---")
    print(f"MAE:                {mae:.4f}  (avg prediction error in vol units)")
    print(f"RMSE:               {rmse:.4f}")
    print(f"Directional Acc:    {dir_accuracy:.2%}  (random = 50%)")

    # Feature importance: which inputs did the model rely on most?
    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    print(f"\n--- Feature Importance ---")
    print(importance.round(4))

    return y_pred, mae, rmse, dir_accuracy


def save_model(model, ticker: str = "SPY"):
    """
    Saves the trained model to the models/ folder.
    This lets us load it later in the dashboard without retraining.
    """
    os.makedirs("models", exist_ok=True)
    path = os.path.join("models", f"{ticker}_xgb_model.json")
    model.save_model(path)
    print(f"\nModel saved to {path}")


if __name__ == "__main__":
    for TICKER in ["SPY", "QQQ", "GLD"]:
        print(f"\n{'='*50}")
        print(f"  Training model for {TICKER}")
        print(f"{'='*50}")

        df = load_featured_data(TICKER)
        X_train, X_test, y_train, y_test, feature_cols = train_test_split_time(df)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test, feature_cols)
        save_model(model, TICKER)