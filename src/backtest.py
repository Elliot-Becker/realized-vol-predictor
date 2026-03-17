# src/backtest.py

import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_featured_data(ticker: str = "SPY") -> pd.DataFrame:
    """
    Same loader we used in model.py — loads and engineers features.
    """
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


def walk_forward_backtest(
    df: pd.DataFrame,
    initial_train_size: float = 0.6,
    step_size: int = 21,
    retrain_every: int = 63
) -> pd.DataFrame:
    """
    Walk-forward backtesting — the industry standard for evaluating
    time series models in quant finance.

    Instead of one static train/test split, we simulate how the model
    would have performed if deployed in real time:

    1. Train on the first 60% of data
    2. Predict the next 21 days
    3. Roll forward 21 days, retrain on all data seen so far
    4. Predict the next 21 days again
    5. Repeat until we reach the end of the data

    This prevents "lookahead bias" — the model never sees the future
    during training, just like in real deployment.

    Args:
        initial_train_size : fraction of data for the first training window
        step_size          : how many days to predict before rolling forward
        retrain_every      : retrain the model every N days (63 ≈ 1 quarter)
    """

    feature_cols = [
        "vol_lag_1", "vol_lag_5", "vol_lag_10", "vol_lag_21",
        "vol_of_vol_5", "vol_of_vol_21",
        "mean_return_5", "mean_return_21",
        "vol_5d", "vol_63d", "vol_ratio",
        "abs_return",
        "vix_normalized", "vol_risk_premium", "vix_lag_1", "vix_rolling_5"
    ]
    target_col = "target_vol"

    # Calculate where the first training window ends
    train_end = int(len(df) * initial_train_size)

    # These lists will collect results as we walk forward
    all_dates      = []
    all_actuals    = []
    all_predictions = []

    # current_pos tracks where we are in the dataset as we walk forward
    current_pos = train_end
    model = None
    days_since_retrain = 0

    print(f"Starting walk-forward backtest...")
    print(f"Initial training window: {train_end} rows "
          f"({df.index[0].date()} to {df.index[train_end].date()})")
    print(f"Walking forward from {df.index[train_end].date()} "
          f"to {df.index[-1].date()}\n")

    while current_pos < len(df) - step_size:

        # Retrain the model periodically (every 63 days)
        # On the first iteration, days_since_retrain == retrain_every
        # so we always train on the first pass
        if days_since_retrain >= retrain_every or model is None:

            # Training data = everything up to current position
            X_train = df[feature_cols].iloc[:current_pos]
            y_train = df[target_col].iloc[:current_pos]

            # Fresh model each time — retraining from scratch
            model = XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                verbosity=0
            )
            model.fit(X_train, y_train)
            days_since_retrain = 0

            print(f"  Retrained at {df.index[current_pos].date()} "
                  f"using {current_pos} rows of history")

        # Predict the next step_size days using the current model
        X_pred = df[feature_cols].iloc[current_pos: current_pos + step_size]
        y_true = df[target_col].iloc[current_pos: current_pos + step_size]

        preds = model.predict(X_pred)

        # Store results
        all_dates.extend(df.index[current_pos: current_pos + step_size])
        all_actuals.extend(y_true.values)
        all_predictions.extend(preds)

        # Roll forward
        current_pos      += step_size
        days_since_retrain += step_size

    # Package results into a clean DataFrame
    results = pd.DataFrame({
        "actual":     all_actuals,
        "predicted":  all_predictions
    }, index=all_dates)

    return results


def evaluate_backtest(results: pd.DataFrame) -> dict:
    """
    Computes evaluation metrics across the full backtest period.
    """
    actual    = results["actual"].values
    predicted = results["predicted"].values

    mae  = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5

    # Directional accuracy: did we predict whether vol went up or down?
    actual_dir    = np.sign(actual[1:]    - actual[:-1])
    predicted_dir = np.sign(predicted[1:] - predicted[:-1])
    dir_acc = (actual_dir == predicted_dir).mean()

    # Correlation: do our predictions move with reality?
    correlation = np.corrcoef(actual, predicted)[0, 1]

    print(f"\n--- Walk-Forward Backtest Results ---")
    print(f"Period:           {results.index[0].date()} "
          f"to {results.index[-1].date()}")
    print(f"Total predictions: {len(results)}")
    print(f"MAE:               {mae:.4f}")
    print(f"RMSE:              {rmse:.4f}")
    print(f"Directional Acc:   {dir_acc:.2%}  (random = 50%)")
    print(f"Correlation:       {correlation:.4f}  (1.0 = perfect)")

    return {
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": dir_acc,
        "correlation": correlation
    }


def save_results(results: pd.DataFrame, ticker: str = "SPY"):
    """
    Saves backtest predictions to data/ for use in the dashboard.
    """
    path = os.path.join("data", f"{ticker}_backtest_results.csv")
    results.to_csv(path)
    print(f"\nBacktest results saved to {path}")


if __name__ == "__main__":
    for TICKER in ["SPY", "QQQ", "GLD"]:
        print(f"\n{'='*50}")
        print(f"  Running backtest for {TICKER}")
        print(f"{'='*50}")

        df = load_featured_data(TICKER)

        results = walk_forward_backtest(
            df,
            initial_train_size=0.6,
            step_size=21,
            retrain_every=63
        )

        metrics = evaluate_backtest(results)
        save_results(results, TICKER)