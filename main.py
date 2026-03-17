# main.py
#
# Single entry point for the Realized Volatility Predictor pipeline.
# Run this file to execute the full pipeline end-to-end:
#
#   python main.py              ← runs everything + launches dashboard
#   python main.py --skip-dash  ← runs pipeline only, no dashboard
#
# Stages:
#   1. Download price + VIX data for all tickers
#   2. Engineer features
#   3. Train XGBoost models
#   4. Run walk-forward backtests
#   5. Launch Plotly Dash dashboard

import sys
import os
import time

# Add src/ to the Python path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_pipeline import (
    download_price_data,
    compute_realized_volatility,
    download_vix,
    save_data
)
from features import add_features
from model import load_featured_data, train_test_split_time, train_model, evaluate_model, save_model
from backtest import walk_forward_backtest, evaluate_backtest, save_results

TICKERS = ["SPY", "QQQ", "GLD"]
START   = "2015-01-01"
END     = "2024-12-31"


def separator(title: str):
    """Prints a clean section header to the terminal."""
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def stage_1_download():
    separator("STAGE 1: Downloading Price + VIX Data")
    start = time.time()

    # Download VIX once — shared across all tickers
    vix = download_vix(START, END)

    for ticker in TICKERS:
        df = download_price_data(ticker, START, END)
        df = compute_realized_volatility(df, window=21)
        df["vix"] = vix
        df.dropna(inplace=True)
        save_data(df, ticker)

    elapsed = time.time() - start
    print(f"\n✓ Stage 1 complete ({elapsed:.1f}s)")


def stage_2_features():
    separator("STAGE 2: Feature Engineering")
    start = time.time()

    import pandas as pd

    for ticker in TICKERS:
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
        print(f"  {ticker}: {df.shape[0]} rows, {df.shape[1]} columns")

    elapsed = time.time() - start
    print(f"\n✓ Stage 2 complete ({elapsed:.1f}s)")


def stage_3_train():
    separator("STAGE 3: Training XGBoost Models")
    start = time.time()

    for ticker in TICKERS:
        print(f"\n  [{ticker}]")
        df = load_featured_data(ticker)
        X_train, X_test, y_train, y_test, feature_cols = train_test_split_time(df)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test, feature_cols)
        save_model(model, ticker)

    elapsed = time.time() - start
    print(f"\n✓ Stage 3 complete ({elapsed:.1f}s)")


def stage_4_backtest():
    separator("STAGE 4: Walk-Forward Backtesting")
    start = time.time()

    for ticker in TICKERS:
        print(f"\n  [{ticker}]")
        df = load_featured_data(ticker)
        results = walk_forward_backtest(
            df,
            initial_train_size=0.6,
            step_size=21,
            retrain_every=63
        )
        evaluate_backtest(results)
        save_results(results, ticker)

    elapsed = time.time() - start
    print(f"\n✓ Stage 4 complete ({elapsed:.1f}s)")


def stage_5_dashboard():
    separator("STAGE 5: Launching Dashboard")
    print("\n  Open your browser to: http://127.0.0.1:8050")
    print("  Press Ctrl+C to stop the dashboard\n")

    # Import and run the dashboard
    # We import here so the pipeline stages run first
    from dashboard import app
    app.run(debug=False)


if __name__ == "__main__":
    total_start = time.time()

    skip_dash = "--skip-dash" in sys.argv

    print("\n" + "="*55)
    print("  REALIZED VOLATILITY PREDICTOR")
    print("  Full Pipeline Run")
    print("="*55)
    print(f"  Tickers : {', '.join(TICKERS)}")
    print(f"  Period  : {START} → {END}")
    print(f"  Dashboard: {'disabled' if skip_dash else 'will launch after pipeline'}")

    stage_1_download()
    stage_2_features()
    stage_3_train()
    stage_4_backtest()

    total_elapsed = time.time() - total_start
    print(f"\n{'='*55}")
    print(f"  ✓ Full pipeline complete in {total_elapsed:.1f}s")
    print(f"{'='*55}\n")

    if not skip_dash:
        stage_5_dashboard()