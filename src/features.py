# src/features.py

import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the processed DataFrame from data_pipeline.py and adds
    all the features our XGBoost model will use to make predictions.

    In ML, these input columns are called "features" — they're the
    signals the model learns patterns from.
    """

    # ----------------------------------------------------------------
    # 1. LAGGED VOLATILITY FEATURES
    # "What was realized vol 1, 5, 10, 21 days ago?"
    # ----------------------------------------------------------------
    for lag in [1, 5, 10, 21]:
        df[f"vol_lag_{lag}"] = df["realized_vol"].shift(lag)

    # ----------------------------------------------------------------
    # 2. ROLLING VOLATILITY OF VOLATILITY
    # "How much has vol itself been moving recently?"
    # ----------------------------------------------------------------
    df["vol_of_vol_5"]  = df["realized_vol"].rolling(5).std()
    df["vol_of_vol_21"] = df["realized_vol"].rolling(21).std()

    # ----------------------------------------------------------------
    # 3. ROLLING MEAN RETURN
    # "What's the average daily return over the past 5 and 21 days?"
    # ----------------------------------------------------------------
    df["mean_return_5"]  = df["log_return"].rolling(5).mean()
    df["mean_return_21"] = df["log_return"].rolling(21).mean()

    # ----------------------------------------------------------------
    # 4. ROLLING VOLATILITY AT DIFFERENT WINDOWS
    # ----------------------------------------------------------------
    df["vol_5d"]  = df["log_return"].rolling(5).std()  * (252 ** 0.5)
    df["vol_63d"] = df["log_return"].rolling(63).std() * (252 ** 0.5)

    # ----------------------------------------------------------------
    # 5. VOL RATIO (short-term vs long-term)
    # ----------------------------------------------------------------
    df["vol_ratio"] = df["vol_5d"] / df["vol_63d"]

    # ----------------------------------------------------------------
    # 6. ABSOLUTE RETURN
    # ----------------------------------------------------------------
    df["abs_return"] = df["log_return"].abs()

    # ----------------------------------------------------------------
    # 7. VIX FEATURES
    # "What is implied vol doing vs realized vol?"
    #
    # VIX is forward-looking (implied), our realized_vol is backward-looking.
    # The gap between them — the "vol risk premium" — is one of the most
    # researched signals in quantitative finance.
    # ----------------------------------------------------------------
    if "vix" in df.columns:
        # Normalize VIX to same scale as realized vol (VIX is in % points)
        df["vix_normalized"] = df["vix"] / 100

        # Vol risk premium: implied vol minus realized vol
        # Positive = market expects MORE vol than recently realized (fear)
        # Negative = market expects LESS vol (complacency)
        df["vol_risk_premium"] = df["vix_normalized"] - df["realized_vol"]

        # Lagged VIX — what was the fear gauge yesterday?
        df["vix_lag_1"] = df["vix_normalized"].shift(1)

        # Rolling VIX mean — is VIX trending up or down?
        df["vix_rolling_5"] = df["vix_normalized"].rolling(5).mean()

    # ----------------------------------------------------------------
    # 8. TARGET: FUTURE REALIZED VOLATILITY (5-day horizon)
    # ----------------------------------------------------------------
    df["target_vol"] = df["realized_vol"].shift(-5)

    df.dropna(inplace=True)

    return df


if __name__ == "__main__":
    import os

    path = os.path.join("data", "SPY_processed.csv")

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

    print(f"Shape after feature engineering: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample (last 3 rows):")
    print(df.tail(3))