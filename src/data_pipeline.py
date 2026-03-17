# src/data_pipeline.py

import yfinance as yf    # downloads stock price data from Yahoo Finance
import pandas as pd      # for working with tables of data (DataFrames)
import numpy as np       # for math operations like log()
import os                # for file path operations


def download_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Downloads daily OHLCV price data for a given ticker.

    ticker : stock symbol, e.g. "SPY"
    start  : start date string, e.g. "2015-01-01"
    end    : end date string,   e.g. "2024-12-31"
    """
    print(f"Downloading {ticker} from {start} to {end}...")

    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.dropna(inplace=True)

    print(f"Downloaded {len(df)} rows.")
    return df


def compute_realized_volatility(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Adds log returns and realized volatility to the DataFrame.

    Realized vol = rolling std of log returns, annualized.
    This is our prediction TARGET — what the model will learn to forecast.

    window : rolling window in trading days (21 ≈ 1 month)
    """
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    rolling_std = df["log_return"].rolling(window=window).std()

    # Annualize by multiplying by sqrt(252)
    # 252 = number of trading days in a year — industry standard
    df["realized_vol"] = rolling_std * (252 ** 0.5)

    df.dropna(inplace=True)
    return df


def download_vix(start: str, end: str) -> pd.Series:
    """
    Downloads the VIX index — the market's 'fear gauge'.

    VIX measures implied volatility (what options markets EXPECT vol to be),
    while our target is realized vol (what vol ACTUALLY was).
    The spread between them is a powerful predictive signal used by
    real volatility traders.

    Returns a pandas Series of daily VIX closing values.
    """
    print(f"Downloading VIX from {start} to {end}...")

    # ^VIX is Yahoo Finance's ticker for the CBOE Volatility Index
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=True)

    # Flatten the multi-level column that yfinance creates
    vix_close = vix["Close"].squeeze()
    vix_close.name = "vix"

    print(f"Downloaded {len(vix_close)} VIX rows.")
    return vix_close


def save_data(df: pd.DataFrame, ticker: str, data_dir: str = "data") -> str:
    """
    Saves the processed DataFrame to a CSV in the data/ folder.
    """
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{ticker}_processed.csv")
    df.to_csv(path)
    print(f"Saved to {path}")
    return path


if __name__ == "__main__":
    START = "2015-01-01"
    END   = "2024-12-31"

    # Download VIX once — shared across all tickers
    vix = download_vix(START, END)

    # Process each ticker the same way
    for TICKER in ["SPY", "QQQ", "GLD"]:
        df = download_price_data(TICKER, START, END)
        df = compute_realized_volatility(df, window=21)
        df["vix"] = vix
        df.dropna(inplace=True)
        save_data(df, TICKER)

        print(f"\n{TICKER} sample:")
        print(df[["Close", "log_return", "realized_vol", "vix"]].tail(3))
        print()