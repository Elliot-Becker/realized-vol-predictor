# tests/test_pipeline.py
#
# Unit tests for the Realized Volatility Predictor pipeline.
# Run with: pytest tests/
#
# Tests verify that each module:
#   - Returns the correct data types
#   - Produces the expected columns
#   - Handles edge cases without crashing

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add src/ to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_pipeline import compute_realized_volatility
from features import add_features


# ----------------------------------------------------------------
# FIXTURES
# A fixture is reusable test data that pytest injects automatically
# into any test function that lists it as a parameter.
# ----------------------------------------------------------------

@pytest.fixture
def sample_price_df():
    """
    Creates a small synthetic price DataFrame that mimics
    what download_price_data() returns from yfinance.
    We use synthetic data so tests don't require internet access.
    """
    np.random.seed(42)
    n = 150  # enough rows for all rolling windows to warm up

    # Simulate a realistic price series using cumulative random walk
    returns = np.random.normal(0.0005, 0.01, n)
    prices  = 400 * np.exp(np.cumsum(returns))

    dates = pd.date_range(start="2022-01-01", periods=n, freq="B")  # B = business days

    df = pd.DataFrame({
        "Close":  prices,
        "High":   prices * 1.005,
        "Low":    prices * 0.995,
        "Open":   prices * 0.999,
        "Volume": np.random.randint(50_000_000, 150_000_000, n)
    }, index=dates)

    df.index.name = "Date"
    return df


@pytest.fixture
def sample_vol_df(sample_price_df):
    """
    Runs compute_realized_volatility on the sample price data.
    Used as input for feature engineering tests.
    """
    return compute_realized_volatility(sample_price_df.copy(), window=21)


@pytest.fixture
def sample_featured_df(sample_vol_df):
    """
    Adds a synthetic VIX column and runs add_features().
    Used as input for model and backtest tests.
    """
    df = sample_vol_df.copy()
    # Simulate VIX as a noisy version of annualized vol * 100
    df["vix"] = df["realized_vol"] * 100 * (1 + np.random.normal(0, 0.1, len(df)))
    return add_features(df)


# ----------------------------------------------------------------
# DATA PIPELINE TESTS
# ----------------------------------------------------------------

class TestComputeRealizedVolatility:

    def test_returns_dataframe(self, sample_price_df):
        """Output must be a DataFrame."""
        result = compute_realized_volatility(sample_price_df.copy())
        assert isinstance(result, pd.DataFrame)

    def test_adds_log_return_column(self, sample_price_df):
        """Must add a 'log_return' column."""
        result = compute_realized_volatility(sample_price_df.copy())
        assert "log_return" in result.columns

    def test_adds_realized_vol_column(self, sample_price_df):
        """Must add a 'realized_vol' column."""
        result = compute_realized_volatility(sample_price_df.copy())
        assert "realized_vol" in result.columns

    def test_no_nan_values(self, sample_price_df):
        """Output must have no NaN values after dropna."""
        result = compute_realized_volatility(sample_price_df.copy())
        assert result.isnull().sum().sum() == 0

    def test_realized_vol_is_positive(self, sample_price_df):
        """Volatility must always be positive."""
        result = compute_realized_volatility(sample_price_df.copy())
        assert (result["realized_vol"] > 0).all()

    def test_realized_vol_is_annualized(self, sample_price_df):
        """
        Annualized vol for a typical equity should be between 5% and 100%.
        This catches mistakes like forgetting to multiply by sqrt(252).
        """
        result = compute_realized_volatility(sample_price_df.copy())
        mean_vol = result["realized_vol"].mean()
        assert 0.05 < mean_vol < 1.0, f"Vol looks wrong: {mean_vol:.4f}"

    def test_row_count_reduced_by_window(self, sample_price_df):
        """
        Rolling window + shift should reduce row count.
        Output must have fewer rows than input.
        """
        result = compute_realized_volatility(sample_price_df.copy(), window=21)
        assert len(result) < len(sample_price_df)


# ----------------------------------------------------------------
# FEATURE ENGINEERING TESTS
# ----------------------------------------------------------------

class TestAddFeatures:

    def test_returns_dataframe(self, sample_vol_df):
        """Output must be a DataFrame."""
        df = sample_vol_df.copy()
        df["vix"] = df["realized_vol"] * 100
        result = add_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_lag_features_exist(self, sample_featured_df):
        """All lagged vol features must be present."""
        for lag in [1, 5, 10, 21]:
            assert f"vol_lag_{lag}" in sample_featured_df.columns, \
                f"Missing column: vol_lag_{lag}"

    def test_rolling_vol_features_exist(self, sample_featured_df):
        """Rolling vol features must be present."""
        for col in ["vol_5d", "vol_63d", "vol_ratio"]:
            assert col in sample_featured_df.columns, \
                f"Missing column: {col}"

    def test_vix_features_exist(self, sample_featured_df):
        """VIX-derived features must be present when VIX column exists."""
        for col in ["vix_normalized", "vol_risk_premium",
                    "vix_lag_1", "vix_rolling_5"]:
            assert col in sample_featured_df.columns, \
                f"Missing column: {col}"

    def test_target_vol_exists(self, sample_featured_df):
        """Target column must be present."""
        assert "target_vol" in sample_featured_df.columns

    def test_no_nan_values(self, sample_featured_df):
        """No NaN values after feature engineering."""
        assert sample_featured_df.isnull().sum().sum() == 0

    def test_vix_normalized_scale(self, sample_featured_df):
        """
        vix_normalized should be in roughly the same range as realized_vol
        (both annualized, both between 0.05 and 1.0 for normal markets).
        """
        mean_vix_norm = sample_featured_df["vix_normalized"].mean()
        assert 0.05 < mean_vix_norm < 1.0, \
            f"vix_normalized looks wrong: {mean_vix_norm:.4f}"

    def test_vol_ratio_is_positive(self, sample_featured_df):
        """Vol ratio (short/long) must always be positive."""
        assert (sample_featured_df["vol_ratio"] > 0).all()

    def test_feature_count(self, sample_featured_df):
        """
        Should have at least 20 columns after full feature engineering.
        This catches cases where features were accidentally dropped.
        """
        assert sample_featured_df.shape[1] >= 20, \
            f"Too few columns: {sample_featured_df.shape[1]}"


# ----------------------------------------------------------------
# INTEGRATION TEST
# ----------------------------------------------------------------

class TestPipelineIntegration:

    def test_full_pipeline_runs(self, sample_price_df):
        """
        End-to-end test: price data → realized vol → features.
        Verifies all three stages connect correctly.
        """
        # Stage 1: compute vol
        vol_df = compute_realized_volatility(sample_price_df.copy(), window=21)
        assert "realized_vol" in vol_df.columns

        # Stage 2: add VIX and engineer features
        vol_df["vix"] = vol_df["realized_vol"] * 100
        featured_df = add_features(vol_df)
        assert "target_vol" in featured_df.columns
        assert featured_df.isnull().sum().sum() == 0

        # Stage 3: verify we have enough rows to train a model
        assert len(featured_df) > 50, \
            f"Not enough rows for training: {len(featured_df)}"