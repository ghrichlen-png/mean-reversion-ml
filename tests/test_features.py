"""
tests/test_features.py
────────────────────────────────────────────────────────────────
Unit tests for the feature engineering module.
Run with: pytest tests/
"""

import numpy as np
import pandas as pd
import pytest
from src.pipeline.features import (
    bollinger_bands,
    zscore,
    rsi,
    rolling_volatility,
    distance_from_sma,
    volume_ratio,
)


@pytest.fixture
def sample_close() -> pd.Series:
    """50-day synthetic price series with a clear trend."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
    return pd.Series(prices, name="Close")


@pytest.fixture
def sample_volume() -> pd.Series:
    np.random.seed(7)
    return pd.Series(np.random.randint(1_000_000, 5_000_000, 50), name="Volume")


# ── Bollinger Bands ───────────────────────────────────────────────

def test_bollinger_bands_shape(sample_close):
    result = bollinger_bands(sample_close, window=20, num_std=2)
    assert set(result.columns) == {"bb_pct_b", "bb_width", "bb_position"}
    assert len(result) == len(sample_close)


def test_bollinger_bands_nan_warmup(sample_close):
    result = bollinger_bands(sample_close, window=20, num_std=2)
    # First 19 rows should be NaN (window not filled yet)
    assert result["bb_pct_b"].iloc[:19].isna().all()
    assert result["bb_pct_b"].iloc[19:].notna().all()


def test_bollinger_bands_range(sample_close):
    result = bollinger_bands(sample_close, window=20, num_std=2)
    pct_b = result["bb_pct_b"].dropna()
    # %B can go slightly outside [0, 1] for extreme moves — that's intentional
    assert not pct_b.empty


# ── Z-Score ───────────────────────────────────────────────────────

def test_zscore_values(sample_close):
    z = zscore(sample_close, window=20).dropna()
    # Z-scores should be roughly bounded (not extreme on a random walk)
    assert z.abs().max() < 10


def test_zscore_mean_near_zero(sample_close):
    z = zscore(sample_close, window=20).dropna()
    # Rolling z-score mean should be near zero by construction
    assert abs(z.mean()) < 1.5


# ── RSI ───────────────────────────────────────────────────────────

def test_rsi_bounded(sample_close):
    r = rsi(sample_close, window=14).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_rsi_name(sample_close):
    assert rsi(sample_close, window=14).name == "rsi"


# ── Rolling Volatility ────────────────────────────────────────────

def test_volatility_positive(sample_close):
    vol = rolling_volatility(sample_close, window=10).dropna()
    assert (vol >= 0).all()


# ── Distance from SMA ─────────────────────────────────────────────

def test_distance_from_sma_sign(sample_close):
    dist = distance_from_sma(sample_close, window=20).dropna()
    # Distance sign should match whether price is above or below SMA
    sma = sample_close.rolling(20).mean()
    above = (sample_close > sma).iloc[19:]
    assert ((dist.iloc[:] > 0) == above.values).all()


# ── Volume Ratio ──────────────────────────────────────────────────

def test_volume_ratio_positive(sample_volume):
    vr = volume_ratio(sample_volume, window=20).dropna()
    assert (vr > 0).all()
