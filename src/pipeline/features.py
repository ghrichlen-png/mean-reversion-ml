"""
Feature engineering: compute technical indicators from raw OHLCV data.

Indicators:
    Bollinger Bands (%B, width, position)
    Z-score (rolling)
    RSI (14-day)
    Rolling volatility (annualised)
    Distance from SMA
    Volume ratio
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def bollinger_bands(close: pd.Series, window: int, num_std: float) -> pd.DataFrame:
    """
    %B = (Price - Lower) / (Upper - Lower)
    0 = at lower band, 1 = at upper band. Values outside [0,1] indicate extremes.
    """
    sma   = close.rolling(window).mean()
    std   = close.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std

    return pd.DataFrame({
        "bb_pct_b":    (close - lower) / (upper - lower),
        "bb_width":    (upper - lower) / sma,
        "bb_position": close - sma,
    })


def zscore(close: pd.Series, window: int) -> pd.Series:
    """Rolling z-score relative to a lookback window."""
    mu  = close.rolling(window).mean()
    sig = close.rolling(window).std()
    return ((close - mu) / sig).rename("zscore")


def rsi(close: pd.Series, window: int) -> pd.Series:
    """Wilder RSI. Values > 70 overbought, < 30 oversold."""
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("rsi")


def rolling_volatility(close: pd.Series, window: int) -> pd.Series:
    """Annualised volatility from log returns."""
    log_ret = np.log(close / close.shift(1))
    return (log_ret.rolling(window).std() * np.sqrt(252)).rename("volatility")


def distance_from_sma(close: pd.Series, window: int) -> pd.Series:
    """Percentage deviation from the simple moving average."""
    sma = close.rolling(window).mean()
    return ((close - sma) / sma).rename("dist_from_sma")


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Current volume relative to its rolling average."""
    return (volume / volume.rolling(window).mean()).rename("volume_ratio")


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    bb_window  = cfg["features"]["bb_window"]
    bb_std     = cfg["features"]["bb_std"]
    rsi_window = cfg["features"]["rsi_window"]
    vol_window = cfg["features"]["vol_window"]
    zs_window  = cfg["features"]["zscore_window"]

    results = []

    for ticker, group in df.groupby("Ticker"):
        logger.info(f"Building features for {ticker}")
        g      = group.sort_values("Date").copy()
        close  = g["Close"]
        volume = g["Volume"]

        feature_df = pd.concat([
            g[["Date", "Ticker", "Close", "Volume"]],
            bollinger_bands(close, bb_window, bb_std),
            zscore(close, zs_window),
            rsi(close, rsi_window),
            rolling_volatility(close, vol_window),
            distance_from_sma(close, bb_window),
            volume_ratio(volume),
        ], axis=1)

        results.append(feature_df)

    combined = pd.concat(results, ignore_index=True)
    before   = len(combined)
    combined = combined.dropna()
    logger.info(f"Features built: {len(combined):,} rows ({before - len(combined):,} dropped for warmup)")

    return combined


def run_features(config_path: str = "config.yaml") -> pd.DataFrame:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    raw_path = Path(cfg["paths"]["raw_data"])
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Run ingest.py first.")

    raw      = pd.read_parquet(raw_path)
    features = build_features(raw, cfg)

    out_path = raw_path.parent / "features.parquet"
    features.to_parquet(out_path, index=False)
    logger.info(f"Saved features → {out_path}")

    return features


if __name__ == "__main__":
    run_features()