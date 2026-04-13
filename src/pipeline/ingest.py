"""
Download OHLCV stock data via yfinance and save to Parquet.
"""

import yaml
import yfinance as yf
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def fetch_ticker(ticker: str, period: str, interval: str) -> pd.DataFrame:
    logger.info(f"Fetching {ticker} | period={period} | interval={interval}")
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)

    if df.empty:
        logger.warning(f"No data returned for {ticker} — skipping.")
        return pd.DataFrame()

    df["Ticker"] = ticker
    df.index.name = "Date"
    return df.reset_index()


def ingest(config_path: str = "config.yaml") -> pd.DataFrame:
    cfg      = load_config(config_path)
    tickers  = cfg["data"]["tickers"]
    period   = cfg["data"]["period"]
    interval = cfg["data"]["interval"]
    out_path = Path(cfg["paths"]["raw_data"])

    frames = []
    for ticker in tickers:
        df = fetch_ticker(ticker, period, interval)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("No data fetched. Check tickers and internet connection.")

    combined = pd.concat(frames, ignore_index=True)

    # yfinance v0.2+ returns MultiIndex columns when downloading multiple tickers
    if isinstance(combined.columns, pd.MultiIndex):
        combined.columns = [col[0] for col in combined.columns]
    combined = combined.loc[:, ~combined.columns.duplicated()]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(combined):,} rows → {out_path}")

    return combined


if __name__ == "__main__":
    ingest()