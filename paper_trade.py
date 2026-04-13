"""
Paper trading tracker — runs daily to log model signals and track outcomes.

Usage:
    python paper_trade.py          # log today's signals + check 5-day outcomes
    python paper_trade.py --report # print the full scorecard

Run this once a day (after market close) to build a live forward-test record.
Signals and outcomes are stored in paper_trades.csv.
"""

import argparse
import yaml
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

from src.pipeline.features import build_features
from src.utils.logger import get_logger

logger = get_logger("paper_trade")

LOG_FILE     = Path("paper_trades.csv")
HOLD_DAYS    = 5
THRESHOLD    = 0.65
CONFIG_PATH  = "config.yaml"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

FEATURE_COLS = [
    "bb_pct_b", "bb_width", "bb_position",
    "zscore", "rsi", "volatility",
    "dist_from_sma", "volume_ratio",
]


def load_model():
    model_path = Path(CFG["paths"]["model"])
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run python run_pipeline.py first.")
    return joblib.load(model_path)


def fetch_prices(tickers: list) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        df = yf.download(ticker, period="60d", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        df["Ticker"] = ticker
        df.index.name = "Date"
        frames.append(df.reset_index())
    return pd.concat(frames, ignore_index=True)


def load_log() -> pd.DataFrame:
    if LOG_FILE.exists():
        return pd.read_csv(LOG_FILE, parse_dates=["signal_date", "exit_date"])
    return pd.DataFrame(columns=[
        "signal_date", "ticker", "entry_price", "confidence",
        "exit_date", "exit_price", "trade_return", "outcome"
    ])


def save_log(df: pd.DataFrame):
    df.to_csv(LOG_FILE, index=False)


def log_signals(model, tickers: list):
    """Scan tickers today and log any signals above the threshold."""
    today     = pd.Timestamp(datetime.today().date())
    raw       = fetch_prices(tickers)
    features  = build_features(raw, CFG)
    log       = load_log()

    new_signals = []

    for ticker in tickers:
        tf = features[features["Ticker"] == ticker].sort_values("Date")
        if tf.empty:
            continue

        latest = tf.iloc[-1]
        x      = np.array([[latest[c] for c in FEATURE_COLS]])
        prob   = model.predict_proba(x)[0][1]

        if prob < THRESHOLD:
            logger.info(f"{ticker}: NO_SIGNAL ({prob:.2%})")
            continue

        # Check we haven't already logged this ticker today
        if not log.empty and ((log["ticker"] == ticker) & (log["signal_date"] == today)).any():
            logger.info(f"{ticker}: already logged today, skipping.")
            continue

        entry_price = float(latest["Close"])
        exit_date   = today + pd.Timedelta(days=HOLD_DAYS)

        new_signals.append({
            "signal_date": today,
            "ticker":      ticker,
            "entry_price": round(entry_price, 4),
            "confidence":  round(prob, 4),
            "exit_date":   exit_date,
            "exit_price":  None,
            "trade_return": None,
            "outcome":     "PENDING",
        })

        logger.info(f"{ticker}: SIGNAL logged | confidence={prob:.2%} | entry=${entry_price:.2f} | exit_date={exit_date.date()}")

    if new_signals:
        log = pd.concat([log, pd.DataFrame(new_signals)], ignore_index=True)
        save_log(log)
        print(f"\n  {len(new_signals)} new signal(s) logged.")
    else:
        print("\n  No new signals today.")


def check_outcomes():
    """Fill in exit prices and returns for trades that have matured."""
    log   = load_log()
    today = pd.Timestamp(datetime.today().date())

    pending = log[(log["outcome"] == "PENDING") & (log["exit_date"] <= today)]

    if pending.empty:
        print("  No trades ready to settle today.")
        return

    settled = 0
    for idx, row in pending.iterrows():
        ticker = row["ticker"]
        df = yf.download(ticker, period="10d", interval="1d", progress=False, auto_adjust=True)

        if df.empty:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        exit_price   = float(df["Close"].iloc[-1])
        trade_return = (exit_price - row["entry_price"]) / row["entry_price"]
        outcome      = "WIN" if trade_return > 0 else "LOSS"

        log.at[idx, "exit_price"]   = round(exit_price, 4)
        log.at[idx, "trade_return"] = round(trade_return, 6)
        log.at[idx, "outcome"]      = outcome
        settled += 1

        logger.info(f"{ticker}: settled | return={trade_return:+.2%} | {outcome}")

    save_log(log)
    print(f"  {settled} trade(s) settled.")


def print_report():
    """Print the full live scorecard."""
    log = load_log()

    if log.empty:
        print("No trades logged yet. Run python paper_trade.py first.")
        return

    settled = log[log["outcome"].isin(["WIN", "LOSS"])]
    pending = log[log["outcome"] == "PENDING"]

    print("\n" + "═" * 50)
    print("  LIVE PAPER TRADING SCORECARD")
    print("═" * 50)
    print(f"  Total signals:   {len(log)}")
    print(f"  Settled trades:  {len(settled)}")
    print(f"  Pending:         {len(pending)}")

    if not settled.empty:
        win_rate   = (settled["outcome"] == "WIN").mean() * 100
        avg_return = settled["trade_return"].mean() * 100
        total_return = ((1 + settled["trade_return"]).prod() - 1) * 100
        best  = settled["trade_return"].max() * 100
        worst = settled["trade_return"].min() * 100

        print("─" * 50)
        print(f"  Win rate:        {win_rate:.1f}%")
        print(f"  Avg return:      {avg_return:+.2f}%")
        print(f"  Total return:    {total_return:+.2f}%")
        print(f"  Best trade:      {best:+.2f}%")
        print(f"  Worst trade:     {worst:+.2f}%")
        print("─" * 50)
        print("\n  Recent trades:")
        for _, row in settled.tail(10).iterrows():
            print(f"  {str(row['signal_date'].date())}  {row['ticker']:<6}  "
                  f"{row['trade_return']:+.2%}  {row['outcome']}")

    if not pending.empty:
        print("\n  Pending trades:")
        for _, row in pending.iterrows():
            print(f"  {str(row['signal_date'].date())}  {row['ticker']:<6}  "
                  f"entry=${row['entry_price']:.2f}  "
                  f"exits={str(row['exit_date'].date())}")

    print("═" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="Print scorecard only")
    args = parser.parse_args()

    if args.report:
        print_report()
        return

    model   = load_model()
    tickers = CFG["data"]["tickers"]

    print(f"\n  Scanning {len(tickers)} tickers...")
    log_signals(model, tickers)

    print("  Checking settled trades...")
    check_outcomes()

    print_report()


if __name__ == "__main__":
    main()
