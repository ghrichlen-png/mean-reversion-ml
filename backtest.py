"""
backtest.py
══════════════════════════════════════════════════════════════════════
Walk-forward backtest for the Mean Reversion ML Pipeline.

Strategy:
  • Scan each trading day in the test window
  • When the model fires REVERSION_LIKELY (probability ≥ threshold),
    enter a long position at the NEXT day's open
  • Exit after `hold_days` trading days at close
  • No overlapping trades on the same ticker (one position at a time)

Benchmark:
  • Buy-and-hold SPY (S&P 500 ETF) over the same period

Key metrics reported:
  • Total Return          – raw % gain over the period
  • Annualised Return     – CAGR
  • Sharpe Ratio          – risk-adjusted return (annualised, rf=0)
  • Max Drawdown          – worst peak-to-trough loss
  • Win Rate              – % of trades that were profitable
  • Avg Trade Return      – mean return per trade
  • Total Trades          – number of signals acted on
  • Alpha vs S&P 500      – excess annualised return over benchmark

Usage:
  python backtest.py
"""

import warnings
warnings.filterwarnings("ignore")

import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

from src.pipeline.features import build_features
from src.utils.logger import get_logger

logger = get_logger("backtest")

# ── Config ────────────────────────────────────────────────────────
with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

HOLD_DAYS          = CFG["labels"]["reversion_days"]   # days to hold each trade
SIGNAL_THRESHOLD   = 0.65                               # high-confidence signals only
STOP_LOSS          = -0.05                              # cut losses at -5%
INITIAL_CAPITAL    = 10_000                             # starting portfolio value ($)
POSITION_SIZE      = 0.10                               # 10% of capital per trade
BACKTEST_PERIOD    = "3y"                               # how far back to test
BENCHMARK_TICKER   = "SPY"

FEATURE_COLS = [
    "bb_pct_b", "bb_width", "bb_position",
    "zscore", "rsi", "volatility",
    "dist_from_sma", "volume_ratio",
]

# ── Helpers ───────────────────────────────────────────────────────

def compute_metrics(returns: pd.Series, label: str = "Strategy") -> dict:
    """
    Compute a full suite of performance metrics from a daily returns series.
    Returns dict of clean, rounded numbers ready for display.
    """
    r = returns.dropna()
    if len(r) == 0:
        return {}

    # Cumulative returns
    cum = (1 + r).cumprod()
    total_return = cum.iloc[-1] - 1

    # CAGR
    n_years = len(r) / 252
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Sharpe (annualised, risk-free = 0)
    sharpe = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0

    # Max drawdown
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    max_dd = drawdown.min()

    # Volatility (annualised)
    ann_vol = r.std() * np.sqrt(252)

    return {
        "label":           label,
        "total_return":    round(total_return * 100, 2),
        "cagr":            round(cagr * 100, 2),
        "sharpe":          round(sharpe, 3),
        "max_drawdown":    round(max_dd * 100, 2),
        "ann_volatility":  round(ann_vol * 100, 2),
    }


def fetch_price_data(tickers: list, period: str) -> pd.DataFrame:
    """Download OHLCV data for a list of tickers + the benchmark."""
    all_tickers = list(set(tickers + [BENCHMARK_TICKER]))
    logger.info(f"Downloading backtest price data for: {all_tickers}")

    frames = []
    for ticker in all_tickers:
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty:
            logger.warning(f"No data for {ticker}")
            continue

        # Flatten MultiIndex if present (newer yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]

        df["Ticker"] = ticker
        df.index.name = "Date"
        frames.append(df.reset_index())

    return pd.concat(frames, ignore_index=True)


# ── Core backtest ─────────────────────────────────────────────────

def run_backtest():
    # ── Load model ─────────────────────────────────────────────────
    model_path  = Path(CFG["paths"]["model"])
    scaler_path = Path(CFG["paths"]["scaler"])

    if not model_path.exists():
        raise FileNotFoundError("Model not found — run python run_pipeline.py first.")

    model  = joblib.load(model_path)
    logger.info("Model loaded.")

    # ── Fetch fresh price data ──────────────────────────────────────
    tickers  = CFG["data"]["tickers"]
    raw      = fetch_price_data(tickers, BACKTEST_PERIOD)

    # ── Build features ─────────────────────────────────────────────
    logger.info("Building features for backtest window...")
    features = build_features(raw, CFG)
    features["Date"] = pd.to_datetime(features["Date"])

    # ── Merge in next-day open (entry price) and future close (exit) ──
    price_pivot = raw.copy()
    price_pivot["Date"] = pd.to_datetime(price_pivot["Date"])

    trades     = []
    open_positions = {}   # ticker → exit date

    trading_dates = sorted(features["Date"].unique())

    logger.info(f"Scanning {len(trading_dates)} trading days across {len(tickers)} tickers...")

    for ticker in tickers:
        ticker_feat   = features[features["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        ticker_prices = price_pivot[price_pivot["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)

        # Merge open prices for entry (next day open)
        ticker_prices["next_open"]    = ticker_prices["Open"].shift(-1)
        ticker_prices["future_close"] = ticker_prices["Close"].shift(HOLD_DAYS)

        merged = ticker_feat.merge(
            ticker_prices[["Date", "next_open", "future_close", "Close"]],
            on="Date", how="left"
        )

        last_exit = pd.Timestamp("1900-01-01")

        for _, row in merged.iterrows():
            # Skip if we already have an open position on this ticker
            if row["Date"] <= last_exit:
                continue

            # Skip if no entry/exit price available
            if pd.isna(row["next_open"]) or pd.isna(row["future_close"]):
                continue

            # Run model
            x = np.array([[row[c] for c in FEATURE_COLS]])
            prob = model.predict_proba(x)[0][1]

            if prob >= SIGNAL_THRESHOLD:
                entry_price  = row["next_open"]
                exit_price   = row["future_close"]
                trade_return = (exit_price - entry_price) / entry_price
                stopped_out  = trade_return < STOP_LOSS
                trade_return = max(trade_return, STOP_LOSS)

                trades.append({
                    "ticker":       ticker,
                    "entry_date":   row["Date"] + pd.Timedelta(days=1),
                    "exit_date":    row["Date"] + pd.Timedelta(days=HOLD_DAYS + 1),
                    "entry_price":  round(entry_price, 4),
                    "exit_price":   round(exit_price, 4),
                    "trade_return": round(trade_return, 6),
                    "confidence":   round(prob, 4),
                    "stopped_out":  stopped_out,
                })
                last_exit = row["Date"] + pd.Timedelta(days=HOLD_DAYS + 1)

    if not trades:
        logger.error("No trades generated. Check your model and data.")
        return

    trades_df = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)
    logger.info(f"Total trades executed: {len(trades_df)}")

    # ── Build daily equity curve ────────────────────────────────────
    # Create a daily portfolio value series
    all_dates = pd.date_range(
        start=trades_df["entry_date"].min(),
        end=trades_df["exit_date"].max(),
        freq="B"    # business days
    )

    capital    = INITIAL_CAPITAL
    daily_pnl  = pd.Series(0.0, index=all_dates)

    for _, trade in trades_df.iterrows():
        position_value = capital * POSITION_SIZE
        pnl = position_value * trade["trade_return"]

        # Book P&L on exit date
        exit = trade["exit_date"]
        if exit in daily_pnl.index:
            daily_pnl[exit] += pnl

    portfolio_value = INITIAL_CAPITAL + daily_pnl.cumsum()
    strategy_returns = portfolio_value.pct_change().fillna(0)

    # ── Benchmark (SPY buy-and-hold) ────────────────────────────────
    spy_data = raw[raw["Ticker"] == BENCHMARK_TICKER].sort_values("Date")
    spy_data["Date"] = pd.to_datetime(spy_data["Date"])
    spy_data = spy_data.set_index("Date")["Close"]
    spy_data = spy_data.reindex(all_dates, method="ffill").dropna()
    spy_returns = spy_data.pct_change().fillna(0)
    spy_equity  = INITIAL_CAPITAL * (1 + spy_returns).cumprod()

    # ── Compute metrics ─────────────────────────────────────────────
    strat_metrics = compute_metrics(strategy_returns, "Strategy")
    bench_metrics = compute_metrics(spy_returns,      "S&P 500 (SPY)")

    alpha = strat_metrics["cagr"] - bench_metrics["cagr"]

    win_rate   = (trades_df["trade_return"] > 0).mean() * 100
    avg_return = trades_df["trade_return"].mean() * 100
    best_trade = trades_df["trade_return"].max() * 100
    worst_trade= trades_df["trade_return"].min() * 100

    # ── Print results ───────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  BACKTEST RESULTS")
    print("═"*55)
    print(f"  Period:              {BACKTEST_PERIOD}  |  Hold: {HOLD_DAYS} days  |  Stop-loss: {STOP_LOSS:.0%}")
    print(f"  Tickers:             {', '.join(tickers)}")
    print(f"  Initial Capital:     ${INITIAL_CAPITAL:,.0f}")
    print(f"  Total Trades:        {len(trades_df)}")
    print("─"*55)
    print(f"  {'Metric':<26} {'Strategy':>10} {'S&P 500':>10}")
    print("─"*55)
    print(f"  {'Total Return':<26} {strat_metrics['total_return']:>9.1f}% {bench_metrics['total_return']:>9.1f}%")
    print(f"  {'Ann. Return (CAGR)':<26} {strat_metrics['cagr']:>9.1f}% {bench_metrics['cagr']:>9.1f}%")
    print(f"  {'Sharpe Ratio':<26} {strat_metrics['sharpe']:>10.3f} {bench_metrics['sharpe']:>10.3f}")
    print(f"  {'Max Drawdown':<26} {strat_metrics['max_drawdown']:>9.1f}% {bench_metrics['max_drawdown']:>9.1f}%")
    print(f"  {'Ann. Volatility':<26} {strat_metrics['ann_volatility']:>9.1f}% {bench_metrics['ann_volatility']:>9.1f}%")
    print(f"  {'Alpha vs S&P 500':<26} {alpha:>+9.1f}%")
    print("─"*55)
    print(f"  Win Rate:            {win_rate:.1f}%")
    print(f"  Avg Trade Return:    {avg_return:+.2f}%")
    print(f"  Best Trade:          {best_trade:+.2f}%")
    print(f"  Worst Trade:         {worst_trade:+.2f}%")
    print("═"*55 + "\n")

    # ── Plots ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10), facecolor="#0d1117")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    DARK   = "#0d1117"
    PANEL  = "#161b22"
    GREEN  = "#3fb950"
    BLUE   = "#58a6ff"
    RED    = "#f85149"
    GRAY   = "#8b949e"
    WHITE  = "#e6edf3"

    def style_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=WHITE, fontsize=11, fontweight="bold", pad=10)
        ax.tick_params(colors=GRAY, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.yaxis.label.set_color(GRAY)
        ax.xaxis.label.set_color(GRAY)

    # ── Plot 1: Equity curve ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(portfolio_value.index, portfolio_value.values,
             color=GREEN, linewidth=1.8, label="Mean Reversion Strategy", zorder=3)
    ax1.plot(spy_equity.index, spy_equity.values,
             color=BLUE, linewidth=1.5, linestyle="--", label="Buy & Hold SPY", zorder=2)
    ax1.fill_between(portfolio_value.index, INITIAL_CAPITAL,
                     portfolio_value.values,
                     where=portfolio_value.values >= INITIAL_CAPITAL,
                     alpha=0.08, color=GREEN)
    ax1.fill_between(portfolio_value.index, INITIAL_CAPITAL,
                     portfolio_value.values,
                     where=portfolio_value.values < INITIAL_CAPITAL,
                     alpha=0.08, color=RED)
    ax1.axhline(INITIAL_CAPITAL, color=GRAY, linewidth=0.8, linestyle=":", alpha=0.6)
    ax1.set_ylabel("Portfolio Value ($)", color=GRAY)

    # Annotate final values
    final_strat = portfolio_value.iloc[-1]
    final_spy   = spy_equity.iloc[-1]
    ax1.annotate(f"${final_strat:,.0f}  ({strat_metrics['total_return']:+.1f}%)",
                 xy=(portfolio_value.index[-1], final_strat),
                 xytext=(-90, 10), textcoords="offset points",
                 color=GREEN, fontsize=9, fontweight="bold")
    ax1.annotate(f"${final_spy:,.0f}  ({bench_metrics['total_return']:+.1f}%)",
                 xy=(spy_equity.index[-1], final_spy),
                 xytext=(-90, -18), textcoords="offset points",
                 color=BLUE, fontsize=9)

    legend = ax1.legend(facecolor=PANEL, edgecolor="#30363d",
                        labelcolor=WHITE, fontsize=9)
    style_ax(ax1, "Portfolio Equity Curve")

    # ── Plot 2: Drawdown ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    cum_strat = (1 + strategy_returns).cumprod()
    dd_strat  = (cum_strat - cum_strat.cummax()) / cum_strat.cummax() * 100
    ax2.fill_between(dd_strat.index, dd_strat.values, 0,
                     color=RED, alpha=0.6, linewidth=0)
    ax2.plot(dd_strat.index, dd_strat.values, color=RED, linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)", color=GRAY)
    ax2.annotate(f"Max: {strat_metrics['max_drawdown']:.1f}%",
                 xy=(dd_strat.idxmin(), dd_strat.min()),
                 xytext=(10, -15), textcoords="offset points",
                 color=RED, fontsize=8)
    style_ax(ax2, "Strategy Drawdown")

    # ── Plot 3: Trade return distribution ─────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    trade_pcts = trades_df["trade_return"] * 100
    wins  = trade_pcts[trade_pcts >= 0]
    losses= trade_pcts[trade_pcts < 0]
    bins  = np.linspace(trade_pcts.min() - 0.5, trade_pcts.max() + 0.5, 30)
    ax3.hist(wins,   bins=bins, color=GREEN, alpha=0.8, label=f"Wins ({len(wins)})")
    ax3.hist(losses, bins=bins, color=RED,   alpha=0.8, label=f"Losses ({len(losses)})")
    ax3.axvline(0, color=WHITE, linewidth=0.8, linestyle="--", alpha=0.5)
    ax3.axvline(avg_return, color=BLUE, linewidth=1.2,
                linestyle="--", label=f"Avg {avg_return:+.2f}%")
    ax3.set_xlabel("Trade Return (%)", color=GRAY)
    ax3.set_ylabel("Count", color=GRAY)
    legend3 = ax3.legend(facecolor=PANEL, edgecolor="#30363d",
                         labelcolor=WHITE, fontsize=8)
    style_ax(ax3, f"Trade Return Distribution  (Win Rate: {win_rate:.0f}%)")

    # ── Title & metrics banner ────────────────────────────────────
    fig.suptitle(
        f"Mean Reversion ML Strategy  ·  Sharpe {strat_metrics['sharpe']:.2f}  ·  "
        f"Alpha {alpha:+.1f}%  ·  {len(trades_df)} Trades",
        color=WHITE, fontsize=13, fontweight="bold", y=0.98
    )

    out_path = Path("backtest_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    logger.info(f"Chart saved → {out_path}")
    print(f"  Chart saved → {out_path.resolve()}\n")

    # ── Save trade log ─────────────────────────────────────────────
    log_path = Path("backtest_trades.csv")
    trades_df.to_csv(log_path, index=False)
    logger.info(f"Trade log saved → {log_path}")
    print(f"  Trade log saved → {log_path.resolve()}\n")

    return trades_df, strat_metrics, bench_metrics


if __name__ == "__main__":
    run_backtest()
