"""
FastAPI app — serves mean reversion predictions.

Endpoints:
    GET  /         health check
    GET  /health   model status
    POST /predict  run prediction for a ticker
    GET  /docs     Swagger UI
"""

import joblib
import numpy as np
import yaml
import yfinance as yf
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import PredictRequest, PredictResponse, SignalDetails
from src.pipeline.features import (
    bollinger_bands, zscore, rsi,
    rolling_volatility, distance_from_sma, volume_ratio,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

MODEL_PATH  = Path(CFG["paths"]["model"])
SCALER_PATH = Path(CFG["paths"]["scaler"])

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        _state["model"]  = joblib.load(MODEL_PATH)
        _state["scaler"] = joblib.load(SCALER_PATH)
        logger.info("Model and scaler loaded.")
    else:
        logger.warning("Model files not found — run the training pipeline first.")
    yield
    _state.clear()


app = FastAPI(
    title="Mean Reversion Signal API",
    description="Predicts whether a stock is likely to mean-revert based on Bollinger Bands, Z-score, RSI, and volatility features.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def compute_live_features(ticker: str, lookback_days: int) -> dict:
    period = f"{lookback_days + 60}d"
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for ticker '{ticker}'")

    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    bb   = bollinger_bands(close, CFG["features"]["bb_window"], CFG["features"]["bb_std"])
    zs   = zscore(close, CFG["features"]["zscore_window"])
    rs   = rsi(close, CFG["features"]["rsi_window"])
    vol  = rolling_volatility(close, CFG["features"]["vol_window"])
    dist = distance_from_sma(close, CFG["features"]["bb_window"])
    volr = volume_ratio(volume)

    latest = {
        "bb_pct_b":      float(bb["bb_pct_b"].iloc[-1]),
        "bb_width":      float(bb["bb_width"].iloc[-1]),
        "bb_position":   float(bb["bb_position"].iloc[-1]),
        "zscore":        float(zs.iloc[-1]),
        "rsi":           float(rs.iloc[-1]),
        "volatility":    float(vol.iloc[-1]),
        "dist_from_sma": float(dist.iloc[-1]),
        "volume_ratio":  float(volr.iloc[-1]),
        "last_close":    float(close.iloc[-1]),
    }

    if any(np.isnan(v) for v in latest.values()):
        raise HTTPException(status_code=422, detail="Insufficient data. Try a longer lookback_days.")

    return latest


@app.get("/", tags=["Meta"])
def root():
    return {"message": "Mean Reversion Signal API — visit /docs for the Swagger UI."}


@app.get("/health", tags=["Meta"])
def health():
    model_loaded = "model" in _state
    return {"status": "ok" if model_loaded else "degraded", "model_loaded": model_loaded}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest) -> PredictResponse:
    """
    Predict whether a stock is in a mean reversion setup.

    - **ticker**: e.g. `AAPL`, `TSLA`, `JPM`
    - **lookback_days**: days of history to use (default 60)
    """
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded. Run the training pipeline first.")

    ticker = request.ticker.upper().strip()
    feats  = compute_live_features(ticker, request.lookback_days)

    feature_vector = np.array([[
        feats["bb_pct_b"],
        feats["bb_width"],
        feats["bb_position"],
        feats["zscore"],
        feats["rsi"],
        feats["volatility"],
        feats["dist_from_sma"],
        feats["volume_ratio"],
    ]])

    confidence = float(_state["model"].predict_proba(feature_vector)[0][1])
    signal     = "REVERSION_LIKELY" if confidence >= 0.5 else "NO_SIGNAL"

    z  = feats["zscore"]
    rs = feats["rsi"]
    bb = feats["bb_pct_b"]

    direction = "above" if z > 0 else "below"
    rsi_label = "overbought" if rs > 70 else ("oversold" if rs < 30 else "neutral RSI")
    interpretation = (
        f"{ticker} is {abs(z):.2f} std devs {direction} its 20-day average "
        f"with {rsi_label} (RSI {rs:.1f}). "
        f"Bollinger %B = {bb:.2f}. "
        f"Model assigns {confidence:.0%} probability of reversion within 5 days."
    )

    logger.info(f"Prediction → ticker={ticker} signal={signal} confidence={confidence:.3f}")

    return PredictResponse(
        ticker=ticker,
        signal=signal,
        confidence=round(confidence, 4),
        last_close=feats["last_close"],
        interpretation=interpretation,
        features=SignalDetails(
            zscore=round(feats["zscore"], 4),
            rsi=round(feats["rsi"], 2),
            bb_pct_b=round(feats["bb_pct_b"], 4),
            dist_from_sma=round(feats["dist_from_sma"], 4),
            volatility=round(feats["volatility"], 4),
            volume_ratio=round(feats["volume_ratio"], 4),
        ),
    )