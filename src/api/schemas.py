"""
api/schemas.py
──────────────────────────────────────────
Pydantic models for FastAPI request/response validation.
Pydantic ensures the API rejects bad inputs automatically.
"""

from pydantic import BaseModel, Field
from typing import Literal


class PredictRequest(BaseModel):
    """Input: a stock ticker symbol."""
    ticker: str = Field(
        ...,
        description="Stock ticker symbol (e.g. 'AAPL', 'MSFT')",
        examples=["AAPL"],
        min_length=1,
        max_length=10,
    )
    lookback_days: int = Field(
        default=60,
        description="How many recent days to use for feature calculation",
        ge=30,
        le=365,
    )


class SignalDetails(BaseModel):
    """Computed feature values used to generate the prediction."""
    zscore:         float
    rsi:            float
    bb_pct_b:       float
    dist_from_sma:  float
    volatility:     float
    volume_ratio:   float


class PredictResponse(BaseModel):
    """Output: reversion signal and model confidence."""
    ticker:              str
    signal:              Literal["REVERSION_LIKELY", "NO_SIGNAL"]
    confidence:          float = Field(description="Model confidence (0–1)")
    last_close:          float = Field(description="Most recent closing price")
    interpretation:      str   = Field(description="Plain-English explanation")
    features:            SignalDetails
