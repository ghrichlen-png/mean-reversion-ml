"""
Label generation and model training.

A reversion event (label=1) is defined as: the stock's z-score exceeds 1.5
in magnitude AND price returns toward its rolling mean within N days.
"""

import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from src.utils.logger import get_logger

logger = get_logger(__name__)

FEATURE_COLS = [
    "bb_pct_b",
    "bb_width",
    "bb_position",
    "zscore",
    "rsi",
    "volatility",
    "dist_from_sma",
    "volume_ratio",
]


def create_labels(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rev_days      = cfg["labels"]["reversion_days"]
    rev_threshold = cfg["labels"]["reversion_threshold"]
    bb_window     = cfg["features"]["bb_window"]

    labeled = []

    for ticker, group in df.groupby("Ticker"):
        g     = group.sort_values("Date").copy()
        g["label"] = 0

        close = g["Close"].values
        sma   = g["Close"].rolling(bb_window).mean().values
        z     = g["zscore"].values

        for i in range(len(g) - rev_days):
            if abs(z[i]) < 1.5:
                continue

            future      = close[i + 1 : i + rev_days + 1]
            dist_today  = close[i] - sma[i]

            if any(abs(fc - sma[i]) < abs(dist_today) * (1 - rev_threshold) for fc in future):
                g.iloc[i, g.columns.get_loc("label")] = 1

        labeled.append(g)
        logger.info(f"{ticker}: {g['label'].mean() * 100:.1f}% of extreme days reverted")

    return pd.concat(labeled, ignore_index=True)


def train(config_path: str = "config.yaml") -> RandomForestClassifier:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    feat_path = Path(cfg["paths"]["raw_data"]).parent / "features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Features not found at {feat_path}. Run features.py first.")

    df = pd.read_parquet(feat_path)
    logger.info(f"Loaded {len(df):,} feature rows")

    df = create_labels(df, cfg)
    logger.info(f"Label rate: {df['label'].mean() * 100:.1f}% positive")

    X = df[FEATURE_COLS].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["model"]["test_size"],
        random_state=cfg["model"]["random_state"],
        stratify=y,
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    scaler = StandardScaler()
    scaler.fit(X_train)

    model = RandomForestClassifier(
        n_estimators=cfg["model"]["n_estimators"],
        max_depth=cfg["model"]["max_depth"],
        random_state=cfg["model"]["random_state"],
        class_weight="balanced",
        n_jobs=-1,
    )
    logger.info("Training Random Forest...")
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    logger.info(f"\n{classification_report(y_test, y_pred)}")
    logger.info(f"ROC-AUC: {auc:.4f}")

    importance = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: x[1], reverse=True)
    logger.info("Feature importances:")
    for feat, imp in importance:
        logger.info(f"  {feat:<18} {imp:.4f}")

    model_dir = Path(cfg["paths"]["model"]).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model,  cfg["paths"]["model"])
    joblib.dump(scaler, cfg["paths"]["scaler"])
    logger.info(f"Saved model → {cfg['paths']['model']}")

    return model


if __name__ == "__main__":
    train()
