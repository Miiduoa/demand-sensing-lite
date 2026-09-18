"""Time-split sklearn forecast with MAE / MAPE."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


FEATURE_COLS = ["dow", "month", "promo", "lag_1", "lag_7", "roll_mean_7", "roll_std_7"]


def _add_lags(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["lag_1"] = g["demand"].shift(1)
    g["lag_7"] = g["demand"].shift(7)
    g["roll_mean_7"] = g["demand"].shift(1).rolling(7).mean()
    g["roll_std_7"] = g["demand"].shift(1).rolling(7).std()
    return g


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    parts = [_add_lags(g) for _, g in df.groupby("sku", sort=False)]
    out = pd.concat(parts, ignore_index=True)
    return out.dropna().reset_index(drop=True)


def time_split_forecast(
    df: pd.DataFrame,
    test_days: int = 56,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Per-SKU HistGradientBoosting with chronological hold-out.

    Returns (predictions_df, metrics_dict).
    """
    feat = build_features(df)
    max_date = feat["date"].max()
    cutoff = max_date - pd.Timedelta(days=test_days - 1)

    train = feat[feat["date"] < cutoff]
    test = feat[feat["date"] >= cutoff]
    if train.empty or test.empty:
        raise ValueError("Insufficient data for time split; increase n_days.")

    preds: list[pd.DataFrame] = []
    mae_list: list[float] = []
    mape_list: list[float] = []

    for sku, tr in train.groupby("sku"):
        te = test[test["sku"] == sku]
        if te.empty or len(tr) < 30:
            continue
        model = HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.08,
            max_iter=120,
            random_state=seed,
        )
        model.fit(tr[FEATURE_COLS], tr["demand"])
        yhat = model.predict(te[FEATURE_COLS])
        yhat = np.clip(yhat, 0, None)
        part = te[["date", "sku", "demand"]].copy()
        part["yhat"] = yhat
        preds.append(part)

        y_true = part["demand"].to_numpy(dtype=float)
        err = np.abs(y_true - yhat)
        mae_list.append(float(err.mean()))
        denom = np.maximum(y_true, 1.0)
        mape_list.append(float((err / denom).mean() * 100.0))

    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    metrics = {
        "mae": float(np.mean(mae_list)) if mae_list else float("nan"),
        "mape": float(np.mean(mape_list)) if mape_list else float("nan"),
        "n_skus": len(mae_list),
        "test_days": test_days,
        "cutoff": str(cutoff.date()),
    }
    return pred_df, metrics
