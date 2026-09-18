"""End-to-end demo pipeline."""

from __future__ import annotations

from .forecast import time_split_forecast
from .generate import generate_demand
from .inventory import compare_policies


def run_demo(
    n_days: int = 365,
    test_days: int = 56,
    seed: int = 42,
    lead_time: int = 3,
    days_of_cover: float = 7.0,
) -> dict:
    demand = generate_demand(n_days=n_days, seed=seed)
    pred_df, metrics = time_split_forecast(demand, test_days=test_days, seed=seed)
    inv = compare_policies(pred_df, lead_time=lead_time, days_of_cover=days_of_cover)
    return {
        "demand": demand,
        "predictions": pred_df,
        "forecast_metrics": metrics,
        "inventory": inv,
    }
