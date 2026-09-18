"""Seeded synthetic multi-SKU daily demand generator."""

from __future__ import annotations

import numpy as np
import pandas as pd

SKUS = [
    {"sku": "SKU-A", "base": 42.0, "trend": 0.02, "season_amp": 8.0, "noise": 4.0},
    {"sku": "SKU-B", "base": 28.0, "trend": -0.01, "season_amp": 5.0, "noise": 3.0},
    {"sku": "SKU-C", "base": 65.0, "trend": 0.04, "season_amp": 12.0, "noise": 6.0},
    {"sku": "SKU-D", "base": 18.0, "trend": 0.00, "season_amp": 3.5, "noise": 2.5},
    {"sku": "SKU-E", "base": 50.0, "trend": 0.015, "season_amp": 9.0, "noise": 5.0},
]


def generate_demand(
    n_days: int = 365,
    start: str = "2024-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Return long-format daily demand with columns: date, sku, demand, dow, week, month."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n_days, freq="D")
    rows: list[dict] = []

    for meta in SKUS:
        t = np.arange(n_days, dtype=float)
        seasonal = meta["season_amp"] * np.sin(2 * np.pi * t / 7.0)  # weekly
        monthly = 0.4 * meta["season_amp"] * np.sin(2 * np.pi * t / 30.0)
        # mild promo spikes every ~45 days
        promo = np.zeros(n_days)
        promo_days = rng.choice(n_days, size=max(1, n_days // 45), replace=False)
        promo[promo_days] = rng.uniform(8, 18, size=len(promo_days))
        noise = rng.normal(0, meta["noise"], size=n_days)
        demand = meta["base"] + meta["trend"] * t + seasonal + monthly + promo + noise
        demand = np.clip(np.round(demand), 0, None).astype(int)

        for i, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "sku": meta["sku"],
                    "demand": int(demand[i]),
                    "dow": int(d.dayofweek),
                    "week": int(d.isocalendar().week),
                    "month": int(d.month),
                    "promo": int(promo[i] > 0),
                }
            )

    df = pd.DataFrame(rows).sort_values(["sku", "date"]).reset_index(drop=True)
    return df
