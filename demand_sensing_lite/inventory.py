"""(s,S) / days-of-cover reorder simulation vs naive baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _simulate_policy(
    demand: np.ndarray,
    forecast: np.ndarray | None,
    policy: str,
    lead_time: int = 3,
    review: int = 1,
    days_of_cover: float = 7.0,
    initial_inventory: float | None = None,
) -> dict:
    """Simulate daily inventory.

    policy:
      - 'ss': (s,S) using forecast mean * days_of_cover for S, s = 0.5 * S
      - 'naive': reorder fixed qty = mean(train demand) when stock < mean * 3
    """
    n = len(demand)
    if forecast is None:
        forecast = np.full(n, float(np.mean(demand[: max(14, n // 4)])))

    mean_d = float(np.mean(demand[: max(14, n // 4)]))
    if initial_inventory is None:
        initial_inventory = mean_d * days_of_cover

    inv = float(initial_inventory)
    pipeline: list[tuple[int, float]] = []  # (arrive_day, qty)
    stockouts = 0
    fulfilled = 0.0
    total_demand = 0.0
    inv_trace: list[float] = []
    orders = 0

    for t in range(n):
        # arrivals
        arrive = [q for day, q in pipeline if day == t]
        inv += sum(arrive)
        pipeline = [(day, q) for day, q in pipeline if day != t]

        d = float(demand[t])
        total_demand += d
        shipped = min(inv, d)
        fulfilled += shipped
        if shipped < d - 1e-9:
            stockouts += 1
        inv -= shipped
        inv_trace.append(inv)

        # reorder decision
        if t % review != 0:
            continue

        on_order = sum(q for _, q in pipeline)
        position = inv + on_order
        f = max(float(forecast[t]), 0.1)

        if policy == "ss":
            S = f * days_of_cover + f * lead_time
            s = 0.5 * S
            if position <= s:
                qty = max(S - position, 0.0)
                if qty > 0:
                    pipeline.append((t + lead_time, qty))
                    orders += 1
        else:  # naive
            threshold = mean_d * 3
            if position < threshold:
                qty = mean_d * days_of_cover
                pipeline.append((t + lead_time, qty))
                orders += 1

    service_level = 1.0 - stockouts / n
    fill_rate = fulfilled / total_demand if total_demand > 0 else 1.0
    return {
        "service_level": service_level,
        "fill_rate": fill_rate,
        "avg_inventory": float(np.mean(inv_trace)),
        "orders": orders,
        "stockout_days": stockouts,
    }


def compare_policies(
    pred_df: pd.DataFrame,
    lead_time: int = 3,
    days_of_cover: float = 7.0,
) -> pd.DataFrame:
    """Compare (s,S) using forecast vs naive baseline per SKU on hold-out window."""
    rows = []
    for sku, g in pred_df.groupby("sku"):
        g = g.sort_values("date")
        demand = g["demand"].to_numpy(dtype=float)
        yhat = g["yhat"].to_numpy(dtype=float)
        ss = _simulate_policy(demand, yhat, "ss", lead_time=lead_time, days_of_cover=days_of_cover)
        nv = _simulate_policy(demand, None, "naive", lead_time=lead_time, days_of_cover=days_of_cover)
        rows.append(
            {
                "sku": sku,
                "ss_service_level": round(ss["service_level"], 4),
                "ss_avg_inventory": round(ss["avg_inventory"], 1),
                "ss_orders": ss["orders"],
                "naive_service_level": round(nv["service_level"], 4),
                "naive_avg_inventory": round(nv["avg_inventory"], 1),
                "naive_orders": nv["orders"],
            }
        )
    return pd.DataFrame(rows)
