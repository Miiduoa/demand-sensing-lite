"""CLI: python -m demand_sensing_lite"""

from __future__ import annotations

import argparse

from .pipeline import run_demo


def main() -> None:
    p = argparse.ArgumentParser(description="Demand Sensing Lite — forecast + (s,S) demo")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--test-days", type=int, default=56)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lead-time", type=int, default=3)
    p.add_argument("--doc", type=float, default=7.0, help="days of cover for (s,S)")
    args = p.parse_args()

    result = run_demo(
        n_days=args.days,
        test_days=args.test_days,
        seed=args.seed,
        lead_time=args.lead_time,
        days_of_cover=args.doc,
    )
    m = result["forecast_metrics"]
    inv = result["inventory"]

    print("=== Demand Sensing Lite ===")
    print(f"SKUs={m['n_skus']}  holdout={m['test_days']}d  cutoff={m['cutoff']}")
    print(f"Forecast  MAE={m['mae']:.2f}  MAPE={m['mape']:.1f}%")
    print()
    print("--- Inventory: (s,S) with forecast vs naive ---")
    print(inv.to_string(index=False))
    print()
    print(
        f"Avg service level  (s,S)={inv['ss_service_level'].mean():.3f}  "
        f"naive={inv['naive_service_level'].mean():.3f}"
    )
    print(
        f"Avg inventory      (s,S)={inv['ss_avg_inventory'].mean():.1f}  "
        f"naive={inv['naive_avg_inventory'].mean():.1f}"
    )


if __name__ == "__main__":
    main()
