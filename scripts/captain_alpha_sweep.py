"""Cheap eval: for each quantile alpha, retrain a captain model and measure
the actual points the top-1-by-prediction player scored per GW on 2025-26.

Used to find the best alpha for captain selection before committing to a full
multi-season backtest.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from fpl_agent.features import build_features
from fpl_agent.model import default_params_quantile, predict, train

SEASON = "2025-26"
ALPHAS = (0.85, 0.90, 0.95, 0.99)


def _captain_actuals(rows: pd.DataFrame, pred_col: str) -> tuple[float, float]:
    """Mean and median actual points scored by the per-GW top-1 captain
    selected by `pred_col`. Aggregates across DGW fixtures per player per GW."""
    picks = []
    for _, g in rows.groupby("GW"):
        by_id = g.groupby("element").agg(
            pred=(pred_col, "sum"),
            actual=("total_points", "sum"),
        )
        if by_id.empty:
            continue
        picks.append(by_id.sort_values("pred", ascending=False).iloc[0]["actual"])
    return float(np.mean(picks)), float(np.median(picks))


def main() -> None:
    df = pd.read_parquet("data/processed/gw_history.parquet")
    print(f"Loaded {len(df):,} rows; building features...")
    t = time.time()
    feats = build_features(df)
    print(f"Features built in {time.time()-t:.1f}s")

    feats_s = feats[feats["season"] == SEASON].copy()
    print(f"Test season {SEASON}: {len(feats_s):,} rows")

    # Hindsight perfect captain — upper bound for context
    perfect_mean = []
    for _, g in feats_s.groupby("GW"):
        by_id = g.groupby("element")["total_points"].sum()
        if not by_id.empty:
            perfect_mean.append(by_id.max())
    print(f"\nHindsight perfect captain: avg actual = {np.mean(perfect_mean):.2f} "
          f"(upper bound)\n")

    print(f"{'alpha':>7}  {'val_loss':>8}  {'cap_avg':>8}  {'cap_med':>8}  "
          f"{'pred_max':>9}  {'pred_p95':>9}")

    for alpha in ALPHAS:
        t = time.time()
        booster, m = train(
            feats, train_until="2023-24", val_season="2024-25",
            params=default_params_quantile(alpha),
        )
        # Use full features (predict handles split_xy internally)
        preds = predict(booster, feats)
        feats_pred_idx = feats_s.index.intersection(preds.index)
        rows = feats_s.loc[feats_pred_idx].copy()
        rows["pred"] = preds.loc[feats_pred_idx]
        pred_s = rows["pred"]

        cap_mean, cap_med = _captain_actuals(rows, "pred")
        print(f"  {alpha:>5.2f}  {m['val_mae']:>8.3f}  "
              f"{cap_mean:>8.2f}  {cap_med:>8.2f}  "
              f"{pred_s.max():>9.2f}  {pred_s.quantile(0.95):>9.2f}  "
              f"({time.time()-t:.1f}s)")


if __name__ == "__main__":
    main()
