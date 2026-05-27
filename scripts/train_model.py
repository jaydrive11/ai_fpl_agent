"""Train both xPts models: L2-mean (squad/XI) + P90 quantile (captain ceiling)."""
import time

import pandas as pd
from sklearn.metrics import mean_absolute_error

from fpl_agent.features import build_features, split_xy
from fpl_agent.model import (
    DEFAULT_MEAN_PATH, DEFAULT_P90_PATH,
    default_params_mean, default_params_p90,
    evaluate_season, save_model, save_two_stage, train, train_two_stage,
)


def main() -> None:
    df = pd.read_parquet("data/processed/gw_history.parquet")
    print(f"Loaded {len(df):,} rows across {df['season'].nunique()} seasons")

    t = time.time()
    df_feat = build_features(df)
    print(f"Built features in {time.time()-t:.1f}s -> shape {df_feat.shape}")

    # ---- MEAN model (L2 regression, used for squad + XI selection) ----
    print("\n=== Training MEAN model (L2) ===")
    t = time.time()
    booster_mean, m = train(df_feat, train_until="2023-24", val_season="2024-25",
                            params=default_params_mean())
    print(f"  trained in {time.time()-t:.1f}s; "
          f"train MAE {m['train_mae']:.3f}, val MAE {m['val_mae']:.3f}, "
          f"best iter {m['best_iteration']}")
    save_model(booster_mean, DEFAULT_MEAN_PATH)
    print(f"  saved -> {DEFAULT_MEAN_PATH}")

    print("\nMEAN test on 2025-26:")
    mean_test = evaluate_season(booster_mean, df_feat, "2025-26")
    print(f"  overall MAE: {mean_test['mae']:.3f}")
    for pos in ("GK", "DEF", "MID", "FWD"):
        if f"mae_{pos}" in mean_test:
            print(f"  {pos}: MAE {mean_test[f'mae_{pos}']:.3f}")

    # ---- P90 model (quantile regression, used for captain choice) ----
    print("\n=== Training P90 ceiling model (quantile α=0.9) ===")
    t = time.time()
    booster_p90, m = train(df_feat, train_until="2023-24", val_season="2024-25",
                           params=default_params_p90())
    print(f"  trained in {time.time()-t:.1f}s; "
          f"train MAE {m['train_mae']:.3f}, val MAE {m['val_mae']:.3f}, "
          f"best iter {m['best_iteration']}")
    save_model(booster_p90, DEFAULT_P90_PATH)
    print(f"  saved -> {DEFAULT_P90_PATH}")

    # ---- Compare prediction distributions on 2025-26 ----
    print("\n=== Prediction distributions on 2025-26 ===")
    X25, y25 = split_xy(df_feat[df_feat["season"] == "2025-26"])
    mean_preds = booster_mean.predict(X25)
    p90_preds = booster_p90.predict(X25)
    print(f"  ACTUAL:  mean={y25.mean():.2f}  std={y25.std():.2f}  "
          f"max={y25.max():.0f}  P95={y25.quantile(0.95):.0f}")
    print(f"  MEAN model: mean={mean_preds.mean():.2f}  std={mean_preds.std():.2f}  "
          f"max={mean_preds.max():.2f}  P95={pd.Series(mean_preds).quantile(0.95):.2f}")
    print(f"  P90 model:  mean={p90_preds.mean():.2f}  std={p90_preds.std():.2f}  "
          f"max={p90_preds.max():.2f}  P95={pd.Series(p90_preds).quantile(0.95):.2f}")

    # ---- TWO-STAGE models (hurdle decomposition) ----
    print("\n=== Training TWO-STAGE models (stage1 classifier + stage2 regressors) ===")
    t = time.time()
    two_stage = train_two_stage(df_feat, train_until="2023-24", val_season="2024-25")
    save_two_stage(two_stage)
    print(f"  trained in {time.time()-t:.1f}s (stage1 + stage2_mean + stage2_p90)")

    print("\nTwo-stage prediction distributions on 2025-26:")
    X25, y25 = split_xy(df_feat[df_feat["season"] == "2025-26"])
    ts_mean = two_stage["predictor_mean"].predict(X25)
    ts_p90 = two_stage["predictor_p90"].predict(X25)
    s1_prob = two_stage["stage1"].predict(X25)
    print(f"  P(plays>=60):    mean={s1_prob.mean():.3f}  P90={pd.Series(s1_prob).quantile(0.9):.3f}")
    print(f"  combined MEAN:   mean={ts_mean.mean():.2f}  std={ts_mean.std():.2f}  "
          f"max={ts_mean.max():.2f}")
    print(f"  combined P90:    mean={ts_p90.mean():.2f}  std={ts_p90.std():.2f}  "
          f"max={ts_p90.max():.2f}")

    # ---- Baselines ----
    print("\n=== Baselines on 2025-26 ===")
    mean_pred = [y25.mean()] * len(y25)
    print(f"  predict mean ({y25.mean():.2f}):   MAE {mean_absolute_error(y25, mean_pred):.3f}")
    lag = X25["total_points_lag1"].fillna(0).values
    print(f"  predict last GW (lag1):    MAE {mean_absolute_error(y25, lag):.3f}")
    form = X25["total_points_avg5"].fillna(0).values
    print(f"  predict form (avg5):       MAE {mean_absolute_error(y25, form):.3f}")


if __name__ == "__main__":
    main()
