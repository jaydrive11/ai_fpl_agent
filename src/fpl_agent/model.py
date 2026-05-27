"""LightGBM xPts model: train, evaluate, save/load."""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_error

from . import storage
from .features import CATEGORICAL_FEATURES, FEATURES, TARGET, split_xy

DEFAULT_MEAN_PATH: Path = storage.processed_dir() / "xpts_mean.lgb"
DEFAULT_P90_PATH: Path = storage.processed_dir() / "xpts_p90.lgb"
# Back-compat alias for older scripts/tests that imported DEFAULT_MODEL_PATH.
DEFAULT_MODEL_PATH: Path = DEFAULT_MEAN_PATH

# Two-stage model artifact paths
STAGE1_PATH: Path = storage.processed_dir() / "xpts_stage1.lgb"
STAGE2_MEAN_PATH: Path = storage.processed_dir() / "xpts_stage2_mean.lgb"
STAGE2_P90_PATH: Path = storage.processed_dir() / "xpts_stage2_p90.lgb"

# Minutes cutoff that defines "played" for the two-stage decomposition.
# FPL awards 2 appearance pts at 60+ min, so this is the natural boundary.
PLAY_THRESHOLD_MINUTES: int = 60


def default_params_mean() -> dict:
    """L2 (squared-error) regression. Spreads predictions wider than L1 —
    less shrinkage toward 0 on heavy zero-inflated FPL data, so high-ceiling
    players get the high predictions they deserve."""
    return {
        "objective": "regression",      # L2 loss
        "metric": "mae",                # still report MAE for comparability
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "num_threads": 4,  # cap explicitly — default num_threads=0 thrashes on many-core hosts
    }


def default_params_quantile(alpha: float = 0.9) -> dict:
    """Quantile regression — predicts the alpha-th percentile outcome.
    Used for captain selection where upside matters more than mean."""
    return {
        "objective": "quantile",
        "alpha": alpha,
        "metric": "quantile",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "num_threads": 4,
    }


def default_params_p90() -> dict:
    """Back-compat: quantile regression at alpha=0.9."""
    return default_params_quantile(0.9)


def default_params_classifier() -> dict:
    """Binary classifier for P(plays >= 60 minutes). Stage 1 of two-stage model."""
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "num_threads": 4,
    }


def train(
    features_df: pd.DataFrame,
    *,
    train_until: str = "2023-24",
    val_season: str = "2024-25",
    num_rounds: int = 2000,
    params: dict | None = None,
) -> tuple[lgb.Booster, dict[str, float]]:
    """Train on all rows with season <= train_until; early stop on val_season.

    `features_df` must be the output of `features.build_features(df_raw)` —
    callers are responsible for building features once and passing through,
    so we don't re-compute them per call.
    """
    train_df = features_df[features_df["season"] <= train_until]
    val_df = features_df[features_df["season"] == val_season]

    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)

    if params is None:
        params = default_params_mean()

    train_set = lgb.Dataset(
        X_train, y_train, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False
    )
    val_set = lgb.Dataset(
        X_val, y_val, categorical_feature=CATEGORICAL_FEATURES,
        reference=train_set, free_raw_data=False,
    )

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_rounds,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    metrics = {
        "train_mae": float(mean_absolute_error(y_train, booster.predict(X_train))),
        "val_mae": float(mean_absolute_error(y_val, booster.predict(X_val))),
        "best_iteration": int(booster.best_iteration or num_rounds),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
    }
    return booster, metrics


def predict(booster: lgb.Booster, features_df: pd.DataFrame) -> pd.Series:
    """Predict xPts for every row in features_df that has at least 1 GW of prior
    history. Returns a Series indexed like the surviving rows."""
    X, _ = split_xy(features_df)
    preds = booster.predict(X)
    return pd.Series(preds, index=X.index, name="xpts")


def evaluate_season(
    booster: lgb.Booster, features_df: pd.DataFrame, season: str
) -> dict[str, float]:
    """MAE overall and per position on a held-out season."""
    season_df = features_df[features_df["season"] == season]
    X, y = split_xy(season_df)
    preds = booster.predict(X)

    out: dict[str, float] = {"mae": float(mean_absolute_error(y, preds)), "n": float(len(y))}
    eval_df = pd.DataFrame({"y": y.values, "pred": preds, "pos": X["position"].values})
    for pos, sub in eval_df.groupby("pos", observed=True):
        out[f"mae_{pos}"] = float(mean_absolute_error(sub["y"], sub["pred"]))
        out[f"n_{pos}"] = float(len(sub))
    return out


def save_model(booster: lgb.Booster, path: Path = DEFAULT_MEAN_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))
    return path


def load_model(path: Path = DEFAULT_MEAN_PATH) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))


# ---------------------------------------------------------------------------
# Two-stage model: hurdle decomposition for zero-inflated FPL points
#
#   E[points]        = P(plays >= 60 min) * E[points | plays >= 60 min]
#   P90[points]      = P(plays >= 60 min) * P90[points | plays >= 60 min]
#
# Stage 1 (classifier): trained on ALL rows, target = (minutes >= 60).
# Stage 2 (regressors): trained on ROWS WHERE minutes >= 60 only —
#   isolates the "playing" regime so predictions aren't dragged toward 0
#   by the ~62% of rows that are bench warmers.
# ---------------------------------------------------------------------------


class TwoStagePredictor:
    """Duck-types lgb.Booster.predict() so backtest/build_players_for_gw can
    use it interchangeably with a regular booster."""

    def __init__(self, stage1: lgb.Booster, stage2: lgb.Booster):
        self.stage1 = stage1
        self.stage2 = stage2

    def predict(self, X) -> "pd.Series":  # type: ignore[name-defined]
        p_plays = self.stage1.predict(X)
        e_given_plays = self.stage2.predict(X)
        return p_plays * e_given_plays

    def feature_name(self) -> list[str]:
        return list(self.stage1.feature_name())

    def feature_importance(self, importance_type: str = "gain"):
        # Returns the stage-2 importance — that's where the "what drives points"
        # signal lives. Stage 1 importance is "what drives starts".
        return self.stage2.feature_importance(importance_type=importance_type)

    @property
    def best_iteration(self) -> int:
        return max(self.stage1.best_iteration or 0, self.stage2.best_iteration or 0)


def _train_lgb(
    X_train, y_train, X_val, y_val, params: dict, num_rounds: int,
) -> lgb.Booster:
    train_set = lgb.Dataset(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES,
                            free_raw_data=False)
    val_set = lgb.Dataset(X_val, y_val, categorical_feature=CATEGORICAL_FEATURES,
                          reference=train_set, free_raw_data=False)
    return lgb.train(
        params, train_set, num_boost_round=num_rounds,
        valid_sets=[train_set, val_set], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )


def train_stage1(
    features_df: pd.DataFrame,
    *,
    train_until: str = "2023-24",
    val_season: str = "2024-25",
    num_rounds: int = 2000,
    threshold: int = PLAY_THRESHOLD_MINUTES,
) -> lgb.Booster:
    """Binary classifier: P(plays >= `threshold` minutes)."""
    df = features_df.assign(plays=(features_df["minutes"] >= threshold).astype(int))
    X_tr, y_tr = split_xy(df[df["season"] <= train_until], target_col="plays")
    X_va, y_va = split_xy(df[df["season"] == val_season], target_col="plays")
    return _train_lgb(X_tr, y_tr, X_va, y_va, default_params_classifier(), num_rounds)


def _train_stage2(
    features_df: pd.DataFrame,
    *,
    train_until: str,
    val_season: str,
    num_rounds: int,
    params: dict,
    threshold: int,
) -> lgb.Booster:
    """Stage 2: regressor trained ONLY on rows where minutes >= threshold."""
    df = features_df[features_df["minutes"] >= threshold]
    X_tr, y_tr = split_xy(df[df["season"] <= train_until])
    X_va, y_va = split_xy(df[df["season"] == val_season])
    return _train_lgb(X_tr, y_tr, X_va, y_va, params, num_rounds)


def train_two_stage(
    features_df: pd.DataFrame,
    *,
    train_until: str = "2023-24",
    val_season: str = "2024-25",
    num_rounds: int = 2000,
    threshold: int = PLAY_THRESHOLD_MINUTES,
) -> dict:
    """Train all three sub-models. Returns a dict with stage1, stage2_mean,
    stage2_p90, and two ready-to-use TwoStagePredictor wrappers."""
    stage1 = train_stage1(features_df, train_until=train_until,
                          val_season=val_season, num_rounds=num_rounds,
                          threshold=threshold)
    stage2_mean = _train_stage2(features_df, train_until=train_until,
                                 val_season=val_season, num_rounds=num_rounds,
                                 params=default_params_mean(), threshold=threshold)
    stage2_p90 = _train_stage2(features_df, train_until=train_until,
                                val_season=val_season, num_rounds=num_rounds,
                                params=default_params_quantile(0.9), threshold=threshold)
    return {
        "stage1": stage1,
        "stage2_mean": stage2_mean,
        "stage2_p90": stage2_p90,
        "predictor_mean": TwoStagePredictor(stage1, stage2_mean),
        "predictor_p90": TwoStagePredictor(stage1, stage2_p90),
    }


def save_two_stage(models: dict, *,
                   stage1_path: Path = STAGE1_PATH,
                   stage2_mean_path: Path = STAGE2_MEAN_PATH,
                   stage2_p90_path: Path = STAGE2_P90_PATH) -> None:
    stage1_path.parent.mkdir(parents=True, exist_ok=True)
    models["stage1"].save_model(str(stage1_path))
    models["stage2_mean"].save_model(str(stage2_mean_path))
    models["stage2_p90"].save_model(str(stage2_p90_path))


def load_two_stage(stage1_path: Path = STAGE1_PATH,
                   stage2_mean_path: Path = STAGE2_MEAN_PATH,
                   stage2_p90_path: Path = STAGE2_P90_PATH) -> dict:
    stage1 = lgb.Booster(model_file=str(stage1_path))
    stage2_mean = lgb.Booster(model_file=str(stage2_mean_path))
    stage2_p90 = lgb.Booster(model_file=str(stage2_p90_path))
    return {
        "stage1": stage1,
        "stage2_mean": stage2_mean,
        "stage2_p90": stage2_p90,
        "predictor_mean": TwoStagePredictor(stage1, stage2_mean),
        "predictor_p90": TwoStagePredictor(stage1, stage2_p90),
    }


# Position name -> FPL element_type id (matches rules.POSITIONS keys)
_POS_NAME_TO_ID: dict[str, int] = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}


def build_players_for_gw(
    features_df: pd.DataFrame,
    predictions: pd.Series,
    season: str,
    gw: int,
    *,
    ceiling_predictions: pd.Series | None = None,
    horizon_xpts: dict[int, float] | None = None,
) -> list:
    """Bridge model output -> optimizer input.

    Aggregates predictions into Player objects for one (season, gw):
      - Sums xpts across multiple fixtures (DGW players play twice in one GW)
      - Assigns a stable per-season int team_id (optimizer's max-3-per-club rule)
      - Drops players with no prediction (no prior-GW history)
      - If `ceiling_predictions` is given (P90 model), sets player.ceiling_xpts
        so the optimizer can use it for captain selection.
      - If `horizon_xpts` is given (player_id -> summed future-GW xpts), sets
        player.horizon_xpts so the optimizer can favour good fixture runs.
    """
    from .data import Player  # local import — Player lives in data.py

    rows = features_df[
        (features_df["season"] == season) & (features_df["GW"] == gw)
    ]
    rows = rows.loc[rows.index.intersection(predictions.index)]
    if rows.empty:
        return []

    rows = rows.assign(xpts=predictions.loc[rows.index])
    if ceiling_predictions is not None:
        rows = rows.assign(
            ceiling=ceiling_predictions.reindex(rows.index).fillna(rows["xpts"])
        )
    else:
        rows = rows.assign(ceiling=rows["xpts"])

    team_ids = {t: i for i, t in enumerate(sorted(rows["team"].dropna().unique()))}

    agg = rows.groupby("element", as_index=False).agg(
        name=("name", "first"),
        team=("team", "first"),
        position=("position", "first"),
        value=("value", "first"),
        xpts=("xpts", "sum"),
        ceiling=("ceiling", "sum"),
    )

    return [
        Player(
            id=int(r["element"]),
            name=str(r["name"]),
            team_id=team_ids[r["team"]],
            team_name=str(r["team"]),
            position=_POS_NAME_TO_ID[str(r["position"])],
            price=int(r["value"]),
            xpts=float(r["xpts"]),
            ceiling_xpts=float(r["ceiling"]),
            horizon_xpts=float(horizon_xpts.get(int(r["element"]), 0.0))
                          if horizon_xpts else 0.0,
        )
        for _, r in agg.iterrows()
    ]
