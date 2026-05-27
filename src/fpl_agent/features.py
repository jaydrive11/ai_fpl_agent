"""Feature engineering for the xPts prediction model.

Target: `total_points` scored in a given gameweek.
All historical features are lagged via `.shift(1)` to prevent leakage from
the gameweek we're trying to predict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Categoricals — passed to LightGBM as categorical_feature so it can split on them
# directly (no manual one-hot).
CATEGORICAL_FEATURES: list[str] = ["position", "team", "opponent_team", "was_home"]

# Direct features (known before kickoff)
DIRECT_NUMERIC_FEATURES: list[str] = [
    "value", "GW",
    # Fixture-difficulty (home/away-aware). NaN for seasons lacking teams.csv;
    # LightGBM handles NaN as a separate split direction.
    "opp_attack", "opp_defence", "team_attack", "team_defence",
]

# Rolling-window source columns -> only the 5-window form is kept for these
ROLLING_5_COLS: list[str] = [
    "starts",
    "goals_scored", "assists", "bonus",
    "bps", "ict_index",
    "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded",
    "saves",
]

# Minutes and total_points get richer windows (3, 5, 10)
WIDE_ROLLING_COLS: list[str] = ["minutes", "total_points"]
ROLLING_WINDOWS: tuple[int, ...] = (3, 5, 10)

# Single-step lag columns
LAG_COLS: list[str] = ["minutes", "total_points", "value"]

TARGET: str = "total_points"


def _rolling_feature_names() -> list[str]:
    names: list[str] = []
    for col in WIDE_ROLLING_COLS:
        for w in ROLLING_WINDOWS:
            names.append(f"{col}_avg{w}")
    for col in ROLLING_5_COLS:
        names.append(f"{col}_avg5")
    return names


def _lag_feature_names() -> list[str]:
    return [f"{col}_lag1" for col in LAG_COLS]


FEATURES: list[str] = (
    CATEGORICAL_FEATURES
    + DIRECT_NUMERIC_FEATURES
    + _rolling_feature_names()
    + _lag_feature_names()
)


def normalize_position(df: pd.DataFrame) -> pd.DataFrame:
    """Map GKP->GK; drop Assistant Manager (AM) rows (a chip, not a real player)."""
    df = df.copy()
    df["position"] = df["position"].astype(str).replace({"GKP": "GK"})
    df = df[df["position"].isin(["GK", "DEF", "MID", "FWD"])].reset_index(drop=True)
    return df


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Add any missing columns as all-NaN (older seasons lack xG fields)."""
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _join_fixture_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Add opp_attack, opp_defence, team_attack, team_defence features.

    Joins teams.csv strength ratings via (season, opponent_team id) and
    (season, own team name). Picks the home/away strength based on which side
    of the fixture I'm on. NaN for seasons without teams.csv.
    """
    from .history import load_team_strengths  # avoid circular at import

    seasons = df["season"].unique()
    parts = []
    for s in seasons:
        ts = load_team_strengths(s)
        if ts is not None:
            ts = ts.assign(season=s)
            parts.append(ts)

    out_cols = ["opp_attack", "opp_defence", "team_attack", "team_defence"]
    if not parts:
        return df.assign(**{c: np.nan for c in out_cols})

    teams_df = pd.concat(parts, ignore_index=True)

    opp_lookup = teams_df.rename(columns={
        "id": "opponent_team",
        "attack_h": "_opp_atk_h", "attack_a": "_opp_atk_a",
        "defence_h": "_opp_def_h", "defence_a": "_opp_def_a",
    })[["season", "opponent_team", "_opp_atk_h", "_opp_atk_a",
        "_opp_def_h", "_opp_def_a"]]

    own_lookup = teams_df.rename(columns={
        "name": "team",
        "attack_h": "_own_atk_h", "attack_a": "_own_atk_a",
        "defence_h": "_own_def_h", "defence_a": "_own_def_a",
    })[["season", "team", "_own_atk_h", "_own_atk_a",
        "_own_def_h", "_own_def_a"]]

    df = df.merge(opp_lookup, on=["season", "opponent_team"], how="left")
    df = df.merge(own_lookup, on=["season", "team"], how="left")

    # was_home: True means I'm at home → opponent is away. Pick the opponent's
    # *away* rating; my own *home* rating. And vice versa.
    home = df["was_home"].astype("boolean").fillna(False).to_numpy()
    df["opp_attack"]   = np.where(home, df["_opp_atk_a"], df["_opp_atk_h"])
    df["opp_defence"]  = np.where(home, df["_opp_def_a"], df["_opp_def_h"])
    df["team_attack"]  = np.where(home, df["_own_atk_h"], df["_own_atk_a"])
    df["team_defence"] = np.where(home, df["_own_def_h"], df["_own_def_a"])

    return df.drop(columns=[c for c in df.columns if c.startswith("_opp_")
                            or c.startswith("_own_")])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute lag and rolling features per (season, player). Sorted by GW.

    Implementation: pre-shift each source column with the fast vectorized
    `groupby().shift(1)`, then use `groupby().rolling().mean()` which is
    Cython-fast — avoids per-group Python lambda overhead.
    Crucially the shift(1) means features at GW N never observe GW N — no leakage.
    """
    df = normalize_position(df)
    df = _join_fixture_difficulty(df)
    df = _ensure_columns(df, WIDE_ROLLING_COLS + ROLLING_5_COLS + LAG_COLS)
    df = df.sort_values(["season", "name", "GW"]).reset_index(drop=True)

    g = df.groupby(["season", "name"], sort=False, observed=True)
    src_cols = WIDE_ROLLING_COLS + ROLLING_5_COLS

    # Pre-shift every source column once. Returns a DataFrame aligned with df.
    shifted = pd.DataFrame({c: g[c].shift(1) for c in src_cols}, index=df.index)
    # Re-attach group keys so we can groupby+rolling on the shifted frame.
    shifted["_s"] = df["season"].values
    shifted["_n"] = df["name"].values
    g_shifted = shifted.groupby(["_s", "_n"], sort=False, observed=True)

    def _rolling(cols: list[str], window: int, suffix: str) -> None:
        rolled = g_shifted[cols].rolling(window, min_periods=1).mean()
        # Drop the two group levels of the MultiIndex, keep the original int index,
        # then reindex back to df's order.
        rolled.index = rolled.index.get_level_values(-1)
        rolled = rolled.reindex(df.index)
        for col in cols:
            df[f"{col}_avg{suffix}"] = rolled[col].values

    for w in ROLLING_WINDOWS:
        _rolling(WIDE_ROLLING_COLS, w, str(w))
    _rolling(ROLLING_5_COLS, 5, "5")

    for col in LAG_COLS:
        df[f"{col}_lag1"] = g[col].shift(1)

    return df


def split_xy(df: pd.DataFrame, *, target_col: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    """Pull (X, y) for training/eval. Drops rows with no prior history."""
    # Need at least 1 prior GW so lag features are populated
    df = df[df["minutes_lag1"].notna()].copy()
    X = df[FEATURES].copy()
    # Cast categoricals to pandas 'category' dtype so LightGBM picks them up
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")
    y = df[target_col].astype(float)
    return X, y
