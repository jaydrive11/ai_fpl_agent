import numpy as np
import pandas as pd

from fpl_agent.features import (
    FEATURES,
    build_features,
    normalize_position,
    split_xy,
)


def test_normalize_position_merges_gkp_and_drops_am():
    df = pd.DataFrame({"position": ["GK", "GKP", "DEF", "MID", "FWD", "AM"]})
    out = normalize_position(df)
    assert set(out["position"]) == {"GK", "DEF", "MID", "FWD"}
    assert (out["position"] == "GK").sum() == 2  # GK + (GKP -> GK)
    assert (out["position"] == "AM").sum() == 0


def _toy_player_season() -> pd.DataFrame:
    return pd.DataFrame({
        "season": ["2024-25"] * 6,
        "name": ["X"] * 6,
        "team": ["A"] * 6,
        "opponent_team": [2, 3, 4, 5, 6, 7],
        "was_home": [True, False, True, False, True, False],
        "position": ["MID"] * 6,
        "value": [50, 50, 50, 51, 51, 52],
        "GW": [1, 2, 3, 4, 5, 6],
        "minutes": [90, 60, 0, 90, 45, 90],
        "total_points": [2, 5, 0, 8, 3, 6],
        "starts": [1, 1, 0, 1, 1, 1],
        "goals_scored": [0, 1, 0, 1, 0, 1],
        "assists": [0, 0, 0, 0, 1, 0],
        "bps": [10, 25, 0, 35, 15, 28],
        "ict_index": [5.0, 8.0, 0.0, 9.0, 6.0, 7.5],
        "bonus": [0, 1, 0, 2, 0, 1],
        "expected_goals": [0.1, 0.5, 0.0, 0.7, 0.2, 0.6],
        "expected_assists": [0.1, 0.2, 0.0, 0.3, 0.4, 0.2],
        "expected_goals_conceded": [1.0, 1.5, 0.0, 1.1, 1.4, 1.2],
        "expected_goal_involvements": [0.2, 0.7, 0.0, 1.0, 0.6, 0.8],
        "saves": [0, 0, 0, 0, 0, 0],
    })


def test_rolling_features_use_prior_gws_only():
    """The whole point of shift(1) — features at GW N can't see GW N."""
    out = build_features(_toy_player_season()).sort_values("GW").reset_index(drop=True)

    # GW1: nothing prior → all rolling/lag features are NaN
    gw1 = out.iloc[0]
    assert np.isnan(gw1["minutes_avg5"])
    assert np.isnan(gw1["minutes_lag1"])

    # GW2: only GW1 in window → avg = GW1's value
    gw2 = out.iloc[1]
    assert gw2["minutes_avg5"] == 90.0
    assert gw2["total_points_avg5"] == 2.0
    assert gw2["minutes_lag1"] == 90.0

    # GW3: avg over GW1+GW2
    gw3 = out.iloc[2]
    assert gw3["minutes_avg5"] == (90 + 60) / 2
    assert gw3["total_points_avg5"] == (2 + 5) / 2

    # GW6: 5-window includes GW1..GW5; 3-window includes GW3..GW5
    gw6 = out.iloc[5]
    assert gw6["minutes_avg5"] == (90 + 60 + 0 + 90 + 45) / 5
    assert gw6["minutes_avg3"] == (0 + 90 + 45) / 3


def test_rolling_features_do_not_cross_seasons():
    """A new season must reset rolling windows — no leakage from last season."""
    df = pd.DataFrame({
        "season": ["2023-24", "2023-24", "2024-25", "2024-25"],
        "name": ["X"] * 4,
        "team": ["A"] * 4,
        "opponent_team": [2, 3, 2, 3],
        "was_home": [True, False, True, False],
        "position": ["MID"] * 4,
        "value": [50] * 4,
        "GW": [37, 38, 1, 2],
        "minutes": [90, 90, 0, 0],
        "total_points": [10, 10, 0, 0],
        "starts": [1, 1, 0, 0],
        "goals_scored": [0] * 4, "assists": [0] * 4,
        "bps": [20] * 4, "ict_index": [5.0] * 4, "bonus": [0] * 4,
        "expected_goals": [0.3] * 4, "expected_assists": [0.1] * 4,
        "expected_goals_conceded": [1.0] * 4,
        "expected_goal_involvements": [0.4] * 4, "saves": [0] * 4,
    })
    out = build_features(df).sort_values(["season", "GW"]).reset_index(drop=True)
    # 2024-25 GW1 must have no prior history despite 2023-24 GW38 existing
    new_season_gw1 = out[(out.season == "2024-25") & (out.GW == 1)].iloc[0]
    assert np.isnan(new_season_gw1["minutes_lag1"])
    assert np.isnan(new_season_gw1["minutes_avg5"])


def test_split_xy_outputs_consistent_feature_columns():
    out = build_features(_toy_player_season())
    X, y = split_xy(out)
    assert list(X.columns) == FEATURES
    assert len(X) == len(y)
    # Categoricals are pandas categorical dtype
    assert str(X["position"].dtype) == "category"
    assert str(X["team"].dtype) == "category"
