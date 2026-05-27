"""End-to-end test: features -> model -> Player objects -> optimizer."""
import numpy as np
import pandas as pd

from fpl_agent.features import build_features
from fpl_agent.model import build_players_for_gw, predict, train
from fpl_agent.optimizer import pick_squad


def _synthetic_raw_history() -> pd.DataFrame:
    """Generate a multi-season per-GW history just rich enough to train a tiny
    model and exercise the squad-picking pipeline. ~6 teams, 4 players per team
    per position, 6 GWs per season, 3 seasons."""
    rng = np.random.default_rng(42)
    rows = []
    seasons = ["2022-23", "2023-24", "2024-25"]
    positions = ["GK", "DEF", "DEF", "DEF", "DEF", "DEF",
                 "MID", "MID", "MID", "MID", "MID",
                 "FWD", "FWD", "FWD"]
    n_teams = 6

    pid = 0
    for season in seasons:
        for team_id in range(1, n_teams + 1):
            for pos_idx, pos in enumerate(positions):
                pid += 1
                base = rng.uniform(2.0, 6.0)  # player's true skill
                price = 40 + int(base * 8 + rng.normal(0, 5))
                for gw in range(1, 7):
                    mins = int(np.clip(rng.normal(75, 25), 0, 90))
                    pts = max(0, int(rng.normal(base * (mins / 90), 2)))
                    rows.append({
                        "season": season,
                        "name": f"P{pid}",
                        "team": f"T{team_id}",
                        "opponent_team": (team_id % n_teams) + 1,
                        "was_home": gw % 2 == 0,
                        "position": pos,
                        "value": price,
                        "element": pid,
                        "GW": gw,
                        "minutes": mins,
                        "total_points": pts,
                        "starts": int(mins >= 60),
                        "goals_scored": 0, "assists": 0,
                        "bps": pts * 3, "ict_index": float(pts) * 1.5,
                        "bonus": 0,
                        "expected_goals": rng.uniform(0, 0.5),
                        "expected_assists": rng.uniform(0, 0.3),
                        "expected_goals_conceded": rng.uniform(0.5, 2.0),
                        "expected_goal_involvements": rng.uniform(0, 0.6),
                        "saves": rng.integers(0, 4) if pos == "GK" else 0,
                    })
    return pd.DataFrame(rows)


def test_pipeline_end_to_end_produces_valid_squad():
    raw = _synthetic_raw_history()
    feats = build_features(raw)

    # Train a tiny model (few rounds is enough — we only need plausible preds)
    booster, _ = train(
        feats,
        train_until="2023-24",
        val_season="2024-25",
        num_rounds=50,
        params={
            "objective": "regression_l1", "metric": "mae",
            "learning_rate": 0.1, "num_leaves": 15, "min_data_in_leaf": 5,
            "verbose": -1, "num_threads": 2,
        },
    )

    preds = predict(booster, feats)

    target_season, target_gw = "2024-25", 6
    players = build_players_for_gw(feats, preds, target_season, target_gw)

    # Must have enough players in each position to satisfy the squad
    pos_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for p in players:
        pos_counts[p.position] += 1
    assert pos_counts[1] >= 2  # GK
    assert pos_counts[2] >= 5  # DEF
    assert pos_counts[3] >= 5  # MID
    assert pos_counts[4] >= 3  # FWD

    sel = pick_squad(players)

    assert len(sel.squad) == 15
    assert len(sel.starting_xi) == 11
    assert sel.captain in sel.starting_xi
    assert sel.cost <= 1000
    # Captain should be among the highest-predicted players
    top_xpts = max(p.xpts for p in players)
    assert sel.captain.xpts >= 0.9 * top_xpts


def test_build_players_for_gw_sums_double_gameweek_xpts():
    """When a player appears twice in one GW (DGW), their xpts should sum."""
    feats = pd.DataFrame({
        "season": ["2024-25"] * 3,
        "GW": [5, 5, 5],
        "element": [100, 100, 200],   # player 100 has 2 fixtures (DGW)
        "name": ["A", "A", "B"],
        "team": ["X", "X", "Y"],
        "position": ["MID", "MID", "FWD"],
        "value": [60, 60, 80],
    })
    preds = pd.Series([3.5, 2.0, 5.0], index=feats.index)

    players = build_players_for_gw(feats, preds, "2024-25", 5)
    by_id = {p.id: p for p in players}
    assert by_id[100].xpts == 5.5  # 3.5 + 2.0 summed across DGW fixtures
    assert by_id[200].xpts == 5.0
