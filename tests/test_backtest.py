"""Tests for the transfer-aware optimiser and backtest engine."""
import numpy as np
import pandas as pd

from fpl_agent.data import Player
from fpl_agent.optimizer import pick_squad, pick_xi
from fpl_agent.rules import POSITIONS


def _make_player_pool(n_per_pos_per_club: int = 4, n_clubs: int = 6) -> list[Player]:
    rng = np.random.default_rng(0)
    players: list[Player] = []
    pid = 0
    for team_id in range(1, n_clubs + 1):
        for pos_id in POSITIONS:
            for _ in range(n_per_pos_per_club):
                pid += 1
                players.append(Player(
                    id=pid,
                    name=f"P{pid}",
                    team_id=team_id,
                    team_name=f"T{team_id}",
                    position=pos_id,
                    price=40 + int(rng.uniform(0, 80)),
                    xpts=float(rng.uniform(0, 8)),
                ))
    return players


def test_pick_squad_with_current_squad_no_changes_when_free_transfers_high():
    """If existing squad is already optimal and we have many free transfers,
    no transfers should be made."""
    players = _make_player_pool()
    sel0 = pick_squad(players)  # fresh pick = optimal
    sel1 = pick_squad(
        players,
        current_squad_ids={p.id for p in sel0.squad},
        free_transfers=15,
        hit_cost=4.0,
    )
    assert len(sel1.transfers_in) == 0
    assert sel1.hits == 0


def _build_valid_suboptimal_squad(players: list[Player]) -> list[Player]:
    """Pick the best squad from only the bottom half of players (by xpts).
    The result is a *valid* FPL squad (budget, positions, per-club all OK)
    but with weak xpts — so the full-pool optimiser will want to upgrade it.
    """
    weak_pool = sorted(players, key=lambda p: p.xpts)[: len(players) // 2]
    sel = pick_squad(weak_pool)
    return sel.squad


def test_pick_squad_applies_hit_for_extra_transfers():
    """Starting from a valid suboptimal squad, with hit_cost=4 the optimiser
    will make some transfers — the upgrades are worth the hits."""
    players = _make_player_pool()
    sub_squad = _build_valid_suboptimal_squad(players)
    sub_ids = {p.id for p in sub_squad}

    sel = pick_squad(players, current_squad_ids=sub_ids,
                     free_transfers=1, hit_cost=4.0)
    # At least one transfer (1 is free anyway)
    assert len(sel.transfers_in) >= 1
    # Hits = transfers beyond free
    assert sel.hits == max(0, len(sel.transfers_in) - 1)
    # transfers_out_ids must equal members no longer in the squad
    new_ids = {p.id for p in sel.squad}
    assert set(sel.transfers_out_ids) == sub_ids - new_ids


def test_higher_hit_cost_reduces_or_equals_transfers():
    """Monotone: hit_cost↑ ⇒ transfers↓ (or stay equal) for the same start."""
    players = _make_player_pool()
    sub_squad = _build_valid_suboptimal_squad(players)
    sub_ids = {p.id for p in sub_squad}

    n_free = pick_squad(players, current_squad_ids=sub_ids,
                        free_transfers=0, hit_cost=0.0).transfers_in
    n_cheap = pick_squad(players, current_squad_ids=sub_ids,
                         free_transfers=0, hit_cost=4.0).transfers_in
    n_dear = pick_squad(players, current_squad_ids=sub_ids,
                        free_transfers=0, hit_cost=1000.0).transfers_in

    assert len(n_free) >= len(n_cheap) >= len(n_dear)
    # With a prohibitive hit cost the optimiser should keep the squad as-is
    assert len(n_dear) == 0


def test_expected_points_is_net_of_hits():
    """Selection.expected_points should equal XI+captain xpts minus hits*hit_cost."""
    players = _make_player_pool()
    sub_squad = _build_valid_suboptimal_squad(players)
    sel = pick_squad(players, current_squad_ids={p.id for p in sub_squad},
                     free_transfers=0, hit_cost=4.0)
    base = sum(p.xpts for p in sel.starting_xi) + sel.captain.xpts
    assert abs(sel.expected_points - (base - 4.0 * sel.hits)) < 1e-6


def test_pick_xi_respects_formation_bounds():
    players = _make_player_pool()
    sel = pick_squad(players)
    xi, cap, bench = pick_xi(sel.squad)

    assert len(xi) == 11
    assert len(bench) == 4
    assert cap in xi
    pos_counts = {pid: 0 for pid in POSITIONS}
    for p in xi:
        pos_counts[p.position] += 1
    for pos_id, (_, _, smin, smax) in POSITIONS.items():
        assert smin <= pos_counts[pos_id] <= smax


def test_backtest_runs_end_to_end_on_synthetic_data():
    """Tiny end-to-end run: features -> train -> backtest -> per-GW scores."""
    from fpl_agent.backtest import run_backtest
    from fpl_agent.features import build_features
    from fpl_agent.model import train

    rng = np.random.default_rng(7)
    seasons = ["2023-24", "2024-25"]
    positions_pool = (["GK"] + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3) * 2  # 28 per club
    rows = []
    pid = 0
    for season in seasons:
        for team in range(1, 7):
            for pos in positions_pool:
                pid += 1
                skill = rng.uniform(1, 6)
                price = 40 + int(skill * 8 + rng.normal(0, 4))
                for gw in range(1, 8):
                    mins = int(np.clip(rng.normal(70, 30), 0, 90))
                    pts = max(0, int(rng.normal(skill * (mins / 90), 2)))
                    rows.append({
                        "season": season, "name": f"P{pid}", "team": f"T{team}",
                        "opponent_team": team % 6 + 1, "was_home": gw % 2 == 0,
                        "position": pos, "value": price, "element": pid, "GW": gw,
                        "minutes": mins, "total_points": pts,
                        "starts": int(mins >= 60),
                        "goals_scored": 0, "assists": 0, "bps": pts * 3,
                        "ict_index": float(pts * 1.5), "bonus": 0,
                        "expected_goals": rng.uniform(0, 0.5),
                        "expected_assists": rng.uniform(0, 0.3),
                        "expected_goals_conceded": rng.uniform(0.5, 2.0),
                        "expected_goal_involvements": rng.uniform(0, 0.6),
                        "saves": rng.integers(0, 4) if pos == "GK" else 0,
                    })
    df = pd.DataFrame(rows)
    feats = build_features(df)
    booster, _ = train(
        feats, train_until="2023-24", val_season="2024-25", num_rounds=30,
        params={"objective": "regression_l1", "metric": "mae", "learning_rate": 0.1,
                "num_leaves": 15, "min_data_in_leaf": 5, "verbose": -1, "num_threads": 2},
    )

    rep = run_backtest(booster, df, "2024-25", start_gw=2, end_gw=7)

    assert len(rep.gws) > 0
    assert len(rep.agent.per_gw) == len(rep.gws)
    assert len(rep.static.per_gw) == len(rep.gws)
    assert len(rep.oracle.per_gw) == len(rep.gws)

    # Oracle should always be >= agent (it has perfect foresight, no constraints)
    assert rep.oracle.total >= rep.agent.total
    # All per-GW scores should be finite non-negative
    for r in rep.agent.per_gw:
        assert r.actual_points >= -100  # bounded; hits could push slightly negative
        assert r.hits >= 0
