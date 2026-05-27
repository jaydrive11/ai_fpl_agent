"""Tests for chip scoring + decision logic."""
import numpy as np

from fpl_agent.backtest import ChipsState, _decide_chip, _score_xi
from fpl_agent.data import Player
from fpl_agent.optimizer import Selection, pick_squad
from fpl_agent.rules import POSITIONS


def _make_pool() -> list[Player]:
    rng = np.random.default_rng(0)
    players = []
    pid = 0
    for team_id in range(1, 7):
        for pos_id in POSITIONS:
            for _ in range(4):
                pid += 1
                players.append(Player(
                    id=pid, name=f"P{pid}",
                    team_id=team_id, team_name=f"T{team_id}",
                    position=pos_id, price=40 + int(rng.uniform(0, 80)),
                    xpts=float(rng.uniform(0, 8)),
                    ceiling_xpts=float(rng.uniform(2, 15)),
                ))
    return players


def test_tc_doubles_captain_extra():
    actuals = {1: 10, 2: 5, 3: 7}
    xi = [Player(1, "A", 1, "T1", 3, 50, 5.0), Player(2, "B", 2, "T2", 3, 50, 5.0)]
    cap = xi[0]
    base = _score_xi(xi, cap, actuals)              # 10+5 + 10  = 25
    tc = _score_xi(xi, cap, actuals, chip="TC")     # 10+5 + 20  = 35
    assert base == 25.0
    assert tc == 35.0


def test_bb_adds_bench_points():
    actuals = {1: 10, 2: 5, 3: 7, 4: 3}
    xi = [Player(1, "A", 1, "T1", 3, 50, 5.0)]
    cap = xi[0]
    bench = [Player(2, "B", 2, "T2", 3, 50, 0.0),
             Player(3, "C", 3, "T3", 3, 50, 0.0),
             Player(4, "D", 4, "T4", 3, 50, 0.0)]
    base = _score_xi(xi, cap, actuals)                          # 10 + 10 = 20
    bb = _score_xi(xi, cap, actuals, bench=bench, chip="BB")    # +5+7+3 = 35
    assert base == 20.0
    assert bb == 35.0


def test_wc_zeroes_hit_cost():
    actuals = {1: 10}
    xi = [Player(1, "A", 1, "T1", 3, 50, 5.0)]
    cap = xi[0]
    with_hits = _score_xi(xi, cap, actuals, hits=2)              # 20 - 8 = 12
    wc = _score_xi(xi, cap, actuals, hits=2, chip="WC")          # 20 (hits zeroed)
    assert with_hits == 12.0
    assert wc == 20.0


def test_decide_chip_plays_wc_on_big_gain():
    chips = ChipsState()
    cap = Player(1, "Cap", 1, "T1", 3, 100, 4.0, ceiling_xpts=5.0)
    sel_constrained = Selection(squad=[], starting_xi=[], captain=cap, bench=[],
                                 expected_points=30.0, cost=500)
    sel_wc = Selection(squad=[], starting_xi=[], captain=cap, bench=[],
                       expected_points=50.0, cost=500)

    chip = _decide_chip(sel_constrained, sel_wc, chips, wc_gain_threshold=10.0)
    assert chip == "WC"
    assert chips.wc_used


def test_decide_chip_plays_tc_on_high_ceiling():
    chips = ChipsState()
    # Make a fake selection where the captain has a sky-high ceiling
    star = Player(1, "Star", 1, "T1", 3, 100, 8.0, ceiling_xpts=20.0)
    others = [Player(i, f"P{i}", 1, "T1", 3, 40, 1.0) for i in range(2, 12)]
    sel = Selection(
        squad=[star] + others, starting_xi=[star] + others[:10],
        captain=star, bench=[], expected_points=20.0, cost=500,
    )
    chip = _decide_chip(sel, None, chips,
                        wc_gain_threshold=100.0, tc_ceiling_threshold=15.0)
    assert chip == "TC"
    assert chips.tc_used


def test_decide_chip_respects_one_chip_per_gw():
    """If WC fires, TC/BB are not also played in the same GW."""
    chips = ChipsState()
    sel_constrained = Selection(
        squad=[], starting_xi=[],
        captain=Player(1, "C", 1, "T1", 3, 100, 8.0, ceiling_xpts=20.0),
        bench=[Player(99, "B", 1, "T1", 3, 40, 8.0)] * 4,  # bench predicted = 32
        expected_points=30.0, cost=500,
    )
    sel_wc = Selection(squad=[], starting_xi=[], captain=sel_constrained.captain,
                       bench=[], expected_points=50.0, cost=500)
    chip = _decide_chip(sel_constrained, sel_wc, chips,
                        wc_gain_threshold=10.0, tc_ceiling_threshold=15.0,
                        bb_bench_threshold=10.0)
    assert chip == "WC"
    # The TC and BB conditions are met, but only one chip per GW — they're untouched
    assert not chips.tc_used
    assert not chips.bb_used
