"""Tests for FPL auto-sub scoring logic."""
from fpl_agent.backtest import (
    _apply_auto_subs, _formation_valid, _ordered_bench, _score_xi,
)
from fpl_agent.data import Player


def _mk(id_, pos, xpts=1.0):
    """Shorthand for a player. pos: 1=GK, 2=DEF, 3=MID, 4=FWD."""
    return Player(id=id_, name=f"P{id_}", team_id=1, team_name="T1",
                  position=pos, price=50, xpts=xpts)


def test_formation_valid_accepts_standard_formations():
    # 4-4-2
    xi = [_mk(1, 1)] + [_mk(i, 2) for i in range(2, 6)] + \
         [_mk(i, 3) for i in range(6, 10)] + [_mk(i, 4) for i in range(10, 12)]
    assert _formation_valid(xi)
    # 3-5-2
    xi = [_mk(1, 1)] + [_mk(i, 2) for i in range(2, 5)] + \
         [_mk(i, 3) for i in range(5, 10)] + [_mk(i, 4) for i in range(10, 12)]
    assert _formation_valid(xi)


def test_formation_valid_rejects_bad_formations():
    # 0 GK
    xi = [_mk(i, 2) for i in range(1, 6)] + [_mk(i, 3) for i in range(6, 10)] + \
         [_mk(i, 4) for i in range(10, 12)]
    assert not _formation_valid(xi)
    # Too few DEF (2)
    xi = [_mk(1, 1)] + [_mk(i, 2) for i in range(2, 4)] + \
         [_mk(i, 3) for i in range(4, 9)] + [_mk(i, 4) for i in range(9, 12)]
    assert not _formation_valid(xi)


def test_ordered_bench_gk_first_then_outfield_by_xpts():
    bench = [_mk(1, 3, xpts=2.0), _mk(2, 1, xpts=0.5),
             _mk(3, 4, xpts=4.0), _mk(4, 2, xpts=1.0)]
    ordered = _ordered_bench(bench)
    assert ordered[0].position == 1  # GK first
    # Remaining 3 outfield in xpts-desc order
    rest = ordered[1:]
    assert [p.xpts for p in rest] == [4.0, 2.0, 1.0]


def test_auto_sub_swaps_gk_for_bench_gk_when_starter_blanks():
    starter_gk = _mk(1, 1, xpts=4.0)
    xi = [starter_gk] + [_mk(i, 2) for i in range(2, 6)] + \
         [_mk(i, 3) for i in range(6, 10)] + [_mk(i, 4) for i in range(10, 12)]
    bench_gk = _mk(99, 1, xpts=2.0)
    bench = [bench_gk, _mk(100, 3), _mk(101, 4), _mk(102, 2)]
    # Starter GK got 0 min, bench GK played
    mins = {1: 0, 99: 90, **{i: 90 for i in range(2, 12)},
            100: 0, 101: 0, 102: 0}
    eff = _apply_auto_subs(xi, bench, mins)
    assert any(p.id == 99 for p in eff)
    assert not any(p.id == 1 for p in eff)


def test_auto_sub_outfield_keeps_formation_valid():
    """If subbing in a FWD for a blanked DEF would break formation
    (e.g. would leave 2 DEF), the sub shouldn't happen for that bench player."""
    # XI: 3-4-3 → 1 GK, 3 DEF, 4 MID, 3 FWD
    xi = [_mk(1, 1)] + [_mk(i, 2) for i in range(2, 5)] + \
         [_mk(i, 3) for i in range(5, 9)] + [_mk(i, 4) for i in range(9, 12)]
    # Bench order (by xpts): MID > FWD > DEF (one of each on bench)
    bench = [_mk(50, 1, xpts=1.0),       # bench GK
             _mk(51, 4, xpts=5.0),       # FWD high
             _mk(52, 3, xpts=4.0),       # MID
             _mk(53, 2, xpts=3.0)]       # DEF
    # One DEF blanks (id=2)
    mins = {1: 90, 2: 0, **{i: 90 for i in range(3, 12)},
            50: 0, 51: 90, 52: 90, 53: 90}
    eff = _apply_auto_subs(xi, bench, mins)
    # The blanked DEF (id=2) should be replaced. Highest-priority outfield bench
    # is FWD (id 51) — would make 2-4-4, invalid (needs 3+ DEF). Skip to MID 52
    # — 2-5-3 also invalid (still 2 DEF). Only DEF (53) keeps formation valid.
    assert not any(p.id == 2 for p in eff), "blanked DEF should be replaced"
    # The replacing player must keep formation valid (must be a DEF here)
    pos_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for p in eff:
        pos_counts[p.position] += 1
    assert pos_counts == {1: 1, 2: 3, 3: 4, 4: 3}, \
        f"formation should still be 3-4-3, got {pos_counts}"


def test_auto_sub_skips_bench_who_also_blanked():
    """Bench players who got 0 minutes can't come on as subs."""
    xi = [_mk(1, 1)] + [_mk(i, 2) for i in range(2, 6)] + \
         [_mk(i, 3) for i in range(6, 10)] + [_mk(i, 4) for i in range(10, 12)]
    bench = [_mk(50, 1), _mk(51, 4, xpts=5.0), _mk(52, 3, xpts=4.0), _mk(53, 2, xpts=3.0)]
    # DEF id=2 blanks; bench FWD id=51 ALSO blanks; MID 52 plays
    mins = {1: 90, 2: 0, 3: 90, 4: 90, 5: 90, 6: 90, 7: 90, 8: 90, 9: 90,
            10: 90, 11: 90, 50: 0, 51: 0, 52: 90, 53: 0}
    eff = _apply_auto_subs(xi, bench, mins)
    # FWD bench can't sub (didn't play); MID bench can but would be 4-3-3
    # wait that's also invalid because it leaves 3 DEF... actually wait.
    # 4-4-3 minus DEF + MID = 3-5-3, total 11. Counts: 1+3+5+3 = 12, NO.
    # Hmm let me recount. Starting XI: 1 GK + 4 DEF + 4 MID + 3 FWD = 12. Wrong.
    # The fixture above has 4 DEF (ids 2-5), 4 MID (ids 6-9), 3 FWD (ids 10,11).
    # Wait — range(6, 10) is 4 elements, range(10, 12) is 2 elements.
    # So: 1 GK + 4 DEF + 4 MID + 2 FWD = 11. That's 4-4-2.
    # Blanking DEF id=2 leaves 3 DEF, valid. Subbing in MID 52 gives 3 DEF, 5 MID,
    # 2 FWD = 10 ... wait. Hmm 1+3+5+2 = 11. So formation 3-5-2 — valid.
    assert not any(p.id == 2 for p in eff)
    assert any(p.id == 52 for p in eff)


def test_score_xi_with_auto_subs_recovers_lost_points():
    starter = _mk(10, 4, xpts=5.0)
    other_xi = [_mk(1, 1)] + [_mk(i, 2) for i in range(2, 6)] + \
               [_mk(i, 3) for i in range(6, 10)] + [_mk(11, 4)]
    xi = other_xi + [starter]   # 1 GK + 4 DEF + 4 MID + 2 FWD
    captain = xi[0]
    # Bench has a high-xpts forward
    bench = [_mk(50, 1), _mk(51, 4, xpts=8.0), _mk(52, 3, xpts=3.0), _mk(53, 2, xpts=2.0)]

    actuals = {10: 0, 51: 9}  # starter blanked, bench FWD scored 9
    actuals.update({i: 1 for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]})
    mins = {10: 0, 51: 90, **{i: 90 for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]},
            50: 0, 52: 0, 53: 0}

    # Without auto-sub knowledge: starter 10 scores 0 → XI total = 10*1 + 0 = 10
    naive = _score_xi(xi, captain, actuals, bench=bench)
    # With auto-sub: starter 10 (0 min) replaced by bench FWD 51 (9 pts)
    with_sub = _score_xi(xi, captain, actuals, bench=bench, actuals_min=mins)
    assert with_sub > naive
    # Specifically: XI = 10*1 + 9 (sub) = 19; captain (id=1) bonus = 1; total = 20
    assert with_sub == 19 + 1


def test_score_xi_no_subs_when_bench_boost_active():
    """BB plays all 15 — auto-sub doesn't apply (would double-count)."""
    starter = _mk(10, 4, xpts=5.0)
    xi = [_mk(1, 1)] + [_mk(i, 2) for i in range(2, 6)] + \
         [_mk(i, 3) for i in range(6, 10)] + [_mk(11, 4)] + [starter]
    captain = xi[0]
    bench = [_mk(50, 1), _mk(51, 4, xpts=8.0), _mk(52, 3, xpts=3.0), _mk(53, 2, xpts=2.0)]
    actuals = {10: 0, 51: 9}
    actuals.update({i: 1 for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]})
    actuals.update({50: 2, 52: 1, 53: 1})
    mins = {10: 0, 51: 90}
    mins.update({i: 90 for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]})
    mins.update({50: 90, 52: 90, 53: 90})

    bb_score = _score_xi(xi, captain, actuals, bench=bench,
                          chip="BB", actuals_min=mins)
    # Without BB: XI = 10 + 0 = 10. With auto-sub: 10 + 9 (51) = 19. cap bonus=1. Total 20.
    # With BB: XI = 10 + 0 = 10. Bench (all 4) = 2+9+1+1 = 13. cap bonus=1. Total 24.
    # Important: bench FWD 51 contributes via the BB bench sum, not via auto-sub.
    assert bb_score == 10 + 1 + 13  # = 24
