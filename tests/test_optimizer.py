from collections import Counter

from fpl_agent.data import Player
from fpl_agent.optimizer import pick_squad
from fpl_agent.rules import BUDGET_TENTHS, MAX_PER_CLUB, POSITIONS, SQUAD_SIZE, STARTING_XI


def _synthetic_players() -> list[Player]:
    players: list[Player] = []
    pid = 0
    for team_id in range(1, 9):
        for pos_id, (name, _, _, _) in POSITIONS.items():
            for k in range(4):
                pid += 1
                players.append(Player(
                    id=pid,
                    name=f"{name}{pid}",
                    team_id=team_id,
                    team_name=f"Club{team_id}",
                    position=pos_id,
                    price=40 + (pid % 80),       # £4.0m – £11.9m
                    xpts=1.0 + (pid % 17) * 0.3,
                ))
    return players


def test_pick_squad_satisfies_all_rules():
    sel = pick_squad(_synthetic_players())

    assert len(sel.squad) == SQUAD_SIZE
    assert len(sel.starting_xi) == STARTING_XI
    assert len(sel.bench) == SQUAD_SIZE - STARTING_XI
    assert sel.captain in sel.starting_xi
    assert sel.cost <= BUDGET_TENTHS

    squad_pos = Counter(p.position for p in sel.squad)
    for pos_id, (_, count, _, _) in POSITIONS.items():
        assert squad_pos[pos_id] == count

    club_counts = Counter(p.team_id for p in sel.squad)
    assert max(club_counts.values()) <= MAX_PER_CLUB

    xi_pos = Counter(p.position for p in sel.starting_xi)
    for pos_id, (_, _, smin, smax) in POSITIONS.items():
        assert smin <= xi_pos[pos_id] <= smax


def test_optimizer_prefers_high_xpts():
    """Among equal-price players, the higher-xpts one should be chosen."""
    players = _synthetic_players()
    # Inject one obviously dominant midfielder
    star = Player(id=9999, name="STAR", team_id=99, team_name="StarFC",
                  position=3, price=50, xpts=100.0)
    sel = pick_squad(players + [star])
    assert star in sel.squad
    assert star in sel.starting_xi
    assert sel.captain.id == star.id
