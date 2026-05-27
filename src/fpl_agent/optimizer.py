"""MILP optimizer for FPL squad selection.

Decision variables (per player i):
    x[i] in {0,1}  -- player is in the 15-man squad
    s[i] in {0,1}  -- player is in the starting XI
    c[i] in {0,1}  -- player is captain

Objective: maximise   sum_i s[i] * xpts[i]  +  sum_i c[i] * xpts[i]
           (XI scores once; captain scores again -> total 2x for the captain)

With transfer accounting (when current_squad_ids is given):
    hits >= sum(x[i] for i NOT in current_squad_ids) - free_transfers
    objective -= hit_cost * hits
"""
from dataclasses import dataclass, field

import pulp

from .data import Player
from .rules import BUDGET_TENTHS, MAX_PER_CLUB, POSITIONS, SQUAD_SIZE, STARTING_XI


@dataclass
class Selection:
    squad: list[Player]        # 15
    starting_xi: list[Player]  # 11
    captain: Player
    bench: list[Player]        # 4
    expected_points: float     # XI + captain double, net of any hit cost
    cost: int                  # tenths of millions
    # Transfer accounting (populated only when current_squad_ids given)
    transfers_in: list[Player] = field(default_factory=list)
    transfers_out_ids: list[int] = field(default_factory=list)
    hits: int = 0


def pick_squad(
    players: list[Player],
    budget: int = BUDGET_TENTHS,
    max_per_club: int = MAX_PER_CLUB,
    *,
    current_squad_ids: set[int] | None = None,
    free_transfers: int = 0,
    hit_cost: float = 4.0,
    lookahead_weight: float = 0.0,
) -> Selection:
    """Pick squad + starting XI + captain that maximise expected points.

    If `current_squad_ids` is provided, transfers from that squad to the new one
    cost `hit_cost` points each beyond `free_transfers`.

    `lookahead_weight` (default 0 = off) adds a per-squad-member bonus equal to
    `lookahead_weight × player.horizon_xpts` (the player's predicted xpts summed
    over the next H-1 GWs). This favours picking players with good upcoming
    fixture runs into the 15-man squad without changing this week's XI choice.
    """
    n = len(players)
    idx = range(n)

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in idx]
    s = [pulp.LpVariable(f"s_{i}", cat="Binary") for i in idx]
    c = [pulp.LpVariable(f"c_{i}", cat="Binary") for i in idx]

    # Captain term uses ceiling_xpts (P90) when available — captures upside
    # the mean-trained model under-weights. Falls back to mean xpts otherwise.
    cap_xpts = [
        p.ceiling_xpts if p.ceiling_xpts is not None else p.xpts
        for p in players
    ]
    points_obj = pulp.lpSum(
        s[i] * players[i].xpts + c[i] * cap_xpts[i] for i in idx
    )
    if lookahead_weight > 0:
        points_obj += lookahead_weight * pulp.lpSum(
            x[i] * players[i].horizon_xpts for i in idx
        )

    hits_var = None
    if current_squad_ids is not None:
        new_indices = [i for i, p in enumerate(players) if p.id not in current_squad_ids]
        transfers_in_expr = pulp.lpSum(x[i] for i in new_indices)
        hits_var = pulp.LpVariable("hits", lowBound=0)
        prob += hits_var >= transfers_in_expr - free_transfers
        prob += points_obj - hit_cost * hits_var
    else:
        prob += points_obj

    # Squad: total + per-position counts
    prob += pulp.lpSum(x) == SQUAD_SIZE
    for pos_id, (_, count, _, _) in POSITIONS.items():
        prob += pulp.lpSum(x[i] for i in idx if players[i].position == pos_id) == count

    # Budget
    prob += pulp.lpSum(x[i] * players[i].price for i in idx) <= budget

    # Max per club
    for club in {p.team_id for p in players}:
        prob += pulp.lpSum(x[i] for i in idx if players[i].team_id == club) <= max_per_club

    # Starting XI: total + per-position bounds + must be in squad
    prob += pulp.lpSum(s) == STARTING_XI
    for pos_id, (_, _, smin, smax) in POSITIONS.items():
        in_pos = [s[i] for i in idx if players[i].position == pos_id]
        prob += pulp.lpSum(in_pos) >= smin
        prob += pulp.lpSum(in_pos) <= smax
    for i in idx:
        prob += s[i] <= x[i]

    # Captain: exactly 1, must be in starting XI
    prob += pulp.lpSum(c) == 1
    for i in idx:
        prob += c[i] <= s[i]

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Optimizer status: {pulp.LpStatus[status]}")

    squad = [players[i] for i in idx if x[i].value() > 0.5]
    starting = [players[i] for i in idx if s[i].value() > 0.5]
    captain = next(players[i] for i in idx if c[i].value() > 0.5)
    starting_ids = {p.id for p in starting}
    bench = [p for p in squad if p.id not in starting_ids]
    cost = sum(p.price for p in squad)
    cap_extra = captain.ceiling_xpts if captain.ceiling_xpts is not None else captain.xpts
    base_ep = sum(p.xpts for p in starting) + cap_extra

    transfers_in_list: list[Player] = []
    transfers_out_ids: list[int] = []
    hits = 0
    ep = base_ep
    if current_squad_ids is not None:
        squad_ids_set = {p.id for p in squad}
        transfers_in_list = [p for p in squad if p.id not in current_squad_ids]
        transfers_out_ids = sorted(current_squad_ids - squad_ids_set)
        hits = max(0, len(transfers_in_list) - free_transfers)
        ep = base_ep - hit_cost * hits

    return Selection(
        squad=squad,
        starting_xi=starting,
        captain=captain,
        bench=bench,
        expected_points=ep,
        cost=cost,
        transfers_in=transfers_in_list,
        transfers_out_ids=transfers_out_ids,
        hits=hits,
    )


def pick_xi(squad: list[Player]) -> tuple[list[Player], Player, list[Player]]:
    """Given a fixed 15-man squad, pick the best starting XI + captain.

    Used by baselines that hold a squad fixed across gameweeks but still pick
    the best lineup each week.
    Returns (starting_xi, captain, bench).
    """
    n = len(squad)
    idx = range(n)

    prob = pulp.LpProblem("fpl_xi", pulp.LpMaximize)
    s = [pulp.LpVariable(f"s_{i}", cat="Binary") for i in idx]
    c = [pulp.LpVariable(f"c_{i}", cat="Binary") for i in idx]

    prob += pulp.lpSum(s[i] * squad[i].xpts + c[i] * squad[i].xpts for i in idx)
    prob += pulp.lpSum(s) == STARTING_XI
    for pos_id, (_, _, smin, smax) in POSITIONS.items():
        in_pos = [s[i] for i in idx if squad[i].position == pos_id]
        prob += pulp.lpSum(in_pos) >= smin
        prob += pulp.lpSum(in_pos) <= smax
    prob += pulp.lpSum(c) == 1
    for i in idx:
        prob += c[i] <= s[i]

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"pick_xi status: {pulp.LpStatus[status]}")

    starting = [squad[i] for i in idx if s[i].value() > 0.5]
    captain = next(squad[i] for i in idx if c[i].value() > 0.5)
    starting_ids = {p.id for p in starting}
    bench = [p for p in squad if p.id not in starting_ids]
    return starting, captain, bench
