"""Backtest engine: simulate playing a full FPL season gameweek by gameweek.

Three strategies are evaluated side-by-side:
    1. agent       -- model predictions + transfer-aware optimiser
    2. static      -- pick optimal squad once at start_gw, then optimal XI each week
    3. oracle      -- each GW, pick optimal squad+XI+captain using ACTUAL points
                      (no transfer constraints — absolute upper bound)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import pandas as pd

from .data import Player
from .features import build_features
from .model import build_players_for_gw, predict
from .optimizer import Selection, pick_squad, pick_xi


@dataclass
class GWResult:
    gw: int
    predicted_points: float   # what the model expected (net of hits)
    actual_points: float      # XI actual + captain actual (doubled), minus hits
    hits: int
    transfers_made: int
    captain_name: str
    captain_actual: float
    chip: str = ""            # "WC", "TC", "BB", or "" — chip played this GW


@dataclass
class ChipsState:
    """Tracks which chips have been used this backtest. v1: 1 of each per season,
    one chip per GW max. (FPL's 2024-25+ rules give 2 of each via halves; ignored
    here for simplicity — can extend later.)"""
    wc_used: bool = False
    tc_used: bool = False
    bb_used: bool = False

    def available(self, chip: str) -> bool:
        return not getattr(self, f"{chip.lower()}_used")

    def use(self, chip: str) -> None:
        setattr(self, f"{chip.lower()}_used", True)


@dataclass
class StrategyResult:
    name: str
    per_gw: list[GWResult] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(r.actual_points for r in self.per_gw)

    @property
    def avg(self) -> float:
        return self.total / max(1, len(self.per_gw))


@dataclass
class BacktestReport:
    season: str
    gws: list[int]
    agent: StrategyResult
    static: StrategyResult
    oracle: StrategyResult


def _actuals_lookup(df_raw: pd.DataFrame, season: str, gw: int) -> dict[int, float]:
    """Map element id -> actual total_points scored in (season, gw),
    summed across any DGW fixtures."""
    sub = df_raw[(df_raw["season"] == season) & (df_raw["GW"] == gw)]
    return sub.groupby("element")["total_points"].sum().to_dict()


def _players_with_actuals(
    df_raw: pd.DataFrame, season: str, gw: int, prior_players: dict[int, Player]
) -> list[Player]:
    """Build Player objects with xpts = actual points (for oracle strategy).
    Uses team/price/position from the merged_gw row directly."""
    sub = df_raw[(df_raw["season"] == season) & (df_raw["GW"] == gw)].copy()
    if sub.empty:
        return []
    pos_to_id = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
    teams = {t: i for i, t in enumerate(sorted(sub["team"].dropna().unique()))}

    agg = sub.groupby("element", as_index=False).agg(
        name=("name", "first"),
        team=("team", "first"),
        position=("position", "first"),
        value=("value", "first"),
        actual=("total_points", "sum"),
    )

    out: list[Player] = []
    for _, r in agg.iterrows():
        if pd.isna(r["position"]) or r["position"] not in pos_to_id:
            continue
        out.append(Player(
            id=int(r["element"]),
            name=str(r["name"]),
            team_id=teams.get(r["team"], -1),
            team_name=str(r["team"]),
            position=pos_to_id[r["position"]],
            price=int(r["value"]) if not pd.isna(r["value"]) else 50,
            xpts=float(r["actual"]),
        ))
    return out


def _fill_missing_squad_members(
    pool: list[Player], current_squad: list[Player]
) -> list[Player]:
    """If a current-squad member doesn't appear in this GW's candidate pool
    (injured, didn't feature), add them with xpts=0 so the optimiser can
    still choose to keep or transfer them."""
    pool_ids = {p.id for p in pool}
    extras = [
        Player(id=p.id, name=p.name, team_id=p.team_id, team_name=p.team_name,
               position=p.position, price=p.price, xpts=0.0)
        for p in current_squad if p.id not in pool_ids
    ]
    return pool + extras


def _score_xi(starting_xi: list[Player], captain: Player, actuals: dict[int, float],
              hits: int = 0, hit_cost: float = 4.0,
              bench: list[Player] | None = None, chip: str = "") -> float:
    """Score a gameweek, honouring chip effects.

    - normal:  XI total + 1x captain extra (= 2x captain in total) - hits*4
    - TC:      XI total + 2x captain extra (= 3x captain in total) - hits*4
    - BB:      normal + bench's actual points
    - WC:      normal with hit cost zeroed (transfers were free)
    """
    xi_pts = sum(actuals.get(p.id, 0.0) for p in starting_xi)
    cap_pts = actuals.get(captain.id, 0.0)
    cap_multiplier = 2 if chip == "TC" else 1
    bench_pts = (sum(actuals.get(p.id, 0.0) for p in (bench or []))
                 if chip == "BB" else 0)
    effective_hits = 0 if chip == "WC" else hits
    return xi_pts + cap_multiplier * cap_pts + bench_pts - effective_hits * hit_cost


def _decide_chip(
    sel_constrained: Selection,
    sel_wildcard: Selection | None,
    chips: ChipsState,
    *,
    wc_min_hits_saved: int = 2,
    wc_gain_threshold: float = 30.0,
    tc_ceiling_threshold: float = 11.0,
    bb_bench_threshold: float = 10.0,
) -> str:
    """Decide which chip (if any) to play this GW.

    Greedy heuristics, one chip per GW max, priority WC > TC > BB:
      - WC: fire only when conservative. Triggers if either
          (a) constrained pick was forced to take ≥ `wc_min_hits_saved` hits
              (WC saves at least that many * 4 pts), OR
          (b) unconstrained gain over constrained > `wc_gain_threshold`
              (a deliberately strict bar — first backtest showed our model
              over-trusts WC picks and cascades badly in following GWs).
      - TC if captain's ceiling_xpts > `tc_ceiling_threshold`.
      - BB if bench's predicted total > `bb_bench_threshold`.
    """
    if chips.available("WC") and sel_wildcard is not None:
        wc_gain = sel_wildcard.expected_points - sel_constrained.expected_points
        if sel_constrained.hits >= wc_min_hits_saved or wc_gain > wc_gain_threshold:
            chips.use("WC")
            return "WC"

    if chips.available("TC"):
        cap = sel_constrained.captain
        ceiling = cap.ceiling_xpts if cap.ceiling_xpts is not None else cap.xpts
        if ceiling > tc_ceiling_threshold:
            chips.use("TC")
            return "TC"

    if chips.available("BB"):
        bench_pred = sum(p.xpts for p in sel_constrained.bench)
        if bench_pred > bb_bench_threshold:
            chips.use("BB")
            return "BB"

    return ""


def _horizon_xpts_by_id(
    df_feat: pd.DataFrame, preds: pd.Series, season: str,
    start_gw: int, horizon_h: int,
) -> dict[int, float]:
    """Sum predicted xpts per player across GWs (start_gw .. start_gw+H-1)."""
    if horizon_h <= 0:
        return {}
    out: dict[int, float] = {}
    for g in range(start_gw, start_gw + horizon_h):
        future_pool = build_players_for_gw(df_feat, preds, season, g)
        for p in future_pool:
            out[p.id] = out.get(p.id, 0.0) + p.xpts
    return out


def run_backtest(
    booster: lgb.Booster,
    df_raw: pd.DataFrame,
    season: str,
    *,
    booster_p90: lgb.Booster | None = None,
    start_gw: int = 2,
    end_gw: int | None = None,
    free_transfers_per_gw: int = 1,
    max_banked_extra: int = 1,
    hit_cost: float = 4.0,
    use_chips: bool = True,
    horizon_h: int = 0,
    lookahead_weight: float = 0.5,
) -> BacktestReport:
    """Run agent, static-squad, and per-GW oracle strategies on a season.

    `booster` is the mean (L2) model used for squad/XI selection.
    `booster_p90` is the optional P90 quantile model used for captain choice —
        if given, players carry both xpts (mean) and ceiling_xpts (P90).
    `horizon_h`: how many FUTURE GWs to include in the lookahead bonus
        (0 = no lookahead, just current GW. Default 0 for back-compat.)
    `lookahead_weight`: weight multiplier on the squad-ownership bonus term
        (per-player Σ future-GW xpts). 0 disables.
    `free_transfers_per_gw`: 1 in standard FPL.
    `max_banked_extra`: extra unused FTs you can carry over. FPL allowed banking
        up to 1 (so max 2 available) before 2024-25, up to 5 after.
        Default 1 (conservative; agent will use them rather than waste them).
    `start_gw`: first GW to play (default 2 — GW1 has no in-season history).
    """
    df_feat = build_features(df_raw)
    preds = predict(booster, df_feat)
    preds_p90 = predict(booster_p90, df_feat) if booster_p90 is not None else None

    season_gws = sorted(df_raw.loc[df_raw["season"] == season, "GW"].unique())
    if end_gw is None:
        end_gw = max(season_gws)
    gws_to_play = [g for g in season_gws if start_gw <= g <= end_gw]

    agent = StrategyResult(name="agent")
    static = StrategyResult(name="static")
    oracle = StrategyResult(name="oracle")

    agent_squad: list[Player] = []
    agent_squad_ids: set[int] | None = None
    banked_ft = 0
    chips_state = ChipsState()

    static_squad: list[Player] = []

    for gw in gws_to_play:
        actuals = _actuals_lookup(df_raw, season, gw)
        horizon = _horizon_xpts_by_id(df_feat, preds, season, gw + 1, horizon_h) \
            if horizon_h > 0 else None
        pool = build_players_for_gw(df_feat, preds, season, gw,
                                    ceiling_predictions=preds_p90,
                                    horizon_xpts=horizon)
        if not pool:
            continue

        # ---- AGENT strategy ----
        agent_pool = _fill_missing_squad_members(pool, agent_squad) if agent_squad else pool
        if agent_squad_ids is None:
            # Initial squad — fresh pick, USE lookahead (we're committing to a 15-man
            # squad we'll mostly carry for several weeks).
            sel = pick_squad(agent_pool, lookahead_weight=lookahead_weight)
            chip_played = ""
        else:
            available_ft = min(banked_ft + free_transfers_per_gw,
                               max_banked_extra + free_transfers_per_gw)
            sel_constrained = pick_squad(
                agent_pool,
                current_squad_ids=agent_squad_ids,
                free_transfers=available_ft,
                hit_cost=hit_cost,
                lookahead_weight=lookahead_weight,
            )
            # For WC consideration, compute an unconstrained pick (also w/ lookahead)
            sel_wildcard = pick_squad(agent_pool, lookahead_weight=lookahead_weight) \
                if use_chips else None
            chip_played = _decide_chip(sel_constrained, sel_wildcard, chips_state) \
                if use_chips else ""
            sel = sel_wildcard if chip_played == "WC" else sel_constrained

        agent.per_gw.append(GWResult(
            gw=gw,
            predicted_points=sel.expected_points,
            actual_points=_score_xi(sel.starting_xi, sel.captain, actuals,
                                    hits=sel.hits, hit_cost=hit_cost,
                                    bench=sel.bench, chip=chip_played),
            hits=sel.hits,
            transfers_made=len(sel.transfers_in),
            captain_name=sel.captain.name,
            captain_actual=actuals.get(sel.captain.id, 0.0),
            chip=chip_played,
        ))

        # Update agent state
        agent_squad = sel.squad
        agent_squad_ids = {p.id for p in agent_squad}
        if agent_squad_ids is not None:  # not first GW
            used_free = min(len(sel.transfers_in), free_transfers_per_gw + banked_ft)
            banked_ft = min(max_banked_extra,
                            (banked_ft + free_transfers_per_gw) - used_free)

        # ---- STATIC strategy ----
        # Pick the squad once on the first played GW; then re-rank XI/captain
        # each week using model predictions on the same 15 players.
        if not static_squad:
            static_squad = sel.squad[:]  # mirror initial pick for fair start

        # Re-build predictions for these 15 in this GW's context
        pool_by_id = {p.id: p for p in pool}
        static_now = [
            Player(id=p.id, name=p.name, team_id=p.team_id, team_name=p.team_name,
                   position=p.position, price=p.price,
                   xpts=pool_by_id[p.id].xpts if p.id in pool_by_id else 0.0,
                   ceiling_xpts=pool_by_id[p.id].ceiling_xpts if p.id in pool_by_id else None)
            for p in static_squad
        ]
        try:
            xi, cap, _ = pick_xi(static_now)
            static_pts = _score_xi(xi, cap, actuals)
        except RuntimeError:
            xi, cap, static_pts = [], None, 0.0

        static.per_gw.append(GWResult(
            gw=gw,
            predicted_points=sum(p.xpts for p in xi) + (cap.xpts if cap else 0.0),
            actual_points=static_pts,
            hits=0,
            transfers_made=0,
            captain_name=cap.name if cap else "-",
            captain_actual=actuals.get(cap.id, 0.0) if cap else 0.0,
        ))

        # ---- ORACLE strategy (unconstrained best each week) ----
        actual_players = _players_with_actuals(df_raw, season, gw, {})
        if actual_players:
            sel_oracle = pick_squad(actual_players)
            oracle_pts = _score_xi(sel_oracle.starting_xi, sel_oracle.captain, actuals)
            oracle.per_gw.append(GWResult(
                gw=gw,
                predicted_points=sel_oracle.expected_points,
                actual_points=oracle_pts,
                hits=0, transfers_made=0,
                captain_name=sel_oracle.captain.name,
                captain_actual=actuals.get(sel_oracle.captain.id, 0.0),
            ))

    return BacktestReport(season=season, gws=gws_to_play,
                          agent=agent, static=static, oracle=oracle)
