"""Walk-forward multi-season backtest.

For each test season, train fresh mean + P90 models on data strictly before,
then run a chips-on backtest. Gives a clean N=3 read on agent performance,
free of training-set leakage. Also reports DGW count per season because
chip value scales with DGW availability.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import pandas as pd

from fpl_agent.backtest import run_backtest
from fpl_agent.features import build_features
from fpl_agent.history import SEASONS
from fpl_agent.model import (
    default_params_mean, default_params_p90, train, train_two_stage,
)

DEFAULT_TEST_SEASONS = ("2023-24", "2024-25", "2025-26")


@dataclass
class SeasonResult:
    season: str
    gws: int
    agent_total: float
    agent_avg: float
    static_total: float
    static_avg: float
    oracle_avg: float
    n_dgws: int
    chips: list[tuple[int, str, float]]  # (gw, chip, actual_pts)


def _prior(season: str) -> str:
    return SEASONS[SEASONS.index(season) - 1]


def _count_dgws(df_raw: pd.DataFrame, season: str) -> int:
    sub = df_raw[df_raw["season"] == season]
    team_fix = sub.groupby(["GW", "team", "fixture"]).size().reset_index()
    team_gw = team_fix.groupby(["GW", "team"]).size().reset_index(name="n")
    return int((team_gw["n"] > 1).sum())


def run_season(df_raw: pd.DataFrame, df_feat: pd.DataFrame, test_season: str,
                use_chips: bool = True, horizon_h: int = 0,
                lookahead_weight: float = 0.5,
                two_stage: bool = False) -> SeasonResult:
    val = _prior(test_season)
    train_until = _prior(val)
    print(f"\n>>> {test_season}: train ≤ {train_until}, val = {val}"
          f"{' [two-stage]' if two_stage else ''}")

    t = time.time()
    if two_stage:
        models = train_two_stage(df_feat, train_until=train_until, val_season=val)
        booster_mean = models["predictor_mean"]
        booster_p90 = models["predictor_p90"]
        print(f"    trained in {time.time()-t:.1f}s "
              f"(stage1+stage2_mean+stage2_p90)")
    else:
        booster_mean, m_mean = train(df_feat, train_until=train_until, val_season=val,
                                      params=default_params_mean())
        booster_p90, m_p90 = train(df_feat, train_until=train_until, val_season=val,
                                    params=default_params_p90())
        print(f"    trained in {time.time()-t:.1f}s "
              f"(mean val MAE {m_mean['val_mae']:.3f}, p90 val MAE {m_p90['val_mae']:.3f})")

    rep = run_backtest(booster_mean, df_raw, test_season,
                        booster_p90=booster_p90, start_gw=2, use_chips=use_chips,
                        horizon_h=horizon_h, lookahead_weight=lookahead_weight)

    chips = [(r.gw, r.chip, r.actual_points) for r in rep.agent.per_gw if r.chip]
    return SeasonResult(
        season=test_season,
        gws=len(rep.gws),
        agent_total=rep.agent.total, agent_avg=rep.agent.avg,
        static_total=rep.static.total, static_avg=rep.static.avg,
        oracle_avg=rep.oracle.avg,
        n_dgws=_count_dgws(df_raw, test_season),
        chips=chips,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=list(DEFAULT_TEST_SEASONS))
    ap.add_argument("--no-chips", action="store_true")
    ap.add_argument("--horizon", type=int, default=5,
                    help="Number of future GWs to include in lookahead bonus "
                         "(default 5; 0=off). Tuned across 3 seasons.")
    ap.add_argument("--lookahead-weight", type=float, default=0.5,
                    help="Weight multiplier on the squad-ownership bonus "
                         "(default 0.5). Tuned across 3 seasons.")
    ap.add_argument("--two-stage", action="store_true",
                    help="Use two-stage model: P(plays>=60) × E(pts|plays>=60)")
    args = ap.parse_args()

    df = pd.read_parquet("data/processed/gw_history.parquet")
    print(f"Loaded {len(df):,} rows; building features once...")
    t = time.time()
    df_feat = build_features(df)
    print(f"Built features in {time.time()-t:.1f}s")

    results: list[SeasonResult] = []
    for s in args.seasons:
        results.append(run_season(df, df_feat, s,
                                   use_chips=not args.no_chips,
                                   horizon_h=args.horizon,
                                   lookahead_weight=args.lookahead_weight,
                                   two_stage=args.two_stage))

    print("\n" + "=" * 78)
    print(f"  MULTI-SEASON SUMMARY (chips {'OFF' if args.no_chips else 'ON'})")
    print("=" * 78)
    print(f"{'season':<10} {'gws':>4} {'dgws':>5} {'agent':>7} {'/gw':>6}  "
          f"{'static':>7} {'/gw':>6}  {'oracle/gw':>10}  {'chips':<20}")
    for r in results:
        chip_str = ",".join(f"GW{g}:{c}" for g, c, _ in r.chips) or "-"
        print(f"{r.season:<10} {r.gws:>4} {r.n_dgws:>5} "
              f"{r.agent_total:>7.0f} {r.agent_avg:>6.1f}  "
              f"{r.static_total:>7.0f} {r.static_avg:>6.1f}  "
              f"{r.oracle_avg:>10.1f}  {chip_str:<20}")

    if results:
        avg_agent = sum(r.agent_avg for r in results) / len(results)
        avg_static = sum(r.static_avg for r in results) / len(results)
        print()
        print(f"  Average across {len(results)} seasons:")
        print(f"    agent  {avg_agent:.1f} pts/GW")
        print(f"    static {avg_static:.1f} pts/GW")
        print(f"    FPL benchmark: ~53 avg, ~65 top-10k, ~75 world #1")


if __name__ == "__main__":
    main()
