"""Run the full-season backtest on a target season and print a comparison."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fpl_agent.backtest import BacktestReport, run_backtest
from fpl_agent.model import DEFAULT_MEAN_PATH, DEFAULT_P90_PATH, load_model

HISTORY_PARQUET = Path("data/processed/gw_history.parquet")

FPL_AVG_HINT = 53  # rough Premier League FPL average pts/GW (varies year to year)
TOP10K_HINT = 65   # rough top-10k cutoff pts/GW


def _print_report(rep: BacktestReport) -> None:
    print(f"\n=== Backtest: {rep.season} (GW {rep.gws[0]}..{rep.gws[-1]}, "
          f"{len(rep.gws)} GWs) ===\n")

    header = f"{'GW':>3} | {'AGENT':>5} {'cap':<22} {'tr':>2} {'hits':>4} {'chip':>4} | " \
             f"{'STATIC':>6} {'cap':<22} | {'ORACLE':>6} {'cap':<22}"
    print(header)
    print("-" * len(header))

    for i, gw in enumerate(rep.gws):
        ag = rep.agent.per_gw[i]
        st = rep.static.per_gw[i] if i < len(rep.static.per_gw) else None
        orc = rep.oracle.per_gw[i] if i < len(rep.oracle.per_gw) else None
        chip_str = ag.chip if ag.chip else "-"
        ag_line = f"{ag.actual_points:>5.0f} {ag.captain_name[:22]:<22} " \
                  f"{ag.transfers_made:>2d} {ag.hits:>4d} {chip_str:>4}"
        st_line = f"{st.actual_points:>6.0f} {st.captain_name[:22]:<22}" if st else ""
        orc_line = f"{orc.actual_points:>6.0f} {orc.captain_name[:22]:<22}" if orc else ""
        print(f"{gw:>3} | {ag_line} | {st_line} | {orc_line}")

    print("\n--- Totals ---")
    print(f"  AGENT   total {rep.agent.total:>5.0f}   avg/GW {rep.agent.avg:>5.1f}")
    print(f"  STATIC  total {rep.static.total:>5.0f}   avg/GW {rep.static.avg:>5.1f}")
    print(f"  ORACLE  total {rep.oracle.total:>5.0f}   avg/GW {rep.oracle.avg:>5.1f}")
    print()
    print(f"  Reference: FPL average ~{FPL_AVG_HINT} pts/GW, "
          f"top-10k cutoff ~{TOP10K_HINT} pts/GW")
    gap_oracle = (rep.agent.avg / rep.oracle.avg * 100) if rep.oracle.avg else 0
    gap_static = (rep.agent.avg / rep.static.avg * 100) if rep.static.avg else 0
    print(f"  Agent captures {gap_oracle:.0f}% of the oracle ceiling, "
          f"{gap_static:.0f}% of static.")

    print("\n--- Agent transfer summary ---")
    total_tr = sum(r.transfers_made for r in rep.agent.per_gw)
    total_hits = sum(r.hits for r in rep.agent.per_gw)
    print(f"  total transfers: {total_tr}")
    print(f"  total hits:      {total_hits} ({total_hits * 4} pts lost)")

    chips_used = [(r.gw, r.chip, r.actual_points, r.captain_name)
                  for r in rep.agent.per_gw if r.chip]
    print(f"\n--- Chips used: {len(chips_used)} ---")
    for gw, chip, pts, cap in chips_used:
        print(f"  GW{gw:>2} {chip:>3}: scored {pts:.0f} pts (captain: {cap})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2025-26")
    ap.add_argument("--start-gw", type=int, default=2)
    ap.add_argument("--end-gw", type=int, default=None)
    ap.add_argument("--hit-cost", type=float, default=4.0)
    ap.add_argument("--no-chips", action="store_true",
                    help="Disable chip strategy (baseline comparison)")
    args = ap.parse_args()

    df = pd.read_parquet(HISTORY_PARQUET)
    print(f"Loaded {len(df):,} rows; loading models...")
    booster_mean = load_model(DEFAULT_MEAN_PATH)
    booster_p90 = load_model(DEFAULT_P90_PATH) if DEFAULT_P90_PATH.exists() else None
    if booster_p90 is None:
        print("  WARN: no P90 model found — captain will fall back to mean preds")
    else:
        print(f"  loaded mean + P90 from {DEFAULT_MEAN_PATH.parent}")

    print(f"Running backtest on {args.season} (start GW {args.start_gw}, "
          f"chips={'OFF' if args.no_chips else 'ON'})...")
    rep = run_backtest(
        booster_mean, df, args.season,
        booster_p90=booster_p90,
        start_gw=args.start_gw,
        end_gw=args.end_gw,
        hit_cost=args.hit_cost,
        use_chips=not args.no_chips,
    )
    _print_report(rep)


if __name__ == "__main__":
    main()
