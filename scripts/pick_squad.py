"""Pick an optimal FPL squad for a target gameweek using the trained xPts model.

The model + parquet pipeline replaces the earlier `ep_next` placeholder.
Without --gw, picks for the latest gameweek present in the data.
With --show-actuals, also reports what each pick actually scored (retrospective).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fpl_agent.features import build_features
from fpl_agent.model import build_players_for_gw, load_model, predict
from fpl_agent.optimizer import pick_squad
from fpl_agent.rules import POSITIONS

HISTORY_PARQUET = Path("data/processed/gw_history.parquet")


def _print_squad(sel, actuals: pd.DataFrame | None) -> None:
    print(f"Squad cost: GBP {sel.cost / 10:.1f}m / 100.0m")
    print(f"Predicted points (XI + captain double): {sel.expected_points:.2f}\n")

    starting_ids = {p.id for p in sel.starting_xi}
    by_pos: dict[int, list] = {pid: [] for pid in POSITIONS}
    for p in sel.squad:
        by_pos[p.position].append(p)

    actual_lookup = {}
    if actuals is not None:
        actual_lookup = (
            actuals.groupby("element")["total_points"].sum().to_dict()
        )

    for pos_id, (name, *_rest) in POSITIONS.items():
        print(f"  {name}")
        for p in sorted(by_pos[pos_id], key=lambda x: -x.xpts):
            tag = "XI " if p.id in starting_ids else "B  "
            cap = " (C)" if p.id == sel.captain.id else ""
            actual = ""
            if p.id in actual_lookup:
                actual = f"  actual {actual_lookup[p.id]:>3.0f}"
            print(
                f"    {tag} {p.name:<22} {p.team_name:<22} "
                f"GBP {p.price/10:>4.1f}m  xP {p.xpts:>5.2f}{cap}{actual}"
            )


def _print_top_predictions_vs_actuals(players, actuals: pd.DataFrame) -> None:
    actual_lookup = actuals.groupby("element")["total_points"].sum().to_dict()
    name_lookup = actuals.drop_duplicates("element").set_index("element")["name"].to_dict()

    print("\n--- Model top-10 predictions for this GW ---")
    for p in sorted(players, key=lambda x: -x.xpts)[:10]:
        a = actual_lookup.get(p.id, 0)
        print(f"  {p.name:<22} {p.team_name:<22}  xP {p.xpts:>5.2f}  actual {a:.0f}")

    print("\n--- Actual top-10 scorers for this GW ---")
    actual_sorted = sorted(actual_lookup.items(), key=lambda kv: -kv[1])[:10]
    for elem_id, pts in actual_sorted:
        print(f"  {name_lookup.get(elem_id, str(elem_id)):<22}  actual {pts:.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2025-26")
    ap.add_argument("--gw", type=int, default=None,
                    help="Gameweek to predict (default: latest in data)")
    ap.add_argument("--show-actuals", action="store_true",
                    help="Also show actual points scored that GW (retrospective)")
    args = ap.parse_args()

    df = pd.read_parquet(HISTORY_PARQUET)
    print(f"Loaded {len(df):,} rows; building features...")
    df_feat = build_features(df)

    season_gws = df_feat.loc[df_feat["season"] == args.season, "GW"]
    if season_gws.empty:
        raise SystemExit(f"No rows in parquet for season {args.season}")
    if args.gw is None:
        args.gw = int(season_gws.max())
    print(f"Target: {args.season} GW {args.gw}")

    print("Loading model and predicting...")
    booster = load_model()
    preds = predict(booster, df_feat)

    players = build_players_for_gw(df_feat, preds, args.season, args.gw)
    print(f"  {len(players)} eligible players (with at least 1 prior GW of history)\n")

    sel = pick_squad(players)

    actuals = None
    if args.show_actuals:
        actuals = df[(df["season"] == args.season) & (df["GW"] == args.gw)]

    _print_squad(sel, actuals)

    if actuals is not None:
        _print_top_predictions_vs_actuals(players, actuals)
        starting_ids = {p.id for p in sel.starting_xi}
        xi_actual = actuals.loc[actuals["element"].isin(starting_ids), "total_points"].sum()
        cap_actual = actuals.loc[actuals["element"] == sel.captain.id, "total_points"].sum()
        total_actual = xi_actual + cap_actual  # +1x extra for captain
        print(f"\nSquad ACTUAL points this GW (XI + captain double): {total_actual:.0f}")


if __name__ == "__main__":
    main()
