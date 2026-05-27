"""Diagnose why the agent's backtest score (~50/GW) is below FPL average.

Five sections, each testing a concrete hypothesis:
  A. Distribution shape — does the model collapse to the mean?
  B. Captain quality  — how big is the gap to hindsight-best captain?
  C. Ranking quality  — does top-K-pred overlap with top-K-actual?
  D. Starter vs bench — is MAE good only because most players score ~0?
  E. Initial squad    — is the GW2 squad pick already capping our ceiling?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

from fpl_agent.backtest import _horizon_xpts_by_id
from fpl_agent.data import Player
from fpl_agent.features import build_features
from fpl_agent.model import (
    DEFAULT_MEAN_PATH, DEFAULT_P90_PATH,
    build_players_for_gw, load_model, predict,
)
from fpl_agent.optimizer import pick_squad


SEASON = "2025-26"
LOOKAHEAD_H = 5
LOOKAHEAD_W = 0.5


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def histogram_compare(pred: pd.Series, actual: pd.Series) -> None:
    """Side-by-side text histogram in identical bins."""
    bins = [-0.5, 0.5, 1.5, 3.5, 6.5, 10.5, 15.5, 30.5]
    labels = ["0", "1", "2-3", "4-6", "7-10", "11-15", "16+"]
    pred_h = pd.cut(pred, bins).value_counts().sort_index().values
    act_h = pd.cut(actual, bins).value_counts().sort_index().values
    n = len(pred)

    print(f"\n  {'bucket':<8} {'predicted':>15} {'actual':>15}")
    print(f"  {'-' * 8} {'-' * 15} {'-' * 15}")
    for label, p, a in zip(labels, pred_h, act_h):
        ppct = 100 * p / n
        apct = 100 * a / n
        bar_p = "█" * max(1, int(ppct / 2)) if p > 0 else ""
        bar_a = "█" * max(1, int(apct / 2)) if a > 0 else ""
        print(f"  {label:<8} {p:>6} ({ppct:>4.1f}%)  {a:>6} ({apct:>4.1f}%)")
        print(f"  {'':<8} {bar_p:<15} {bar_a:<15}")


def main() -> None:
    df = pd.read_parquet("data/processed/gw_history.parquet")
    feats = build_features(df)
    booster_mean = load_model(DEFAULT_MEAN_PATH)
    booster_p90 = load_model(DEFAULT_P90_PATH) if DEFAULT_P90_PATH.exists() else None
    preds = predict(booster_mean, feats)
    preds_p90 = predict(booster_p90, feats) if booster_p90 is not None else None

    feats_s = feats[feats["season"] == SEASON].copy()
    rows = feats_s.loc[feats_s.index.intersection(preds.index)].copy()
    rows["pred"] = preds.loc[rows.index]
    if preds_p90 is not None:
        rows["pred_p90"] = preds_p90.reindex(rows.index)

    # ---------- A: Distribution ----------
    section("A. Distribution shape — does the model predict the MEAN only?")
    pred, actual = rows["pred"], rows["total_points"]
    print(f"  Predicted: mean={pred.mean():.2f}  std={pred.std():.2f}  "
          f"max={pred.max():.2f}  P95={pred.quantile(0.95):.2f}  "
          f"P99={pred.quantile(0.99):.2f}")
    print(f"  Actual:    mean={actual.mean():.2f}  std={actual.std():.2f}  "
          f"max={actual.max():.2f}  P95={actual.quantile(0.95):.2f}  "
          f"P99={actual.quantile(0.99):.2f}")
    histogram_compare(pred, actual)
    print(f"\n  KEY FINDING: max predicted xP this season = {pred.max():.2f}, "
          f"max actual = {actual.max():.0f}")
    print(f"  Variance ratio (actual/predicted): "
          f"{(actual.std() / max(0.01, pred.std()))**2:.1f}x")
    print(f"  → If true, model never recommends captaining for upside; "
          f"\"haul\" outcomes are invisible to it.")

    # ---------- B: Captain selection — uses P90 (what agent ACTUALLY does) ----------
    section("B. Captain selection — gap to hindsight-best (agent uses P90)")
    mean_caps, p90_caps, best_caps = [], [], []
    for gw, g in rows.groupby("GW"):
        if g.empty:
            continue
        by_id = g.groupby("element").agg(
            name=("name", "first"),
            pred_mean=("pred", "sum"),
            pred_p90=("pred_p90", "sum") if "pred_p90" in g.columns else ("pred", "sum"),
            actual=("total_points", "sum"),
        ).reset_index()
        mean_caps.append(by_id.sort_values("pred_mean", ascending=False).iloc[0]["actual"])
        p90_caps.append(by_id.sort_values("pred_p90", ascending=False).iloc[0]["actual"])
        best_caps.append(by_id.sort_values("actual", ascending=False).iloc[0]["actual"])

    print(f"  Mean-model top-1 captain (legacy):  avg actual = {np.mean(mean_caps):.2f}")
    print(f"  P90-model top-1 captain (agent's choice): avg actual = {np.mean(p90_caps):.2f}")
    print(f"  Hindsight perfect captain:                avg actual = {np.mean(best_caps):.2f}")
    print(f"  → Agent's captain gap: {np.mean(best_caps) - np.mean(p90_caps):.2f} pts per single "
          f"(doubled = {2*(np.mean(best_caps) - np.mean(p90_caps)):.1f}/GW).")
    print(f"  → P90 improvement vs mean: "
          f"{np.mean(p90_caps) - np.mean(mean_caps):+.2f} pts per single")
    print(f"\n  Per-GW sample (first 8):")
    print(f"  {'GW':>3}  {'mean-pick':>10}  {'p90-pick':>10}  {'best':>6}")
    for i, (gw, _) in enumerate(rows.groupby("GW")):
        if i >= 8:
            break
        print(f"  {gw:>3}  {mean_caps[i]:>10.0f}  {p90_caps[i]:>10.0f}  {best_caps[i]:>6.0f}")

    # ---------- C: Ranking quality ----------
    section("C. Ranking quality — do top-K predicted overlap top-K actual?")
    overlaps_10, overlaps_30 = [], []
    rhos = []
    for gw, g in rows.groupby("GW"):
        by_id = g.groupby("element").agg(pred=("pred", "sum"),
                                          actual=("total_points", "sum"))
        if len(by_id) < 30:
            continue
        top10_p = set(by_id.nlargest(10, "pred").index)
        top10_a = set(by_id.nlargest(10, "actual").index)
        top30_p = set(by_id.nlargest(30, "pred").index)
        top30_a = set(by_id.nlargest(30, "actual").index)
        overlaps_10.append(len(top10_p & top10_a) / 10)
        overlaps_30.append(len(top30_p & top30_a) / 30)
        rho, _ = spearmanr(by_id["pred"], by_id["actual"])
        if not np.isnan(rho):
            rhos.append(rho)
    print(f"  Top-10 overlap (model vs hindsight): "
          f"{100 * np.mean(overlaps_10):.0f}% on average")
    print(f"  Top-30 overlap (model vs hindsight): "
          f"{100 * np.mean(overlaps_30):.0f}% on average")
    print(f"  Spearman rank correlation (per GW, mean): {np.mean(rhos):.3f}")
    print(f"  → 0.0 = random ranking, 1.0 = perfect. >0.3 is meaningful; "
          f">0.5 would be strong.")

    # ---------- D: Starters vs bench ----------
    section("D. Starters (≥60 min) vs benchwarmers (<30 min)")
    starters = rows[rows["minutes"] >= 60]
    bench = rows[rows["minutes"] < 30]
    print(f"  All rows:  n={len(rows):>5}  MAE={mean_absolute_error(rows['total_points'], rows['pred']):.3f}")
    print(f"  Starters:  n={len(starters):>5}  MAE={mean_absolute_error(starters['total_points'], starters['pred']):.3f}  "
          f"pred mean={starters['pred'].mean():.2f}  actual mean={starters['total_points'].mean():.2f}")
    print(f"  Bench:     n={len(bench):>5}  MAE={mean_absolute_error(bench['total_points'], bench['pred']):.3f}  "
          f"pred mean={bench['pred'].mean():.2f}  actual mean={bench['total_points'].mean():.2f}")
    print(f"  → If headline MAE looks good because we predict bench=0 accurately, "
          f"the starter MAE is the one that actually matters for captain/XI choice.")

    # ---------- E: Initial squad — with AND without lookahead ----------
    section("E. Initial squad — predicted vs oracle (with & without lookahead)")
    # Oracle initial squad: pick the 15 with highest ACTUAL SEASON TOTAL
    season_totals = (
        df[df["season"] == SEASON]
        .groupby("element")
        .agg(name=("name", "first"), team=("team", "first"),
             position=("position", "first"),
             total=("total_points", "sum"),
             value0=("value", "min"))
        .reset_index()
    )
    pos_to_id = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
    team_ids = {t: i for i, t in enumerate(sorted(season_totals["team"].dropna().unique()))}
    oracle_pool = []
    for _, r in season_totals.iterrows():
        if pd.isna(r["position"]) or r["position"] not in pos_to_id:
            continue
        oracle_pool.append(Player(
            id=int(r["element"]), name=str(r["name"]),
            team_id=team_ids.get(r["team"], -1), team_name=str(r["team"]),
            position=pos_to_id[r["position"]],
            price=int(r["value0"]) if not pd.isna(r["value0"]) else 50,
            xpts=float(r["total"]),
        ))
    sel_oracle = pick_squad(oracle_pool)
    oracle_total = sum(p.xpts for p in sel_oracle.squad)
    pool_by_id = {p.id: p for p in oracle_pool}

    # Baseline: no lookahead
    pool_base = build_players_for_gw(feats, preds, SEASON, 2,
                                      ceiling_predictions=preds_p90)
    sel_base = pick_squad(pool_base)
    base_ids = {p.id for p in sel_base.squad}
    base_total = sum(p.xpts for p in oracle_pool if p.id in base_ids)

    # With lookahead — matches what the agent actually does
    horizon = _horizon_xpts_by_id(feats, preds, SEASON, 3, LOOKAHEAD_H)
    pool_la = build_players_for_gw(feats, preds, SEASON, 2,
                                    ceiling_predictions=preds_p90,
                                    horizon_xpts=horizon)
    sel_la = pick_squad(pool_la, lookahead_weight=LOOKAHEAD_W)
    la_ids = {p.id for p in sel_la.squad}
    la_total = sum(p.xpts for p in oracle_pool if p.id in la_ids)

    n_gws = 28
    print(f"  Oracle (hindsight) 15-man squad season total: {oracle_total:.0f}")
    print(f"  Predicted, no lookahead:                       {base_total:.0f}  "
          f"(gap {oracle_total-base_total:.0f} = {(oracle_total-base_total)/n_gws:.1f}/GW)")
    print(f"  Predicted, lookahead (h={LOOKAHEAD_H}, w={LOOKAHEAD_W}):              {la_total:.0f}  "
          f"(gap {oracle_total-la_total:.0f} = {(oracle_total-la_total)/n_gws:.1f}/GW)")
    print(f"  Lookahead lift on initial squad: "
          f"+{la_total-base_total:.0f} pts season ({(la_total-base_total)/n_gws:+.2f}/GW)")

    print(f"\n  Players lookahead ADDED:")
    for pid in sorted(la_ids - base_ids,
                       key=lambda i: -pool_by_id.get(i, Player(i,"?",0,"?",1,0,0.0)).xpts)[:6]:
        p = pool_by_id.get(pid)
        if p:
            print(f"    + {p.name[:24]:<24} {p.team_name[:18]:<18} "
                  f"season total {p.xpts:>4.0f}")

    print(f"\n  Players lookahead DROPPED:")
    for pid in sorted(base_ids - la_ids,
                       key=lambda i: -pool_by_id.get(i, Player(i,"?",0,"?",1,0,0.0)).xpts)[:6]:
        p = pool_by_id.get(pid)
        if p:
            print(f"    - {p.name[:24]:<24} {p.team_name[:18]:<18} "
                  f"season total {p.xpts:>4.0f}")

    # ---------- F: Feature importance ----------
    section("F. Feature importance — what drives predictions now?")
    importances = pd.Series(
        booster_mean.feature_importance(importance_type="gain"),
        index=booster_mean.feature_name(),
    ).sort_values(ascending=False)
    print(f"  Top 15 features by gain (mean model):")
    for name, gain in importances.head(15).items():
        bar = "█" * max(1, int(20 * gain / importances.max()))
        print(f"    {name:<28} {gain:>10.0f}  {bar}")
    print(f"\n  Fixture-difficulty features rank:")
    for fdf in ["opp_attack", "opp_defence", "team_attack", "team_defence"]:
        if fdf in importances.index:
            rank = list(importances.index).index(fdf) + 1
            print(f"    {fdf:<16}  rank {rank:>3}  gain {importances[fdf]:>10.0f}")


if __name__ == "__main__":
    main()
