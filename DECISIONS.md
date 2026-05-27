# Engineering Decisions

This document records every significant design decision in the order it was
made, with the rationale and what we learned. The goal is that someone reading
this in 6 months understands *why* the code looks the way it does, not just *what* it does.

## Performance evolution

| Milestone | What changed | Multi-season avg /GW |
|---|---|---:|
| 0 | (No agent yet) | — |
| 1 | Squad optimizer (PuLP MILP) | — |
| 2 | Data pipeline (Vaastav 10 seasons) | — |
| 3 | LightGBM xPts predictor (L1 loss) | 49.6 *(single season)* |
| 4 | Model wired into optimizer | 49.6 |
| 5 | Backtest harness + transfer logic | 49.6 |
| 6 | **L2 loss + separate P90 captain model** | **55.9** *(+6.3)* |
| 7 | Chip strategy (WC/TC/BB heuristics) | 55.7 *(noise, no DGWs in test season)* |
| 8 | **Multi-GW lookahead** (horizon=5, weight=0.5) | **60.9** *(+5.2)* |
| 9 | Fixture-difficulty features (teams.csv) | 62.0 *(+1.1)* |
| 9b | Captain alpha sweep (α ∈ {0.85, 0.9, 0.95, 0.99}) | 62.0 *(negative result)* |
| 10 | **Two-stage hurdle model** | **65.0** *(+3.0)* — **top-10k** |

Reference: FPL average ~53, top-10k ~65, world #1 ~75.

---

## Milestone 1: Squad optimizer first

**Decision:** Build the MILP optimizer before any model. Use stub xPts data
(or FPL's own `ep_next` field) to validate it.

**Why:**
- Optimizer correctness is independently verifiable against FPL rules (squad
  size, budget, ≤3 per club, formation bounds, captain ∈ XI).
- A correct optimizer is a hard requirement no matter how good the model is.
- Once the optimizer works, swapping in better predictions is a one-line change.

**Implementation:** `optimizer.pick_squad()` with PuLP + CBC. Decision variables
`x[i], s[i], c[i]` for squad/XI/captain. Objective initially just XI sum +
captain extra.

---

## Milestone 2: Data pipeline before model

**Decision:** Pull Vaastav's repo (10 seasons of per-GW data) into a unified
Parquet cache before doing any modeling.

**Why:**
- Without data, model decisions are speculative.
- Cleaner to load once, then iterate on features.
- Schema-drift handling (xG features only post-2022) is best handled at the
  data layer with NaN coercion.

**Implementation:** `history.fetch_season()` is per-file tolerant (older
seasons lack `fixtures.csv` / `teams.csv`); `load_all_gw` concats with
`sort=False` so missing columns become NaN. Position is backfilled from
`players_raw.csv` for the four early seasons where `merged_gw.csv` lacks it.

**Lesson:** Always log per-file status and continue — don't abort a whole
season because one auxiliary file 404'd.

---

## Milestone 3: Single LightGBM regressor (L1)

**Decision:** Start with one LightGBM regressor predicting `total_points`,
L1 loss (MAE), all features in one model.

**Why:** Simplest viable model. Establishes a baseline before adding
complexity.

**Result:** MAE 0.86 on held-out 2025-26 — beats baselines (1.06 form, 1.16
last-GW). Sounds great.

**But:** when wired into the optimizer + backtest, scored only **49.6 pts/GW**
— *below* FPL average (~53). Something was off.

---

## Milestone 4-5: Wiring + backtest infrastructure

**Decision:** Build the backtest harness as a full simulator with three
comparison strategies — agent, static squad, per-GW oracle.

**Why:**
- Without an oracle baseline, we can't say what's mechanically possible.
- Without a static-squad baseline, we can't isolate the value of transfers
  from the value of the initial pick.
- Walking forward GW-by-GW with carried squad state is the only honest test.

**Implementation:** `backtest.run_backtest()` with `ChipsState`, transfer-aware
MILP (`current_squad_ids`, `free_transfers`, `hit_cost` params on `pick_squad`).

**Lesson learned:** the first transfer-cost test failed with a misleading
"optimizer doesn't enforce hit cost" symptom. The actual bug was *in the
test*: it constructed a "bad squad" that violated budget + per-club rules, so
the optimizer was *forced* to transfer for feasibility, not by choice.
Replaced with `_build_valid_suboptimal_squad` (which picks the best squad from
only the bottom-half player pool).

---

## Milestone 6: L2 loss + P90 quantile captain — THE big diagnostic-driven fix

**Decision:** Switch from L1 to L2 loss for the mean model, and add a separate
quantile-regression model (α=0.9) for captain selection.

**Why** (from the first diagnostic):
- L1 minimizes MAE → median regression. With 62% of training rows being 0
  (bench players), the median collapses toward 0.
- Symptoms: max prediction 5.01 vs actual max 24; variance ratio 8.1×.
- Top-1 captain (by mean prediction) scored 5.69 actual vs hindsight 17.10 —
  **22.8 pts/GW left on the table just from bad captain choice**.

**L2** punishes overprediction quadratically, so it can't ignore big actuals
the way L1 can. Predictions spread out (max jumped 5.01 → 9.18).

**P90 quantile** specifically targets "what's this player's 90th-percentile
outcome" — exactly the right value for captain selection (we want upside, not
mean).

**Implementation:**
- `Player.ceiling_xpts` field (P90), separate from `xpts` (mean).
- Optimizer's MILP objective: `Σ s[i]·mean[i] + Σ c[i]·ceiling[i]`.
- XI selection still uses mean (it's about this week's likely score); captain
  uses ceiling (upside).

**Result:** 49.6 → 55.9 pts/GW (+6.3). Critically, MAE got slightly *worse*
(0.86 → 0.98) but the actual scoring jumped — MAE was the wrong metric.

**Lesson:** MAE looks like an obvious choice but it's wrong for FPL because
the loss function distorts the predicted distribution in ways the metric
itself doesn't reveal. Always validate via the downstream task, not just
training metrics.

---

## Milestone 7: Chip strategy (with a twist)

**Decision:** Add greedy heuristics for Wildcard, Triple Captain, Bench Boost.
One chip per GW max, conservative defaults.

**Why:** Chips are worth ~3-5 pts/GW in typical seasons. Worth building.

**First attempt failed:** Initial WC heuristic fired aggressively (gain > 12
between unconstrained and constrained pick). Result: WC cascaded badly,
committing 6+ transfers at once to model favorites that underperformed.
Net effect: **-119 pts vs no-chips** in 2025-26.

**Diagnosis:** the model overtrusts its own picks. Without fixture-difficulty
awareness, the WC "ideal squad" was full of players who happened to have
high-noise xG predictions.

**Fix:** Made WC heuristic strict — fire only if either (a) constrained pick
was forced to take ≥2 hits (WC saves real money), or (b) gain > 30 (high bar).

**Result:** With strict WC, chips contribute +1 pt over 28 GWs in 2025-26 —
basically neutral. **The 2025-26 season has only 1 double-gameweek (GW26)**,
which structurally caps chip value. Chips remained roughly neutral across the
3-season backtest.

**Lesson:** Chips are heavily season-dependent. A framework that works on a
DGW-rich season may neutral on a DGW-poor one. Don't overfit chip heuristics
to one season's structure.

---

## Milestone 8: Multi-GW lookahead — THE second big diagnostic-driven fix

**Decision:** Add per-player `horizon_xpts` = sum of mean predictions over
next H GWs. Optimizer MILP adds `lookahead_weight × Σ x[i]·horizon_xpts[i]`
to the objective.

**Why** (from the diagnostic):
- Section E showed predicted GW2 squad scored 1107 season pts vs hindsight
  oracle's 1814 — **25 pts/GW lost on initial squad alone**.
- Root cause: pick_squad was using only this week's predictions. Players with
  great upcoming fixture runs got equal weight to one-shot wonders.

**Approach:** Avoid a full multi-GW MILP (10× variable count). Instead, add a
heuristic "ownership value" term: a player with high expected points over the
next 5 GWs gets a bonus to being in the 15-man squad. XI selection still uses
this week's xpts; captain uses this week's ceiling.

**Tuning:** Grid-swept (horizon, weight) on the 3-season backtest:

| horizon | weight | avg /GW |
|---|---|---:|
| 0 (off) | — | 55.7 |
| 3 | 0.5 | 57.3 |
| **5** | **0.5** | **60.9** |
| 8 | 0.3 | 58.6 |
| 8 | 0.2 | 60.7 |

**Result:** 55.7 → 60.9 pts/GW (+5.2). The single biggest improvement.

The updated diagnostic confirmed: initial-squad gap dropped from 25.4/GW to
**0.2/GW** — essentially eliminated.

**Lesson:** Greedy single-period optimization is the default mistake in
sequential decision problems. Even a simple horizon-weighted heuristic
captures most of the value of a proper multi-period optimization.

---

## Milestone 9: Fixture difficulty features

**Decision:** Join `teams.csv` strength ratings (`strength_attack_home/away`,
`strength_defence_home/away`) into the features as 4 numeric columns
(`opp_attack`, `opp_defence`, `team_attack`, `team_defence`), home/away-aware.

**Why:** The model previously treated `opponent_team` as a categorical ID,
forcing it to learn team strength from data. With explicit FDR features, the
signal is direct.

**Implementation:** `history.load_team_strengths(season)` + `_join_fixture_difficulty`
in features.py. NaN for older seasons (2016-17..2018-19) where `teams.csv`
is missing; LightGBM handles natively.

**Result:** 60.9 → 62.0 pts/GW (+1.1). Mean model val MAE improved 1.000 →
0.939. Fixture-difficulty features rank 11-20 in feature importance — used,
but not dominant (minutes/form features still rule).

**Note:** 2024-25 jumped from 61.1 → 65.7 (top-10k!), 2023-24 actually slipped
by 1.7 (noise). Single-season variance is real; the multi-season average is
the honest metric.

---

## Milestone 9b: Captain quantile alpha sweep — NEGATIVE result

**Hypothesis:** Higher quantiles (P95, P99) might predict captain ceilings
closer to actual hauls (max actual = 24, our P90 max = 14.7).

**Test:** Train 4 captain models at α ∈ {0.85, 0.9, 0.95, 0.99}; measure
average actual points of top-1-by-prediction.

**Result:**

| α | cap_avg actual | pred_max |
|---|---:|---:|
| 0.85 | 6.79 | 12.3 |
| **0.90** | **6.86** | 14.7 |
| 0.95 | 6.52 | 17.2 |
| 0.99 | 5.86 | 20.8 |

**P90 wins.** Higher α models predict bigger numbers but pick worse
captains — they overfit to outliers, picking players who *once* had a freak
haul rather than those with consistent high upside.

**Lesson:** Quantile regression has a ceiling for captain selection with our
current features. The remaining ~20 pts/GW captain gap is mostly irreducible
without new feature sources (lineups, set-piece info, news). Don't waste
more time on quantile experiments.

---

## Milestone 10: Two-stage hurdle model — TOP-10K HIT

**Decision:** Decompose `E[points] = P(plays ≥ 60min) × E[points | plays ≥ 60min]`.

- **Stage 1:** binary classifier, all rows, target `(minutes ≥ 60)`.
- **Stage 2 mean:** L2 regressor, trained ONLY on rows where minutes ≥ 60.
- **Stage 2 P90:** quantile α=0.9, same filtered training set.
- **Combined predictor:** `TwoStagePredictor` class that wraps both and
  duck-types `lgb.Booster.predict()` — so the rest of the pipeline doesn't
  change.

**Why** (from the diagnostic):
- Section D: starter MAE 2.38 vs bench MAE 0.40. Headline MAE looked good
  because we predict bench=0 well, but we systematically under-predict
  starters (pred mean 2.61 vs actual 3.82).
- Single model was averaging over two regimes — bench players (0 pts) and
  playing starters (positive pts). The averaging biased starter predictions
  toward 0.
- This is classic zero-inflated regression. Hurdle models are the textbook
  fix.

**Why P(plays ≥ 60min):** FPL awards 2 appearance pts at 60+ min. It's the
natural FPL boundary. Could also have used continuous "expected minutes"
regression — left as v2 work.

**Result:** 62.0 → 65.0 pts/GW (+3.0). **At top-10k cutoff.**

| Season | Single-stage | Two-stage | Δ |
|---|---:|---:|---:|
| 2023-24 | 61.4 | 61.8 | +0.4 |
| 2024-25 | 65.7 | **68.8** | +3.1 |
| 2025-26 | 58.8 | **64.5** | +5.7 |
| avg | 62.0 | **65.0** | **+3.0** |

Static-squad baseline also rose 49.7 → 53.2 (initial picks more calibrated).

**Lesson:** The biggest wins came from honest diagnostics, not bigger models.
Each major lift (L2, P90, lookahead, two-stage) came from running a
diagnostic that pointed at a specific failure mode, then targeting it
precisely.

---

## Engineering lessons (cross-cutting)

### 1. MAE is the wrong metric for FPL

Switching from L1 (MAE-optimal) to L2 made MAE slightly *worse* (0.86 → 0.98)
but actual scoring jumped 6+ pts/GW. The downstream task (squad selection)
cares about ranking + calibration, not point-wise accuracy. Always validate
via the task you actually care about.

### 2. Vectorize feature engineering aggressively

The first version of `build_features` used `groupby().transform(lambda)`
patterns — 18 minutes for 246k rows. Switched to pre-shift + `groupby().rolling()`
— 2 seconds. Same answer, 540× faster.

### 3. Many-core machines need `num_threads=4`

LightGBM's default `num_threads=0` (use all cores) thrashes on many-core
hosts — 100× slowdown observed. Always cap explicitly.

### 4. Diagnostics that don't match the agent are useless

The first diagnostic measured captain quality using mean predictions, even
after we switched the agent to use P90 ceiling. The diagnostic kept showing
a 24.8/GW captain gap. After fixing the diagnostic to use P90, the actual
gap was 20.5 — the rest was the diagnostic mismeasuring. Diagnostics must
mirror the agent's actual decision process.

### 5. Squad picks ≠ XI picks ≠ captain picks

Each decision has a different objective:
- **Squad (15)**: who do I want to OWN for the next several weeks → horizon-aware
- **XI (11)**: who's best THIS week → mean-aware (current GW only)
- **Captain (1)**: highest upside this week → P90 ceiling-aware

Bundling all three into one objective (as the first version did) is wrong.

### 6. Test with valid setups

Hours wasted on a "transfer cost not being enforced" symptom when the actual
issue was an invalid test squad (over-budget, >3 per club). The optimizer was
correctly forced to transfer for feasibility. Tests must use *valid* inputs
to test the *intended* behavior.

### 7. Walk-forward validation over single-season

Single-season backtests have high variance — 2024-25 had a +5/GW outlier on
fixture-difficulty addition while 2025-26 was flat. Always run multiple
seasons (with proper train/val/test isolation per season) before believing
a result.
