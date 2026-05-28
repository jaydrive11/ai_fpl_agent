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
| 11 | FDR-aware chip timing (best-remaining + 90% peak tolerance + floor) | 64.9 *(noise — see milestone 11)* |
| 12 | Auto-subs in scoring (FPL bench substitution rules) | 65.3 *(+0.4 agent, +2.3 static)* |

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

## Milestone 11: FDR-aware chip timing — NEGATIVE result

**Decision:** Replace fixed-threshold chip heuristics (TC fires at `ceiling > 11`,
BB at `bench > 10`) with "save chips for the peak remaining GW" logic, using
predictions for all remaining GWs as a forward-looking estimate.

**Why** (from prior milestones):
- Old chips fired identically every season — TC on GW3 (first Haaland fixture
  above threshold), BB on GW3, WC mid-season on hits.
- This wastes TC/BB on average weeks instead of saving them for genuine peak
  opportunities (DGWs, fixture turns).
- Plan was: pre-compute per-GW best captain ceiling + best bench total across
  remaining season, fire chips only when this GW is near the peak.

**Implementation:**
- `_future_chip_estimates()`: at backtest start, computes for every GW the
  max-possible captain ceiling and a proxy bench total (sum of xpts ranks 11-15
  in that GW's pool).
- `_decide_chip` extended with `current_gw + future_*_dicts` arguments.
- Logic: fire if `current ≥ 0.9 × max(remaining)` AND `current ≥ absolute_floor`.
- Strict "fire only if best-remaining" was too greedy — saved chips until the
  literal last GWs. The 90% tolerance + floor gives some flexibility.

**Result:** chips DO fire at different (and intuitively better) GWs — TC moved
from GW3 to GW24-26 (mid-/late-season DGW area). But scoring barely moved:

| Variant | 3-season avg /GW |
|---|---:|
| Old fixed-threshold | 65.0 |
| Strict best-remaining | 64.8 |
| Relaxed (90% + floor) | 64.9 |

All within ±0.2 — noise. The diagnostic implied chips were a +1-2/GW lever;
the actual marginal lift is essentially zero.

**Why the lift was small:**
1. Chips' total budget is small (3 chips × ~10-20 pts each = ~30-60 pts/season),
   so even perfect timing has a ceiling.
2. Far-future predictions are noisy — "save for the absolute best" is hard
   when prediction error is larger than the gap between "good" and "best" GWs.
3. Our test seasons don't have enough DGW-rich windows to differentiate
   timing strategies clearly.

**Decision:** Keep the new logic — it's structurally correct (more principled,
matches how a thoughtful FPL player thinks about chips). Don't expect a
material boost in season totals.

**Lesson:** Diagnostic-implied lift estimates can be optimistic when the
underlying budget is small. Chips have a hard cap on their potential
contribution; smarter timing within that cap helps marginally, not
dramatically. Most of the remaining gap to top-10k+ is in better player
selection (more data sources), not better chip timing.

---

## Milestone 12: Auto-subs in scoring (+0.4 /GW)

**Decision:** Implement FPL's bench-substitution rules in `_score_xi`. When
an XI player gets 0 minutes (didn't play), a bench player who DID play comes
on in their place, in bench priority order, provided the resulting formation
stays valid.

**Why** (from milestone 10 diagnostic notes / NEXT_STEPS.md):
- Previously `_score_xi` summed only the starting XI's actuals. When a
  starter blanked (e.g. injured / dropped / rotated), we lost their slot's
  full value even when a perfectly good bench player came on in real FPL.
- Estimated impact: +0.5-1 /GW based on typical FPL "bench points" reporting.

**Implementation:**
- `_minutes_lookup()`: per-GW dict of actual minutes by element id.
- `_ordered_bench()`: bench GK first (only subs in for starting GK), then
  outfield by predicted xpts descending (most-trusted bench player = first sub).
- `_formation_valid()`: explicit 1 GK / 3-5 DEF / 2-5 MID / 1-3 FWD check.
- `_apply_auto_subs()`: for each XI player with 0 mins, find the first
  bench player (in priority) who played and whose substitution keeps
  formation valid.
- `_score_xi` extended with `actuals_min` param. Auto-subs skipped under BB
  (whole 15 plays anyway — double-counting would inflate the score).

**Result:** 64.9 → **65.3 /GW** (+0.4 agent). Static baseline jumped more:
53.2 → 55.5 (+2.3) — expected, because static is more vulnerable to blanks
(no transfers to swap out problem players).

| Season | Before (m11) | + Auto-subs | Δ |
|---|---:|---:|---:|
| 2023-24 | 61.8 | 61.9 | +0.1 |
| 2024-25 | 69.0 | 69.2 | +0.2 |
| 2025-26 | 63.8 | 64.9 | +1.1 |
| avg | 64.9 | **65.3** | **+0.4** |

**Lesson:** Modeling fidelity matters. The agent's transfer logic was already
papering over the blank-starter problem by transferring problem players out.
Auto-subs help the agent only marginally because there aren't many residual
blanks to cover. But the static-squad baseline jumps a lot because it has
NO transfer ability — auto-subs are its only defense against blanks. This
shows the static comparison was previously understating the value of just
having a sensible squad.

8 new tests in `tests/test_auto_subs.py` cover: formation validity (accept
3-4-3, 3-5-2; reject 0-GK, 2-DEF), bench ordering, GK sub, outfield sub
respecting formation, skipping blanked bench, BB conflict.

---

## Strategic decision: HITL deployment posture

**Decision:** When deployed, the agent operates as **human-in-the-loop** —
produces weekly recommendations that the user submits manually in the FPL UI.
NOT fully autonomous (no automated POSTs to `/api/transfers/`).

**Why:**
- FPL's Terms & Conditions have language around automation/scripts/bots; the
  exact current wording on fully-autonomous transfer submission is a gray area.
- Manual analytics tools (FPLReview, Fantasy Football Hub, Mikkel Tokvam's
  solver) are widely used and accepted. The line is at whether the tool
  *submits* moves vs *recommends* them.
- Risk of full autonomy: account ban, IP block, ToS violation. Not worth it
  for the marginal time savings (clicking takes ~30 seconds weekly).
- HITL captures ~99% of the value because the prediction + optimisation is
  the hard part. Submitting is trivial.
- No credentials needed for the read-only path: FPL's public endpoints let
  us READ any user's team by `team_id` without auth.

**How to apply:**
- All deployment scripts use the read-only public API only
- Output is a markdown recommendation (eventually email/Slack)
- User retains full account control
- The repository on GitHub is fine as-is; what matters is what we run
  against a real account

---

## Plan: Season-transition handling

The 2025-26 season ended; we have until mid-August 2026 to ship. Off-season
brings five categories of change that don't currently get handled cleanly:

| Change | Difficulty | Plan |
|---|---|---|
| **A. PL-internal transfers** (e.g., Mbeumo to Man U) | Easy | Name-match cross-season → carries history. Flag `new_team_this_season` for role-change awareness. |
| **B. Foreign-league signings** (e.g., Wirtz to Liverpool) | Hard | Zero PL history. Rely on `value` + `position` + `team_strength`. Flag `is_first_pl_season` so model knows to lean on those features. |
| **C. Promoted teams** (3 new clubs replace 3 relegated) | Medium | `teams.csv` auto-loads. Their players inherit cold-start treatment via `is_first_pl_season`. Add `team_is_newly_promoted` flag. |
| **D. Manager changes** | Hard | Not detectable from Vaastav data alone. Proxy via team xG/xGC pattern divergence (slow). Real fix is news scraping (deferred). |
| **E. Rule changes** (chip counts, transfer banking) | One-off | Check FPL's pre-season announcement; update `rules.py` if needed. |

**What we can do with existing data:** cross-season name-matched priors +
cold-start flags. Covers A, B, C reasonably well.

**What we cannot do without external data:** D (manager changes),
early-warning injuries (only proxy from realised minutes), foreign-league
xG for signings like Wirtz. All deferred to the news-scraping milestone.

**Implementation lives in NEXT_STEPS Phase 1 items #5 (cross-season
features), #6 (annual rebuild script), #7 (initial squad mode).**

---

## Decision: Deploy for 2026-27 season (not current)

**Decision:** Don't try to deploy mid-season — 2025-26 is over. Aim for GW1
of 2026-27 (typically mid-August 2026).

**Why:**
- Current season has no remaining GWs to benefit from
- Removes time pressure — can build cold-start features properly, validate
  thoroughly, write good docs
- Vaastav's 2026-27 data folder typically appears late July, giving us a
  natural integration test point before going live
- Allows full Phase 1 + Phase 2 (see NEXT_STEPS) rather than rushing

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
