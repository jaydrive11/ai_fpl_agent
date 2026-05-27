# Next Steps

Ranked by expected pts/GW lift × effort. Current state: **65.0 pts/GW**
3-season average (at top-10k cutoff). Goal: stretch toward 70+ for solid
top-10k consistency.

## Tier 1: Pure-engineering improvements (~+1-2 pts/GW each)

These don't require new data sources. ~1-3 days each.

### 1. ~~Smarter chip timing using fixture difficulty~~ DONE — negative result
**Status:** shipped, see DECISIONS milestone 11.
`_decide_chip` now uses FDR-aware "peak remaining" logic (90% tolerance +
floor) instead of fixed thresholds. Chips fire on intuitively better GWs
(mid-/late-season DGWs vs early-season noise), but total scoring barely
moved (±0.2 /GW). Chip budget is structurally capped; smarter timing within
that cap helps marginally, not dramatically.

The original spec is preserved below for historical context:

**Specific changes (originally planned):**
- **Wildcard:** instead of "fire when constrained pick takes 2+ hits", fire
  on a detected *fixture turn* — when the next 3-5 GWs of FDR scores diverge
  sharply from the previous 3-5 (i.e., your current players have hard
  fixtures coming, new ones would have easier).
- **Triple Captain:** save TC for the GW with the highest *expected* captain
  ceiling across the remaining season — use the lookahead's future
  predictions to identify it. Currently TC fires on the first GW above a
  fixed threshold (typically GW3 Haaland/Salah), wasting it on average weeks.
- **Bench Boost:** fire when bench's predicted total + FDR-favorable is in
  the top decile across remaining GWs.

**Why this works now:** the FDR features are already in the model;
heuristics just need to consume them.

**Files to touch:** `src/fpl_agent/backtest.py::_decide_chip`.

---

### 2. Auto-substitutes in scoring (~+0.5-1 /GW)
**Why:** Currently `_score_xi` sums only the starting XI's actuals. In
real FPL, when a starter scores 0 (didn't play), the highest-priority bench
player who DID play takes their slot automatically. We're missing ~1-2
pts/GW from these auto-subs.

**Implementation:**
- Bench needs a priority order (currently we just have a set).
- Modify `_score_xi`: for each starting XI player who scored 0, sub in the
  highest-priority bench player who scored > 0 and respects formation
  validity.

**Files to touch:** `src/fpl_agent/backtest.py::_score_xi`, possibly
`Selection.bench` to be an ordered list.

---

### 3. Cross-season player priors (~+0.5-1 /GW)
**Why:** New signings (transfers between EPL clubs, or new arrivals) appear
in GW1 of a new season with no in-season history. All their `*_lag1` and
`*_avg5` features are NaN — the model effectively predicts them at the
position average.

**Implementation:**
- For each (player_name, season): join player's previous-season aggregates
  (total minutes, total points, goals, assists) as features prefixed `prior_*`.
- LightGBM handles NaN for first-time-ever players.

**Files to touch:** `src/fpl_agent/features.py` (new join function).

---

### 4. Pre-season initial squad strategy (~+1 /GW averaged season)
**Why:** Currently `start_gw=2` in backtests because GW1 has no in-season
history. In real FPL deployment, you have to pick a squad *before* GW1.

**Implementation:** Use prior-season aggregates + position-position priors as
features for "GW0 prediction". Use the highest-confidence projections to
pick the initial 15. Effectively: just retrain on all-but-current-season +
predict using only prior-season features.

This dovetails with #3.

---

## Tier 2: Modeling improvements (~+0.5-1 /GW each)

### 5. Calibrate the Stage 1 classifier
**Status:** Stage 1 outputs raw LightGBM probabilities. They're informative
for ranking but may not be well-calibrated for the multiplication step.

**Fix:** Apply Platt scaling or isotonic regression on the validation set.
Or use `objective='cross_entropy'` and check the resulting calibration.

**Estimated lift:** +0.5 /GW — multiplications get more accurate.

### 6. Per-position sub-models for Stage 2
**Status:** One stage-2 regressor for all positions.

**Idea:** Train separate Stage 2 models per position (GK, DEF, MID, FWD).
GK scoring is fundamentally different from MID — different feature
importances would emerge.

**Risk:** Less training data per model. Need to validate it doesn't overfit.

---

## Tier 3: External data sources (~+1-4 /GW each, but bigger projects)

These break the "modeling only" barrier and are the path toward 70+ pts/GW.

### 7. News/lineup scraping (~+2-4 /GW)
**Why:** The captain gap (~20 pts/GW remaining) is mostly about predicting
*hauls* — which players are due a return today. The model can't see:
- Penalty/free-kick taker changes (huge for predicting goals)
- Rotated/rested starters (manager presser hints)
- Injury news the day before kickoff
- Form against specific opposition (matchup-aware)

**Implementation:**
- Scrape FPL Twitter accounts, press conferences, BBC/Sky lineup news.
- Cache to local DB; join into features as `confirmed_lineup`,
  `is_penalty_taker`, `injury_doubt`, etc.

**This is the single biggest remaining lever** for a fully-automated agent.
~1-2 weeks of work for a first version.

### 8. Set-piece taker classification (~+0.5-1 /GW)
**Subset of #7**, but tractable independently. FPL community maintains
spreadsheets of confirmed/probable set-piece takers; scraping one of those
plus joining as a feature would catch corner/free-kick goal upside.

### 9. xG over/underperformance feature (~+0.5-1 /GW)
**Why:** A player whose recent xG (expected goals) exceeded their actual
goals is "due" — likely to convert in the next few weeks. The reverse for
overperformers regressing to mean.

**Implementation:** rolling diff `xG_avg5 - goals_avg5`; feed as feature.
We have xG in data (post-2022); just need the diff feature.

**This is a feature engineering win that doesn't need external data.**
Could be Tier 1 — moved here because it's slightly more speculative.

---

## Tier 4: Architectural / infrastructure (no direct /GW lift)

### 10. Live FPL API integration
**Status:** `data.py::fetch_bootstrap()` exists but isn't called from
production paths.

**Needed for actual play:** at start of each gameweek, fetch:
- Current squad (via authenticated `/api/my-team/{team_id}/`)
- Latest player prices + ownership
- Upcoming fixtures
- Then run predict → optimize → execute (the `/api/transfers/` POST).

~1-2 days of work, including auth flow.

### 11. Proper multi-GW MILP (replaces lookahead heuristic)
**Current:** lookahead is a heuristic (squad-bonus term in single-GW MILP).
**Proper:** joint optimization over (squad, XI, captain, transfers) for the
next H GWs simultaneously. Variables grow ~10x; CBC may struggle.

**Estimated lift:** +0.5-1 /GW over the current heuristic. The heuristic
captures most of the value already — see milestone 8 diagnostic.

### 12. Continuous evaluation loop / dashboards
- Track each weekly pick + actual outcome
- Per-player prediction calibration over time
- A/B test new features before merging
- Slack/email summary at GW end

Pure infra. Useful for live deployment, not for backtest results.

---

## What we deliberately *won't* do

These were considered and rejected, with reasons (so we don't reconsider
without new info):

- **Higher-α quantile models (P95, P99) for captain** — alpha sweep showed
  they pick *worse* captains, overfitting to outliers. P90 is the sweet
  spot. See DECISIONS milestone 9b.
- **Deep learning models** — for tabular data with ~250k rows and ~30
  features, LightGBM is competitive with or better than DL. Not worth
  the complexity.
- **Reinforcement learning over the whole season** — state space too large,
  reward signal too sparse, way too high variance. The MILP formulation is
  much more tractable for what's essentially a constrained optimization
  problem.
- **Web scraping community team-of-the-week consensus** — borderline
  ToS-questionable. Stick to first-party sources.

---

## Recommended attack order

If you have **1 day**: do #1 (chip timing using FDR) + #2 (auto-subs) —
~+1.5-3 pts/GW combined.

If you have **1 week**: above + #3 (cross-season priors) + #5 (calibrate
stage 1) + #9 (xG over/underperformance) — should push to ~67-68 /GW.

If you have **1 month**: add #7 (news/lineup scraping) — realistic stretch
goal is **70-72 pts/GW** which is consistent top-1k territory.

If you have **3 months and want to go live**: #10 (live API integration) +
weekly monitoring + ongoing data-quality work.
