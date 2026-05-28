# Next Steps

**Current state:** 65.3 pts/GW 3-season average (top-10k cutoff).
**Goal:** deploy as a **human-in-the-loop (HITL)** weekly recommender for the
**2026-27 season** (typically starts mid-August 2026).

The 2025-26 season is over, so there's no rush — months to ship cleanly.

---

## Deployment posture: HITL (decided)

The agent will **recommend** weekly moves; user clicks them in the FPL UI.
Avoids FPL's ToS gray area around automated transfer submission. Captures
~99% of the value (the prediction + decision is the hard part). See
DECISIONS.md "HITL deployment posture" for full reasoning.

Each week:
1. Cron triggers `scripts/recommend.py`
2. Fetches fresh data from FPL's public API + your `team_id`
3. Predicts xPts and runs the optimiser with your real squad as input
4. Delivers a one-pager (markdown / email / Slack): transfers, captain, XI, bench
5. You spend ~30s reviewing and submit manually in the FPL UI

**No credentials needed.** FPL's public endpoints let us READ any user's team
by ID without authentication. The user retains full account control.

---

## Cumulative scorecard (what's already done)

| Milestone | Multi-season avg /GW |
|---|---:|
| 6 (L2 + P90) | 55.7 |
| 8 (lookahead) | 60.9 |
| 9 (fixture difficulty) | 62.0 |
| 10 (two-stage hurdle) | 65.0 |
| 11 (smart chips — neutral) | 64.9 |
| 12 (auto-subs) | **65.3** |

Full details in DECISIONS.md.

---

## What's pending — by deployment phase

### Phase 1: Critical (can't deploy without these) — ~3-4 days

These are the minimum to produce real weekly recommendations.

| # | Component | Effort | Status |
|---|---|---|---|
| 1 | **Live FPL API client** — fetch `bootstrap-static`, `entry/{id}/event/{N}/picks/`, `entry/{id}/history/` | 3-4h | Stub exists in `data.py`; not wired up |
| 2 | **Upcoming-GW feature builder** — synthesise GW N+1 rows using actuals thru GW N + fixture metadata. Backtest cheats by using each row's pre-built features; live needs to CONSTRUCT them carefully without leakage | 4-6h | Not built |
| 3 | **`scripts/recommend.py`** — orchestrator: fetch → predict → optimise → markdown output | 3-4h | Not built |
| 4 | **Config file** (`fpl-config.toml`): `team_id`, preferences | 1h | Not built |
| 5 | **Cross-season player features** — `is_first_pl_season`, `prior_season_total_points`, `prior_season_minutes`, `prior_season_goals`, `prior_season_assists`, `new_team_this_season`, `team_is_newly_promoted` (name-matched across seasons; handles transfers A/B, promoted teams C — see DECISIONS "Season-transition handling") | 2-3h | Not built |
| 6 | **Annual rebuild workflow** (`scripts/season_rollover.py`) — pulls new season's Vaastav data, retrains, runs validation backtest | 3-4h | Not built |
| 7 | **Initial squad mode for GW1** — no in-season form; use cross-season priors + value + position + team strength | 3-4h | Not built |

### Phase 2: High-value polish (should ship before going live) — ~1 day

| # | Component | Effort | Status |
|---|---|---|---|
| 8 | **Audit journal** — save every weekly rec; later compare to actuals; track agent quality over time | 2h | Not built |
| 9 | **Confidence signals in output** — flag high-uncertainty weeks (many new signings, manager change, GW1, etc.) so user knows when to override | 2h | Not built |
| 10 | **Data freshness check** — refuse to run if Vaastav data > 7 days old | 1h | Not built |
| 11 | **Pre-deployment validation script** — pick squad for last GW of 2024-25; manually compare to top managers' picks for sanity check | 2h | Not built |
| 12 | **README "Live deployment" section** | 1h | Not built |

### Phase 3: Automation + post-launch iteration — ongoing

| # | Component | Effort | Status |
|---|---|---|---|
| 13 | Email/Slack delivery (not just stdout) | 3-4h | Not built |
| 14 | Cron/scheduler setup | 1h | Not built |
| 15 | Diff vs last week in output | 2h | Not built |
| 16 | Error handling + alerting (Sentry-like) | 2-3h | Not built |
| 17 | Model improvements: xG over/underperformance, Stage 1 calibration, per-position Stage 2 | days | Layered post-deployment |
| 18 | **News/lineup scraping** (Tier 3 — biggest remaining /GW lever) | 1-2 weeks | The major upgrade after deployment is stable |

---

## Already-shipped items (for reference)

These earlier `NEXT_STEPS.md` items were completed and folded into the
codebase:

- ~~Smarter chip timing using fixture difficulty~~ — milestone 11 (neutral result)
- ~~Auto-substitutes in scoring~~ — milestone 12 (+0.4 /GW agent, +2.3 /GW static)

---

## What we deliberately *won't* do

Considered and rejected, with reasons:

- **Fully autonomous transfer submission** — ToS gray area; HITL captures ~99% of value
- **Higher-α quantile models (P95, P99) for captain** — alpha sweep showed they pick worse captains, overfitting to outliers (DECISIONS milestone 9b)
- **Deep learning models** — tabular ~250k rows × ~30 features; LightGBM is competitive, much simpler
- **Reinforcement learning over the whole season** — state space too large, reward signal too sparse, much higher variance than MILP
- **Web-scraped community team-of-the-week consensus** — borderline ToS; stick to first-party sources

---

## Recommended timeline

Roughly 2.5 months from "now" to deployable agent, with substantial buffer.

| When | What |
|---|---|
| Next ~6 weeks | Phase 1 (#1–#7) — cold-start features + live API + recommend script + rollover/initial-squad modes |
| ~1 week after | Phase 2 (#8–#12) — audit journal, confidence signals, validation, docs |
| Late July | Vaastav publishes 2026-27 folder → run `season_rollover.py`; final backtest validation |
| Early August (pre-GW1) | Generate initial squad recommendation; review carefully |
| Mid-August (GW1 onwards) | Weekly HITL recommendations |
| Throughout season | Iterate on Phase 3 items; learn from audit journal |

---

## Where to start

**Phase 1 item #1 (Live FPL API client).** Reasons:

- Everything downstream depends on it
- Independently testable (just hits public endpoints, no model needed)
- Validates that we can reliably talk to FPL before building anything that needs it
- Concrete and well-scoped (~3-4 hours)

After #1: **#2 (upcoming-GW feature builder)** is the only conceptually-new
piece in Phase 1. Everything else (recommend.py, config, cross-season features,
rollover script, initial-squad mode) is wiring up things we already have.
