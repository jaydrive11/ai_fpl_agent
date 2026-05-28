# AI FPL Agent

A Fantasy Premier League decision agent. Predicts per-player gameweek points,
picks 15-man squads under all FPL constraints (£100m budget, formation,
≤3 per club), plans transfers with hit-cost awareness, plays chips
strategically (Wildcard, Triple Captain, Bench Boost), and applies FPL
auto-sub rules in scoring.

**Current performance:** multi-season walk-forward backtest averages
**65.3 pts/GW** — at the top-10k cutoff for the global FPL leaderboard.

| Benchmark | pts/GW |
|---|---:|
| FPL global average | ~53 |
| Top-10k cutoff | ~65 |
| World #1 | ~75 |
| **This agent (3-season avg)** | **65.3** |

**Deployment posture: human-in-the-loop (HITL).** When deployed, the agent
*recommends* weekly moves via a one-page report; the user clicks them in
the FPL UI. This avoids any ambiguity around FPL's Terms & Conditions on
automated submission while capturing ~99% of the value (the prediction +
optimisation is the hard part). See [DECISIONS.md](./DECISIONS.md) for the
full reasoning, and [NEXT_STEPS.md](./NEXT_STEPS.md) for the deployment
build plan.

## Quickstart

Requires Python 3.11 and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/jaydrive11/ai_fpl_agent.git
cd ai_fpl_agent
uv sync --extra dev

# 1. Build the historical dataset (downloads ~30 MB from Vaastav's repo)
uv run python scripts/build_history.py

# 2. Train models (mean L2 + P90 quantile + two-stage hurdle)
uv run python scripts/train_model.py

# 3. Pick a squad for the most recent gameweek in the data
uv run python scripts/pick_squad.py --show-actuals

# 4. Run a full-season backtest
uv run python scripts/backtest.py --season 2025-26

# 5. Multi-season walk-forward backtest (the headline result)
uv run python scripts/backtest_multi.py --two-stage
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                 │
│  history.py: fetches Vaastav per-GW CSVs (10 seasons)            │
│  storage.py: parquet cache paths                                 │
│  data.py: live FPL API client + Player dataclass                 │
└────────────────────┬────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FEATURE LAYER                                │
│  features.py: vectorized rolling (3/5/10 GW) + lag-1 + fixture  │
│               difficulty join (opp_attack, opp_defence, etc.)   │
│               • shift(1) ensures NO leakage from target GW       │
└────────────────────┬────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL LAYER (LightGBM)                      │
│  Two-stage hurdle decomposition:                                 │
│    E[points]   = P(plays ≥ 60min) × E[pts | plays ≥ 60min]      │
│    P90[points] = P(plays ≥ 60min) × P90[pts | plays ≥ 60min]    │
│                                                                  │
│  Stage 1: binary classifier (all rows, target = minutes ≥ 60)   │
│  Stage 2: L2 + quantile α=0.9 regressors (only rows w/ ≥60 min) │
└────────────────────┬────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OPTIMIZER (PuLP / MILP)                       │
│  Decision vars (per player i):                                   │
│    x[i] ∈ {0,1}  – in 15-man squad                              │
│    s[i] ∈ {0,1}  – in starting XI                               │
│    c[i] ∈ {0,1}  – is captain                                   │
│    hits ≥ 0     – transfers beyond free, costs -4 each          │
│                                                                  │
│  Objective:                                                      │
│    Σ s[i]·mean_xpts                  (XI scores this GW)        │
│  + Σ c[i]·p90_xpts                   (captain uses ceiling)      │
│  + α · Σ x[i]·horizon_xpts           (multi-GW ownership bonus)  │
│  - hit_cost · hits                                              │
│                                                                  │
│  Subject to: budget, position counts, max-3-per-club,           │
│  formation bounds (1-3-5 / 1-5-5-3 etc.), captain∈XI.           │
└────────────────────┬────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                BACKTEST + CHIP STRATEGY                          │
│  backtest.py: GW-by-GW season simulation                         │
│    – 1 free transfer per GW, banked up to 2                      │
│    – chips (WC/TC/BB) fire on heuristic thresholds              │
│    – 3 strategies compared: agent / static squad / oracle        │
└─────────────────────────────────────────────────────────────────┘
```

## Repository layout

```
src/fpl_agent/
  __init__.py
  rules.py          # FPL constants (squad size, budget, positions, etc.)
  data.py           # Player dataclass + live FPL API client
  storage.py        # cache paths under data/
  history.py        # Vaastav dataset fetcher + per-season loader
  features.py       # vectorized feature engineering (rolling, lag, FDR join)
  model.py          # LightGBM training, two-stage hurdle, predictor wrapper
  optimizer.py      # MILP for squad/XI/captain with transfer-cost awareness
  backtest.py       # season simulator with chip strategy

scripts/
  build_history.py        # one-shot data pipeline (parquet cache)
  train_model.py          # train single-stage + two-stage models, save artifacts
  pick_squad.py           # demo: pick squad for a target GW (live or retro)
  backtest.py             # single-season backtest
  backtest_multi.py       # walk-forward across multiple seasons (headline)
  diagnose.py             # 6-section diagnostic (distribution, captain, init squad, etc.)
  captain_alpha_sweep.py  # quick eval: best quantile α for captain

tests/
  test_optimizer.py   # FPL rules, transfer accounting
  test_features.py    # no-leakage rolling, season boundaries
  test_history.py     # schema-drift handling
  test_pipeline.py    # model → optimizer integration
  test_backtest.py    # transfer-aware optimizer correctness
  test_chips.py       # TC/BB/WC scoring + decision logic
```

## Data sources

- **Historical**: [Vaastav's `Fantasy-Premier-League` repo](https://github.com/vaastav/Fantasy-Premier-League) — 10 seasons (2016-17 to 2025-26) of per-GW player stats, fixtures, team strength ratings.
- **Live state** (for production use, not yet integrated): [official FPL API](https://fantasy.premierleague.com/api/) — bootstrap-static + fixtures endpoints.
- **Coverage gaps**:
  - `expected_goals` / `expected_assists` / `expected_goal_involvements` / `expected_goals_conceded` are only present from **2022-23 onward**.
  - `teams.csv` is missing for **2016-17 through 2018-19** (fixture-difficulty features are NaN there; LightGBM handles).
  - `position` is missing from `merged_gw.csv` for **2016-17 through 2019-20** but is backfilled from `players_raw.csv` via the `element` id.

## Documentation

- **[DECISIONS.md](./DECISIONS.md)** — engineering journey, every milestone with rationale, what worked, what didn't, and why.
- **[NEXT_STEPS.md](./NEXT_STEPS.md)** — ranked recommended future work.

## Testing

```bash
uv run pytest        # 23 tests, ~12s
```

## License

MIT (or your preference — TODO: pick one).
