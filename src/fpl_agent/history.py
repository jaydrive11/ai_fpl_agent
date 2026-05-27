"""Fetch and load historical FPL data from the open-source Vaastav dataset.

Source: https://github.com/vaastav/Fantasy-Premier-League
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from . import storage

VAASTAV_BASE = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
)

# Ordered oldest -> newest. Update when a new season is added upstream.
SEASONS: tuple[str, ...] = (
    "2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
)

# Files we pull per season (paths relative to data/<season>/).
SEASON_FILES: tuple[str, ...] = (
    "gws/merged_gw.csv",
    "fixtures.csv",
    "teams.csv",
    "players_raw.csv",
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)


def fetch_season(season: str, *, force: bool = False) -> dict[str, str]:
    """Download raw CSVs for one season. Per-file tolerant: a missing file
    (e.g. older seasons lacking fixtures.csv) doesn't abort the rest.
    Returns a {relative_path: status} map."""
    result: dict[str, str] = {}
    for rel in SEASON_FILES:
        url = f"{VAASTAV_BASE}/{season}/{rel}"
        dest = storage.raw_dir() / season / rel
        if dest.exists() and not force:
            result[rel] = "cached"
            continue
        try:
            _download(url, dest)
            result[rel] = "ok"
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            result[rel] = f"HTTP {code}"
        except requests.RequestException as e:
            result[rel] = f"error: {type(e).__name__}"
    return result


def fetch_all(
    seasons: tuple[str, ...] = SEASONS, *, force: bool = False
) -> dict[str, dict[str, str]]:
    """Fetch every season; return per-season per-file status map."""
    return {s: fetch_season(s, force=force) for s in seasons}


_POSITION_FROM_ELEMENT_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def _enrich_position(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Older seasons (2016-17..2019-20) lack `position` in merged_gw.csv.
    Backfill from players_raw.csv via the `element` (player) id."""
    if "position" in df.columns and df["position"].notna().any():
        return df

    pr_path = storage.raw_dir() / season / "players_raw.csv"
    if not pr_path.exists():
        return df

    pr = pd.read_csv(pr_path, encoding="utf-8", encoding_errors="replace")
    if "element_type" not in pr.columns or "id" not in pr.columns:
        return df

    lookup = pd.DataFrame({
        "element": pr["id"],
        "position": pr["element_type"].map(_POSITION_FROM_ELEMENT_TYPE),
    })
    df = df.drop(columns=["position"], errors="ignore")
    return df.merge(lookup, on="element", how="left")


def load_team_strengths(season: str) -> pd.DataFrame | None:
    """Load FPL team-strength ratings from teams.csv for one season.

    Returns columns: id, name, attack_h, attack_a, defence_h, defence_a
    (renamed from strength_attack_home etc.). Returns None for older seasons
    that lack teams.csv (2016-17 through 2018-19 in Vaastav's repo).
    """
    p = storage.raw_dir() / season / "teams.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, encoding="utf-8", encoding_errors="replace")
    needed = ["id", "name",
              "strength_attack_home", "strength_attack_away",
              "strength_defence_home", "strength_defence_away"]
    if not all(c in df.columns for c in needed):
        return None
    return df[needed].rename(columns={
        "strength_attack_home": "attack_h",
        "strength_attack_away": "attack_a",
        "strength_defence_home": "defence_h",
        "strength_defence_away": "defence_a",
    })


def load_season_gw(season: str) -> pd.DataFrame:
    """Load the per-GW per-player table for one season; adds a `season` column
    and backfills `position` from players_raw.csv where merged_gw lacks it."""
    path = storage.raw_dir() / season / "gws" / "merged_gw.csv"
    # Older seasons sometimes have stray non-UTF8 bytes in player names.
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    df = _enrich_position(df, season)
    df.insert(0, "season", season)
    return df


def load_all_gw(seasons: tuple[str, ...] = SEASONS) -> pd.DataFrame:
    """Concat all seasons' merged_gw tables. Missing columns in older seasons -> NaN."""
    frames = []
    for s in seasons:
        path = storage.raw_dir() / s / "gws" / "merged_gw.csv"
        if not path.exists():
            continue
        frames.append(load_season_gw(s))
    if not frames:
        raise FileNotFoundError(
            f"No merged_gw.csv files under {storage.raw_dir()}. Run fetch_all() first."
        )
    return pd.concat(frames, ignore_index=True, sort=False)


def cache_processed_parquet() -> Path:
    """Write the unified per-GW dataset to Parquet; return the output path."""
    df = load_all_gw()
    out = storage.processed_dir() / "gw_history.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out
