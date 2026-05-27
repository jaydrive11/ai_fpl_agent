"""Fetch all Vaastav historical seasons and write a unified Parquet cache."""
import pandas as pd

from fpl_agent.history import SEASONS, cache_processed_parquet, fetch_all


def main() -> None:
    print(f"Fetching {len(SEASONS)} seasons from Vaastav repo...")
    results = fetch_all()
    for season, files in results.items():
        bad = {f: st for f, st in files.items() if st not in ("ok", "cached")}
        ok_count = sum(1 for st in files.values() if st in ("ok", "cached"))
        suffix = f"  -- missing: {bad}" if bad else ""
        print(f"  {season}: {ok_count}/{len(files)} files{suffix}")

    print("\nBuilding unified Parquet cache...")
    out = cache_processed_parquet()
    df = pd.read_parquet(out)

    print(f"\nWrote {out}")
    print(f"  rows: {len(df):,}")
    print(f"  columns: {len(df.columns)}")
    print(f"  seasons: {sorted(df['season'].unique())}")
    print(f"  file size: {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
