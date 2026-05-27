"""Paths for the local data cache.

Layout under <repo>/data/:
    raw/<season>/                 -- Vaastav CSVs as downloaded
    processed/                    -- unified Parquet tables we derive from raw
"""
from pathlib import Path

# <repo>/data — sits next to src/, scripts/, tests/
DATA_ROOT: Path = Path(__file__).resolve().parents[2] / "data"


def raw_dir() -> Path:
    return DATA_ROOT / "raw"


def processed_dir() -> Path:
    return DATA_ROOT / "processed"
