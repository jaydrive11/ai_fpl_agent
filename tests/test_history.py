import pandas as pd
import pytest

from fpl_agent import history, storage


def test_season_list_is_chronological_and_well_formed():
    years = [int(s.split("-")[0]) for s in history.SEASONS]
    assert years == sorted(years)
    for s in history.SEASONS:
        a, b = s.split("-")
        assert int(b) == (int(a) + 1) % 100


def test_load_all_gw_concats_with_schema_drift(tmp_path, monkeypatch):
    """Old seasons missing newer columns (e.g. expected_goals) must concat cleanly,
    with NaN backfill — that's the whole reason for sort=False, outer concat."""
    monkeypatch.setattr(storage, "DATA_ROOT", tmp_path)

    def write_csv(season: str, df: pd.DataFrame) -> None:
        p = tmp_path / "raw" / season / "gws" / "merged_gw.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)

    write_csv("2018-19", pd.DataFrame({
        "name": ["A"], "total_points": [5], "GW": [1], "minutes": [90],
    }))
    write_csv("2024-25", pd.DataFrame({
        "name": ["B"], "total_points": [7], "GW": [1], "minutes": [90],
        "expected_goals": [0.4], "expected_assists": [0.2],
    }))

    df = history.load_all_gw(seasons=("2018-19", "2024-25"))

    assert set(df["season"]) == {"2018-19", "2024-25"}
    assert "expected_goals" in df.columns
    assert df.loc[df.season == "2018-19", "expected_goals"].isna().all()
    assert df.loc[df.season == "2024-25", "expected_goals"].iloc[0] == 0.4


def test_load_all_gw_errors_when_no_data(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        history.load_all_gw(seasons=("2018-19",))
