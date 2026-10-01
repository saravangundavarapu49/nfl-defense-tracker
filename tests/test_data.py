from pathlib import Path

import pandas as pd
import pytest

from nfl_defense_tracker.data import (
    DataLoadError,
    DataStore,
    capabilities,
    detect_kind,
    read_table,
)


def test_detect_kind(pbp, participation, ftn):
    assert detect_kind(pbp) == "pbp"
    assert detect_kind(participation) == "participation"
    assert detect_kind(ftn) == "ftn"
    with pytest.raises(DataLoadError):
        detect_kind(pd.DataFrame({"a": [1]}))


def test_read_csv_and_parquet_roundtrip(tmp_path: Path, pbp):
    csv = tmp_path / "pbp.csv"
    parquet = tmp_path / "pbp.parquet"
    pbp.to_csv(csv, index=False)
    pbp.to_parquet(parquet)
    assert len(read_table(csv)) == len(pbp)
    assert len(read_table(parquet)) == len(pbp)
    with pytest.raises(DataLoadError):
        read_table(tmp_path / "missing.csv")
    (tmp_path / "x.txt").write_text("x")
    with pytest.raises(DataLoadError):
        read_table(tmp_path / "x.txt")


def test_build_requires_pbp(participation):
    store = DataStore()
    store.add_frame(participation, Path("part.csv"))
    with pytest.raises(DataLoadError):
        store.build()


def test_build_pbp_only_filters_scrimmage_plays(pbp):
    store = DataStore()
    store.add_frame(pbp, Path("pbp.parquet"))
    plays = store.build()
    assert set(plays["play_type"]) == {"pass", "run"}
    assert len(plays) == 60
    caps = capabilities(plays)
    assert not caps.coverage and not caps.blitz and caps.blitz_source == "unavailable"


def test_build_merges_participation(pbp, participation):
    store = DataStore()
    store.add_frame(pbp, Path("pbp.parquet"))
    store.add_frame(participation, Path("part.parquet"))
    plays = store.build()
    assert set(plays["coverage"].dropna()) == {"Cover 1", "Cover 3", "2-Man"}
    assert set(plays["man_zone"].dropna()) == {"Man", "Zone"}
    # Blitz falls back to 5+ rushers, only on dropbacks with a rusher count.
    rushers = plays.loc[plays["blitz"].notna(), "pass_rushers"]
    assert (plays.loc[rushers.index, "blitz"] == (rushers >= 5).astype(float)).all()
    assert plays.loc[~plays["is_pass_play"], "blitz"].isna().all()
    assert capabilities(plays).blitz_source == "5+ pass rushers"


def test_build_prefers_ftn_blitzers(pbp, participation, ftn):
    store = DataStore()
    for frame, name in [(pbp, "pbp"), (participation, "part"), (ftn, "ftn")]:
        store.add_frame(frame, Path(name))
    plays = store.build()
    with_ftn = plays["blitzers"].notna()
    assert (plays.loc[with_ftn, "blitz"] == (plays.loc[with_ftn, "blitzers"] > 0)).all()
    assert capabilities(plays).blitz_source == "FTN blitzer counts"
    assert plays["box_count"].notna().all()


def test_duplicate_files_are_deduplicated(pbp):
    store = DataStore()
    store.add_frame(pbp, Path("a"))
    store.add_frame(pbp, Path("b"))
    assert len(store.build()) == 60
