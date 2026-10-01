from pathlib import Path

import pytest

from nfl_defense_tracker import analytics as an
from nfl_defense_tracker.data import DataStore


@pytest.fixture
def plays(pbp, participation, ftn):
    store = DataStore()
    for frame, name in [(pbp, "pbp"), (participation, "part"), (ftn, "ftn")]:
        store.add_frame(frame, Path(name))
    return store.build()


def test_apply_filters_team_and_weeks(plays):
    scope = an.apply_filters(plays, an.Filters(team="BAL", week_min=1, week_max=1))
    assert set(scope.team["defteam"]) == {"BAL"}
    assert set(scope.league["week"]) == {1}
    assert scope.has_team
    league = an.apply_filters(plays, an.Filters())
    assert len(league.team) == len(league.league) == len(plays)


def test_coverage_tables(plays):
    perf = an.coverage_performance(plays)
    assert list(perf["Coverage"]) == ["Cover 1", "2-Man", "Cover 3"]
    assert perf["Share %"].sum() == pytest.approx(1.0)
    usage = an.coverage_usage(an.apply_filters(plays, an.Filters(team="BAL")))
    assert {"BAL %", "League %"} <= set(usage.columns)
    assert usage["League %"].sum() == pytest.approx(1.0)
    profile = an.team_coverage_profile(plays)
    assert (profile["Zone %"] + profile["Man %"]).round(6).eq(1).all()


def test_third_down_conversion(plays):
    third = an.third_down_plays(plays)
    by_distance = an.third_down_by_distance(plays)
    assert by_distance["Attempts"].sum() == len(third)
    expected = third["third_down_converted"].mean()
    weighted = (by_distance["Conv %"] * by_distance["Attempts"]).sum() / len(third)
    assert weighted == pytest.approx(expected)
    by_team = an.third_down_by_team(plays)
    assert by_team["Conv %"].is_monotonic_increasing


def test_fourth_and_one(plays):
    f = an.fourth_and_one_plays(plays)
    assert (f["down"] == 4).all() and (f["ydstogo"] == 1).all()
    assert "punt" not in set(f["play_type"])
    by_team = an.fourth_and_one_by_team(plays)
    assert by_team["Stops"].sum() == plays["fourth_down_failed"].where(
        (plays["down"] == 4) & (plays["ydstogo"] == 1)).sum()
    log = an.fourth_and_one_log(plays)
    assert set(log["Result"]) <= {"STOP", "Converted"}
    assert len(log) == len(f)


def test_blitz_tables(plays):
    base = an.blitz_vs_base(plays)
    assert set(base["Pressure Call"]) == {"Blitz", "No Blitz"}
    rushers = an.blitz_by_rushers(plays)
    assert set(rushers["Rushers"]) <= set(an.RUSHER_BUCKETS)
    by_down = an.blitz_rate_by_down(an.apply_filters(plays, an.Filters(team="KC")))
    assert {"KC %", "League %"} <= set(by_down.columns)
    profile = an.team_blitz_profile(plays)
    assert profile["Blitz %"].between(0, 1).all()


def test_team_rank():
    import pandas as pd

    values = pd.Series({"A": 0.3, "B": 0.1, "C": 0.2})
    assert an.team_rank(values, "B") == "1 / 3"
    assert an.team_rank(values, "B", ascending=False) == "3 / 3"
    assert an.team_rank(values, "Z") == "-"
