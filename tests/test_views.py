from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from nfl_defense_tracker import analytics as an  # noqa: E402
from nfl_defense_tracker import charts as ch  # noqa: E402
from nfl_defense_tracker.data import DataStore, capabilities  # noqa: E402
from nfl_defense_tracker.views import TABS, format_cell  # noqa: E402


def _plays(*frames):
    store = DataStore()
    for i, frame in enumerate(frames):
        store.add_frame(frame, Path(str(i)))
    return store.build()


@pytest.mark.parametrize("team", ["All", "BAL"])
def test_every_view_renders(pbp, participation, ftn, team):
    plays = _plays(pbp, participation, ftn)
    scope = an.apply_filters(plays, an.Filters(team=team))
    caps = capabilities(plays)
    assert [t.title for t in TABS] == [
        "Coverages & Zones", "3rd Down Efficiency", "4th & 1", "Blitz Defense"]
    for spec in TABS:
        assert spec.kpis(scope, caps)
        for name, build in spec.views.items():
            result = build(scope, caps, ch.DARK)
            assert result.chart.figure.axes, name
            assert not result.table.empty, name
            assert result.chart.hovers, name


def test_player_filter_highlights_player(pbp, participation, ftn):
    plays = _plays(pbp, participation, ftn)
    scope = an.apply_filters(plays, an.Filters(team="KC", player_id="KC-DE",
                                               player_name="KC Edge"))
    caps = capabilities(plays)
    result = TABS[3].views["Players"](scope, caps, ch.DARK)
    assert "KC Edge" in result.table["Player"].tolist()
    assert any("KC Edge" in t.labels[0] for t in result.chart.hovers)
    kpis = TABS[3].kpis(scope, caps)
    assert kpis[-1].label == "KC Edge: Sacks"
    texts = TABS[1].views["By Distance"](scope, caps, ch.DARK).chart.figure.axes[0].texts
    assert any("KC Edge on the field" in t.get_text() for t in texts)


def test_views_explain_missing_data(pbp):
    plays = _plays(pbp)
    scope = an.apply_filters(plays, an.Filters())
    caps = capabilities(plays)
    coverage = TABS[0].views["Coverage Usage"](scope, caps, ch.LIGHT)
    assert coverage.table.empty
    text = coverage.chart.figure.axes[0].texts[0].get_text()
    assert "pbp_participation" in text
    blitz = TABS[3].views["Blitz vs No Blitz"](scope, caps, ch.LIGHT)
    assert "ftn_charting" in blitz.chart.figure.axes[0].texts[0].get_text()
    # Tabs that only need play-by-play still work.
    assert not TABS[1].views["By Distance"](scope, caps, ch.LIGHT).table.empty


def test_empty_filter_shows_message(pbp):
    plays = _plays(pbp)
    scope = an.apply_filters(plays, an.Filters(seasons=(1999,)))
    result = TABS[1].views["By Distance"](scope, capabilities(plays), ch.DARK)
    assert result.table.empty


def test_format_cell():
    assert format_cell("Conv %", 0.4567) == "45.7%"
    assert format_cell("EPA/Play", -0.12345) == "-0.123"
    assert format_cell("Plays", 1234.0) == "1,234"
    assert format_cell("Week", 7.0) == "7"
    assert format_cell("Team", "BAL") == "BAL"
    assert format_cell("EPA", float("nan")) == "-"
    assert format_cell("Yds/Play", 4.5) == "4.50"
    assert format_cell("Sacks", 4.5) == "4.5"
