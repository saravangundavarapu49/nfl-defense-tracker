"""Tab definitions: which charts, tables and KPI cards each tab shows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import analytics as an
from . import charts as ch
from .data import DataCapabilities


@dataclass
class ViewResult:
    chart: ch.ChartResult
    table: pd.DataFrame


@dataclass(frozen=True)
class Kpi:
    label: str
    value: str
    detail: str = ""


ViewBuilder = Callable[[an.Scope, DataCapabilities, ch.Theme], ViewResult]
KpiBuilder = Callable[[an.Scope, DataCapabilities], list[Kpi]]


@dataclass(frozen=True)
class TabSpec:
    title: str
    views: dict[str, ViewBuilder]
    kpis: KpiBuilder


def format_cell(column: str, value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        if "%" in column:
            return ch.pct(v)
        if "EPA" in column:
            return ch.dec(v)
        if column.startswith("Yds"):
            return ch.num2(v)
        if column in ("Season", "Week", "Box"):
            return str(int(v))
        return ch.num(v) if v.is_integer() else f"{v:,.1f}"
    return str(value)


def _who(scope: an.Scope) -> str:
    return scope.label


def _subtitle(scope: an.Scope, plays: float, unit: str = "plays") -> str:
    if scope.has_player:
        return f"{scope.player_name} on the field - {ch.num(plays)} {unit}"
    return f"{_who(scope)} defense - {ch.num(plays)} {unit}"


def _need(flag: bool, what: str, theme: ch.Theme) -> ViewResult | None:
    if flag:
        return None
    return ViewResult(ch.message(what, theme), pd.DataFrame())


NEED_COVERAGE = ("No coverage data found.\nUpload the matching nflverse "
                 "pbp_participation_YYYY file\nalongside play-by-play.")
NEED_BLITZ = ("No pass-rush data found.\nUpload nflverse ftn_charting_YYYY "
              "and/or pbp_participation_YYYY files.")
NEED_PLAYERS = ("No defensive player data found.\nUse a full nflverse play_by_play_YYYY "
                "file (tackle / sack / pass-defense player columns).")
NO_PLAYS = "No plays match the current filters."
TOP_PLAYERS = 15


def _empty(table: pd.DataFrame, theme: ch.Theme) -> ViewResult | None:
    if table.empty:
        return ViewResult(ch.message(NO_PLAYS, theme), table)
    return None


# --- Coverages & zones -----------------------------------------------------------

def coverage_usage_view(scope, caps, theme) -> ViewResult:
    if missing := _need(caps.coverage, NEED_COVERAGE, theme):
        return missing
    table = an.coverage_usage(scope)
    if empty := _empty(table, theme):
        return empty
    series = {c: table[c] for c in table.columns if c.endswith("%")}
    plays = len(an.coverage_plays(scope.team))
    chart = ch.grouped_bar(table["Coverage"].tolist(), series, title="Coverage Usage",
                           ylabel="Share of dropbacks", fmt=ch.pct, theme=theme,
                           subtitle=_subtitle(scope, plays, "charted dropbacks"))
    return ViewResult(chart, table)


def coverage_epa_view(scope, caps, theme) -> ViewResult:
    if missing := _need(caps.coverage, NEED_COVERAGE, theme):
        return missing
    table = an.coverage_performance(scope.team)
    if empty := _empty(table, theme):
        return empty
    chart = ch.grouped_bar(table["Coverage"].tolist(), {"EPA/Play": table["EPA/Play"]},
                           title="EPA Allowed by Coverage", ylabel="EPA per dropback",
                           fmt=ch.dec, theme=theme, counts=table["Plays"].tolist(),
                           signed_colors=True,
                           subtitle=_subtitle(scope, table["Plays"].sum(), "dropbacks")
                           + "  (green = below 0 EPA)")
    return ViewResult(chart, table)


def man_zone_view(scope, caps, theme) -> ViewResult:
    if missing := _need(caps.man_zone, NEED_COVERAGE, theme):
        return missing
    table = an.man_zone_performance(scope.team)
    if empty := _empty(table, theme):
        return empty
    metrics = ["Success %", "Comp %", "Sack %", "Pressure %", "1st Down %"]
    metrics = [m for m in metrics if table[m].notna().any()]
    series = {row["Scheme"]: [row[m] for m in metrics] for _, row in table.iterrows()}
    chart = ch.grouped_bar(metrics, series, title="Man vs Zone", ylabel="Rate",
                           fmt=ch.pct, theme=theme,
                           subtitle=_subtitle(scope, table["Plays"].sum(), "dropbacks")
                           + "  |  EPA/play: " + ", ".join(
                               f"{r['Scheme']} {ch.dec(r['EPA/Play'])}"
                               for _, r in table.iterrows()))
    return ViewResult(chart, table)


def zone_rate_scatter_view(scope, caps, theme) -> ViewResult:
    if missing := _need(caps.man_zone, NEED_COVERAGE, theme):
        return missing
    table = an.team_coverage_profile(scope.league)
    if empty := _empty(table, theme):
        return empty
    chart = ch.team_scatter(table["Team"].tolist(), table["Zone %"], table["EPA/Dropback"],
                            title="Zone Rate vs EPA Allowed (League)", xlabel="Zone %",
                            ylabel="EPA/Dropback", xfmt=ch.pct, yfmt=ch.dec, theme=theme,
                            highlight=scope.team_name,
                            subtitle="Higher on chart = better pass defense")
    return ViewResult(chart, table)


def coverage_kpis(scope, caps) -> list[Kpi]:
    if not caps.coverage:
        return [Kpi("Coverage data", "Missing", "add pbp_participation file")]
    plays = an.coverage_plays(scope.team)
    perf = an.coverage_performance(scope.team)
    mz = an.man_zone_performance(scope.team).set_index("Scheme")
    top = perf.sort_values("Plays", ascending=False).iloc[0] if not perf.empty else None
    best = perf[perf["Plays"] >= 20].sort_values("EPA/Play").head(1)
    return [
        Kpi("Charted dropbacks", ch.num(len(plays))),
        Kpi("Most-used coverage", top["Coverage"] if top is not None else "-",
            ch.pct(top["Share %"]) if top is not None else ""),
        Kpi("Zone rate", ch.pct(mz["Share %"].get("Zone", np.nan))),
        Kpi("EPA/db in Man", ch.dec(mz["EPA/Play"].get("Man", np.nan))),
        Kpi("EPA/db in Zone", ch.dec(mz["EPA/Play"].get("Zone", np.nan))),
        Kpi("Best coverage (20+)", best["Coverage"].iloc[0] if not best.empty else "-",
            ch.dec(best["EPA/Play"].iloc[0]) if not best.empty else ""),
        _top_player(scope, an.player_coverage(scope.team), "Plays on Ball",
                    "Top playmaker"),
    ]


# --- 3rd down --------------------------------------------------------------------

def _distance_series(scope) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    league = an.third_down_by_distance(scope.league).set_index("Distance")
    team = an.third_down_by_distance(scope.team).set_index("Distance")
    cats = [d for d in an.DISTANCE_BUCKETS if d in league.index]
    series = {}
    if scope.has_team:
        series[scope.team_name] = [team["Conv %"].get(d, np.nan) for d in cats]
    series["League"] = [league["Conv %"].get(d, np.nan) for d in cats]
    return team.reset_index(), series


def third_distance_view(scope, caps, theme) -> ViewResult:
    table, series = _distance_series(scope)
    if empty := _empty(table, theme):
        return empty
    chart = ch.grouped_bar(list(table["Distance"]), series,
                           title="3rd Down Conversion Rate Allowed by Distance",
                           ylabel="Conversion rate", fmt=ch.pct, theme=theme,
                           subtitle=_subtitle(scope, table["Attempts"].sum(), "3rd downs"))
    return ViewResult(chart, table)


def third_team_view(scope, caps, theme) -> ViewResult:
    table = an.third_down_by_team(scope.league)
    if empty := _empty(table, theme):
        return empty
    chart = ch.ranked_bar(table["Team"].tolist(), table["Conv %"].tolist(),
                          title="3rd Down Conversion Rate Allowed - League Ranking",
                          ylabel="Conversion rate", fmt=ch.pct, theme=theme,
                          highlight=scope.team_name,
                          extra=[f"Attempts: {ch.num(a)}" for a in table["Attempts"]],
                          subtitle="Lower is better")
    return ViewResult(chart, table)


def third_coverage_view(scope, caps, theme) -> ViewResult:
    if missing := _need(caps.coverage, NEED_COVERAGE, theme):
        return missing
    table = an.third_down_by_coverage(scope.team)
    if empty := _empty(table, theme):
        return empty
    chart = ch.grouped_bar(table["Coverage"].tolist(), {"Conv %": table["Conv %"]},
                           title="3rd Down Conversion Rate Allowed by Coverage",
                           ylabel="Conversion rate", fmt=ch.pct, theme=theme,
                           counts=table["Attempts"].tolist(),
                           subtitle=_subtitle(scope, table["Attempts"].sum(),
                                              "3rd-down dropbacks"))
    return ViewResult(chart, table)


def third_call_view(scope, caps, theme) -> ViewResult:
    play_type = an.third_down_by_play_type(scope.team)
    blitz = an.third_down_by_blitz(scope.team) if caps.blitz else pd.DataFrame()
    if empty := _empty(play_type, theme):
        return empty
    cats = play_type["Play Type"].tolist()
    vals = play_type["Conv %"].tolist()
    counts = play_type["Attempts"].tolist()
    if not blitz.empty:
        cats += [f"Pass vs {c}" for c in blitz["Pressure Call"]]
        vals += blitz["Conv %"].tolist()
        counts += blitz["Attempts"].tolist()
    table = pd.concat([
        play_type.rename(columns={"Play Type": "Split"}),
        blitz.rename(columns={"Pressure Call": "Split"}),
    ], ignore_index=True)
    chart = ch.grouped_bar(cats, {"Conv %": vals}, title="3rd Down: Run/Pass & Blitz Splits",
                           ylabel="Conversion rate", fmt=ch.pct, theme=theme, counts=counts,
                           subtitle=_subtitle(scope, play_type["Attempts"].sum(),
                                              "3rd downs"))
    return ViewResult(chart, table)


def third_kpis(scope, caps) -> list[Kpi]:
    team = an.third_down_plays(scope.team)
    league = an.third_down_plays(scope.league)
    by_team = an.third_down_by_team(scope.league).set_index("Team")["Conv %"]
    kpis = [
        Kpi("3rd downs faced", ch.num(len(team))),
        Kpi("Conv % allowed", ch.pct(team["converted"].mean() if len(team) else np.nan)),
        Kpi("League conv %", ch.pct(league["converted"].mean() if len(league) else np.nan)),
        Kpi("EPA/play", ch.dec(team["epa"].mean() if len(team) else np.nan)),
    ]
    if scope.has_team:
        kpis.append(Kpi("League rank", an.team_rank(by_team, scope.team_name), "1 = best"))
    long = team[team["ydstogo"] >= 7]
    kpis.append(Kpi("Conv % on 3rd & 7+",
                    ch.pct(long["converted"].mean() if len(long) else np.nan)))
    kpis.append(_top_player(scope, an.player_third_down(scope.team), "Stops",
                            "Top 3rd-down stopper"))
    return kpis


# --- 4th & 1 ---------------------------------------------------------------------

def fourth_team_view(scope, caps, theme) -> ViewResult:
    table = an.fourth_and_one_by_team(scope.league)
    if empty := _empty(table, theme):
        return empty
    chart = ch.ranked_bar(table["Team"].tolist(), table["Stop %"].tolist(),
                          title="4th & 1 Stop Rate - League Ranking", ylabel="Stop rate",
                          fmt=ch.pct, theme=theme, highlight=scope.team_name,
                          extra=[f"Stops: {ch.num(s)} of {ch.num(f)}"
                                 for s, f in zip(table["Stops"], table["Faced"], strict=False)],
                          subtitle="Go-for-it plays only (runs & passes)")
    return ViewResult(chart, table)


def fourth_play_type_view(scope, caps, theme) -> ViewResult:
    table = an.fourth_and_one_breakdown(scope.team, "Play Type", "Play Type")
    if empty := _empty(table, theme):
        return empty
    league = an.fourth_and_one_breakdown(scope.league, "Play Type", "Play Type")
    league = league.set_index("Play Type")["Stop %"]
    cats = table["Play Type"].tolist()
    series = {_who(scope): table["Stop %"].tolist()}
    if scope.has_team:
        series["League"] = [league.get(c, np.nan) for c in cats]
    chart = ch.grouped_bar(cats, series, title="4th & 1 Stop Rate: Run vs Pass",
                           ylabel="Stop rate", fmt=ch.pct, theme=theme,
                           counts=table["Faced"].tolist(),
                           subtitle=_subtitle(scope, table["Faced"].sum(), "4th & 1 plays"))
    return ViewResult(chart, table)


def fourth_box_view(scope, caps, theme) -> ViewResult:
    if missing := _need(caps.box, "No box-count data found.\nUpload pbp_participation "
                        "or ftn_charting files.", theme):
        return missing
    table = an.fourth_and_one_breakdown(scope.team, "box_count", "Box")
    if empty := _empty(table, theme):
        return empty
    chart = ch.grouped_bar([f"{b} in box" for b in table["Box"]], {"Stop %": table["Stop %"]},
                           title="4th & 1 Stop Rate by Defenders in Box",
                           ylabel="Stop rate", fmt=ch.pct, theme=theme,
                           counts=table["Faced"].tolist(),
                           subtitle=_subtitle(scope, table["Faced"].sum(), "4th & 1 plays"))
    return ViewResult(chart, table)


def fourth_log_view(scope, caps, theme) -> ViewResult:
    table = an.fourth_and_one_log(scope.team)
    if empty := _empty(table, theme):
        return empty
    by_week = table.groupby("Result").size()
    chart = ch.grouped_bar(by_week.index.tolist(), {"Plays": by_week.values.astype(float)},
                           title="4th & 1 Outcomes", ylabel="Plays", fmt=ch.num,
                           theme=theme, subtitle=_subtitle(scope, len(table), "4th & 1 plays")
                           + "  - full play log below")
    return ViewResult(chart, table)


def fourth_kpis(scope, caps) -> list[Kpi]:
    team = an.fourth_and_one_plays(scope.team)
    league = an.fourth_and_one_plays(scope.league)
    by_team = an.fourth_and_one_by_team(scope.league).set_index("Team")["Stop %"]
    kpis = [
        Kpi("4th & 1 faced", ch.num(len(team))),
        Kpi("Stops", ch.num(team["stopped"].sum())),
        Kpi("Stop rate", ch.pct(team["stopped"].mean() if len(team) else np.nan)),
        Kpi("League stop rate", ch.pct(league["stopped"].mean() if len(league) else np.nan)),
        Kpi("EPA/play", ch.dec(team["epa"].mean() if len(team) else np.nan)),
    ]
    if scope.has_team:
        kpis.append(Kpi("League rank", an.team_rank(by_team, scope.team_name, False),
                        "1 = best"))
    kpis.append(_top_player(scope, an.player_fourth_and_one(scope.team), "Stops",
                            "Top 4th & 1 stopper"))
    return kpis


# --- Blitz -----------------------------------------------------------------------

def blitz_compare_view(scope, caps, theme) -> ViewResult:
    if missing := _need(caps.blitz, NEED_BLITZ, theme):
        return missing
    table = an.blitz_vs_base(scope.team)
    if empty := _empty(table, theme):
        return empty
    metrics = [m for m in ["Success %", "Comp %", "Sack %", "INT %", "Pressure %",
                           "1st Down %"] if table[m].notna().any()]
    series = {row["Pressure Call"]: [row[m] for m in metrics] for _, row in table.iterrows()}
    chart = ch.grouped_bar(metrics, series, title="Blitz vs No Blitz", ylabel="Rate",
                           fmt=ch.pct, theme=theme,
                           subtitle=_subtitle(scope, table["Plays"].sum(), "dropbacks")
                           + f"  |  blitz = {caps.blitz_source}  |  EPA: " + ", ".join(
                               f"{r['Pressure Call']} {ch.dec(r['EPA/Play'])}"
                               for _, r in table.iterrows()))
    return ViewResult(chart, table)


def blitz_rushers_view(scope, caps, theme) -> ViewResult:
    if missing := _need(caps.blitz, NEED_BLITZ, theme):
        return missing
    table = an.blitz_by_rushers(scope.team)
    if empty := _empty(table, theme):
        return empty
    chart = ch.grouped_bar(table["Rushers"].tolist(), {"EPA/Play": table["EPA/Play"]},
                           title="EPA Allowed by Number of Pass Rushers",
                           ylabel="EPA per dropback", fmt=ch.dec, theme=theme,
                           counts=table["Plays"].tolist(), signed_colors=True,
                           subtitle=_subtitle(scope, table["Plays"].sum(), "dropbacks"))
    return ViewResult(chart, table)


def blitz_down_view(scope, caps, theme) -> ViewResult:
    if missing := _need(caps.blitz, NEED_BLITZ, theme):
        return missing
    table = an.blitz_rate_by_down(scope)
    if empty := _empty(table, theme):
        return empty
    series = {c: table[c] for c in table.columns if c.endswith("%")}
    chart = ch.grouped_bar(table["Down"].tolist(), series, title="Blitz Rate by Down",
                           ylabel="Blitz rate", fmt=ch.pct, theme=theme,
                           subtitle=f"{_who(scope)} defense  |  blitz = {caps.blitz_source}")
    return ViewResult(chart, table)


def blitz_scatter_view(scope, caps, theme) -> ViewResult:
    if missing := _need(caps.blitz, NEED_BLITZ, theme):
        return missing
    table = an.team_blitz_profile(scope.league)
    if empty := _empty(table, theme):
        return empty
    chart = ch.team_scatter(table["Team"].tolist(), table["Blitz %"], table["EPA/Dropback"],
                            title="Blitz Rate vs EPA Allowed (League)", xlabel="Blitz %",
                            ylabel="EPA/Dropback", xfmt=ch.pct, yfmt=ch.dec, theme=theme,
                            highlight=scope.team_name,
                            subtitle="Higher on chart = better pass defense")
    return ViewResult(chart, table)


def blitz_kpis(scope, caps) -> list[Kpi]:
    if not caps.blitz:
        return [Kpi("Blitz data", "Missing", "add ftn_charting / participation")]
    table = an.blitz_vs_base(scope.team).set_index("Pressure Call")
    league = an.blitz_plays(scope.league)
    by_team = an.team_blitz_profile(scope.league).set_index("Team")["Blitz %"]
    kpis = [
        Kpi("Blitz rate", ch.pct(table["Share %"].get("Blitz", np.nan)),
            f"league {ch.pct(league['blitz'].mean() if len(league) else np.nan)}"),
        Kpi("EPA when blitzing", ch.dec(table["EPA/Play"].get("Blitz", np.nan))),
        Kpi("EPA no blitz", ch.dec(table["EPA/Play"].get("No Blitz", np.nan))),
        Kpi("Sack % blitzing", ch.pct(table["Sack %"].get("Blitz", np.nan))),
    ]
    if caps.pressure:
        kpis.append(Kpi("Pressure % blitzing", ch.pct(table["Pressure %"].get("Blitz",
                                                                                np.nan))))
    if scope.has_team:
        kpis.append(Kpi("Blitz-rate rank", an.team_rank(by_team, scope.team_name, False),
                        "1 = most aggressive"))
    kpis.append(_top_player(scope, an.player_pass_rush(scope.team), "Sacks", "Sack leader"))
    return kpis


# --- Players ---------------------------------------------------------------------

def _player_label(row: pd.Series) -> str:
    pos = f"{row['Pos']}, " if isinstance(row["Pos"], str) and row["Pos"] else ""
    return f"{row['Player']} ({pos}{row['Team']})"


def _player_view(scope: an.Scope, caps: DataCapabilities, theme: ch.Theme,
                 table: pd.DataFrame, *, stack: list[str], title: str, xlabel: str,
                 detail_cols: list[str], unit: str) -> ViewResult:
    """Top-N leaderboard over the selected defense; the filtered player is always shown."""
    if missing := _need(caps.players or caps.on_field, NEED_PLAYERS, theme):
        return missing
    table = table[table[stack].sum(axis=1) > 0] if len(table) else table
    if empty := _empty(table, theme):
        return empty
    top = table.head(TOP_PLAYERS)
    if scope.has_player and scope.player_id not in set(top["player_id"]):
        top = pd.concat([top, table[table["player_id"] == scope.player_id]])
    highlight = None
    if scope.has_player:
        hits = [i for i, pid in enumerate(top["player_id"]) if pid == scope.player_id]
        highlight = hits[0] if hits else None
    details = [
        "\n".join(f"{c}: {format_cell(c, row[c])}" for c in detail_cols if c in row.index)
        for _, row in top.iterrows()
    ]
    pool = "League" if scope.team_name == an.ALL else scope.team_name
    chart = ch.player_bar([_player_label(r) for _, r in top.iterrows()],
                          {c: top[c].tolist() for c in stack}, title=title, xlabel=xlabel,
                          fmt=lambda v: format_cell("n", v), theme=theme, details=details,
                          highlight=highlight,
                          subtitle=f"{pool} defenders - top {min(len(table), TOP_PLAYERS)} of "
                          f"{ch.num(len(table))} with {unit}  |  full list below")
    return ViewResult(chart, table.drop(columns="player_id").reset_index(drop=True))


def _top_player(scope: an.Scope, table: pd.DataFrame, col: str, label: str) -> Kpi:
    """Leader for `col`, or the selected player's own total when a player is filtered."""
    if scope.has_player:
        mine = table[table["player_id"] == scope.player_id]
        value = mine[col].iloc[0] if len(mine) else 0.0
        return Kpi(f"{scope.player_name}: {col}", format_cell(col, value),
                   f"{format_cell('Snaps', mine['Snaps'].iloc[0]) if len(mine) else 0} snaps")
    table = table[table[col] > 0] if len(table) else table
    if table.empty:
        return Kpi(label, "-")
    row = table.iloc[0]
    return Kpi(label, str(row["Player"]), f"{format_cell(col, row[col])} {col.lower()}")


def coverage_players_view(scope, caps, theme) -> ViewResult:
    return _player_view(scope, caps, theme, an.player_coverage(scope.pool),
                        stack=["PD", "INT"], title="Plays on the Ball (Pass Defense)",
                        xlabel="Passes defensed + interceptions",
                        detail_cols=["Snaps", "Tackles", "EPA/Play", "Man EPA", "Zone EPA"],
                        unit="a PD or INT")


def third_players_view(scope, caps, theme) -> ViewResult:
    return _player_view(scope, caps, theme, an.player_third_down(scope.pool),
                        stack=["Stops"], title="3rd Down Stops by Player",
                        xlabel="Tackles / sacks on failed 3rd downs",
                        detail_cols=["Snaps", "Tackles", "Sacks", "PD", "Stop %", "EPA/Play"],
                        unit="a 3rd-down stop")


def fourth_players_view(scope, caps, theme) -> ViewResult:
    return _player_view(scope, caps, theme, an.player_fourth_and_one(scope.pool),
                        stack=["Stops"], title="4th & 1 Stops by Player",
                        xlabel="Tackles / sacks on failed 4th & 1 attempts",
                        detail_cols=["Snaps", "Tackles", "TFL", "Stop %", "EPA/Play"],
                        unit="a 4th & 1 stop")


def blitz_players_view(scope, caps, theme) -> ViewResult:
    return _player_view(scope, caps, theme, an.player_pass_rush(scope.pool),
                        stack=["Sacks", "Non-Sack Hits"], title="Pass Rush Leaders",
                        xlabel="Sacks + QB hits without a sack",
                        detail_cols=["Snaps", "QB Hits", "Blitz Sacks", "Blitz QB Hits",
                                     "EPA/Play"],
                        unit="a sack or QB hit")


TABS: list[TabSpec] = [
    TabSpec("Coverages & Zones", {
        "Coverage Usage": coverage_usage_view,
        "EPA by Coverage": coverage_epa_view,
        "Man vs Zone": man_zone_view,
        "Zone Rate vs EPA": zone_rate_scatter_view,
        "Players": coverage_players_view,
    }, coverage_kpis),
    TabSpec("3rd Down Efficiency", {
        "By Distance": third_distance_view,
        "League Ranking": third_team_view,
        "By Coverage": third_coverage_view,
        "Run/Pass & Blitz": third_call_view,
        "Players": third_players_view,
    }, third_kpis),
    TabSpec("4th & 1", {
        "League Ranking": fourth_team_view,
        "Run vs Pass": fourth_play_type_view,
        "Defenders in Box": fourth_box_view,
        "Play Log": fourth_log_view,
        "Players": fourth_players_view,
    }, fourth_kpis),
    TabSpec("Blitz Defense", {
        "Blitz vs No Blitz": blitz_compare_view,
        "By Pass Rushers": blitz_rushers_view,
        "Blitz Rate by Down": blitz_down_view,
        "Blitz Rate vs EPA": blitz_scatter_view,
        "Players": blitz_players_view,
    }, blitz_kpis),
]
