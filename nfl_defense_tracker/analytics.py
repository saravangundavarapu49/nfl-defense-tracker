"""Pure pandas analytics for each tracker tab."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import COVERAGE_ORDER, PLAYER_ID_COLUMNS, PLAYER_STATS

ALL = "All"
DISTANCE_BUCKETS = ["Short (1-2)", "Medium (3-6)", "Long (7-10)", "Very Long (11+)"]
RUSHER_BUCKETS = ["3 or fewer", "4", "5", "6", "7+"]


@dataclass(frozen=True)
class Filters:
    seasons: tuple[int, ...] = ()
    season_type: str = ALL
    team: str = ALL
    week_min: int = 1
    week_max: int = 22
    player_id: str = ALL
    player_name: str = ""


@dataclass(frozen=True)
class Scope:
    """League-wide plays (all filters but team/player), the selected defense's plays
    (`defense`) and the analyzed subset (`team`: the defense, narrowed to snaps where the
    selected player was on the field)."""

    league: pd.DataFrame
    team: pd.DataFrame
    team_name: str
    player_id: str = ALL
    player_name: str = ""
    defense: pd.DataFrame | None = None

    @property
    def has_team(self) -> bool:
        return self.team_name != ALL or self.has_player

    @property
    def has_player(self) -> bool:
        return self.player_id != ALL

    @property
    def label(self) -> str:
        if self.has_player:
            return self.player_name
        return self.team_name if self.team_name != ALL else "League"

    @property
    def pool(self) -> pd.DataFrame:
        """Plays used for player leaderboards (ignores the player filter)."""
        return self.defense if self.defense is not None else self.team


def player_mask(df: pd.DataFrame, player_id: str) -> pd.Series:
    """Plays where the player was on the field or credited with a defensive stat."""
    mask = df["defenders"].map(lambda ids: player_id in ids).astype(bool)
    for col in PLAYER_ID_COLUMNS:
        if col in df.columns:
            mask |= (df[col] == player_id).fillna(False).astype(bool)
    return mask


def apply_filters(df: pd.DataFrame, filters: Filters) -> Scope:
    mask = pd.Series(True, index=df.index)
    if filters.seasons:
        mask &= df["season"].isin(filters.seasons)
    if filters.season_type != ALL:
        mask &= df["season_type"] == filters.season_type
    mask &= df["week"].between(filters.week_min, filters.week_max) | df["week"].isna()
    league = df[mask]
    defense = league[league["defteam"] == filters.team] if filters.team != ALL else league
    team = defense
    if filters.player_id != ALL:
        team = defense[player_mask(defense, filters.player_id)]
    return Scope(league=league, team=team, team_name=filters.team,
                 player_id=filters.player_id, player_name=filters.player_name or
                 filters.player_id, defense=defense)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else np.nan


def summarize(df: pd.DataFrame) -> dict[str, float]:
    dropbacks = df["dropback"].sum()
    attempts = df["pass_attempt"].fillna(0).sum() - df["sack"].fillna(0).sum()
    return {
        "Plays": float(len(df)),
        "EPA/Play": df["epa"].mean() if len(df) else np.nan,
        "Success %": df["success"].mean() if len(df) else np.nan,
        "Yds/Play": df["yards_gained"].mean() if len(df) else np.nan,
        "Comp %": _safe_div(df["complete_pass"].fillna(0).sum(), attempts),
        "Sack %": _safe_div(df["sack"].fillna(0).sum(), dropbacks),
        "INT %": _safe_div(df["interception"].fillna(0).sum(), dropbacks),
        "Pressure %": df["pressure"].mean() if df["pressure"].notna().any() else np.nan,
        "1st Down %": df["first_down"].mean() if len(df) else np.nan,
    }


def metric_table(
    df: pd.DataFrame, by: str, order: list[str] | None = None, label: str | None = None
) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(by, observed=True, sort=False):
        rows.append({label or by: key, **summarize(group)})
    out = pd.DataFrame(rows, columns=[label or by, *summarize(df.head(0)).keys()])
    if out.empty:
        return out
    if order:
        rank = {v: i for i, v in enumerate(order)}
        out["_o"] = out[label or by].map(lambda v: rank.get(v, len(rank)))
        out = out.sort_values(["_o", "Plays"], ascending=[True, False]).drop(columns="_o")
    else:
        out = out.sort_values("Plays", ascending=False)
    total = out["Plays"].sum()
    out.insert(2, "Share %", out["Plays"] / total if total else np.nan)
    return out.reset_index(drop=True)


def team_rank(values: pd.Series, team: str, ascending: bool = True) -> str:
    """Rank of `team` in `values` (1 = best) formatted like '5 / 32'."""
    if team not in values.index or values.dropna().empty:
        return "-"
    ranks = values.rank(ascending=ascending, method="min")
    return f"{int(ranks[team])} / {values.notna().sum()}"


# --- Coverages & zones -----------------------------------------------------------

def coverage_plays(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_pass_play"] & df["coverage"].notna()]


def coverage_performance(df: pd.DataFrame) -> pd.DataFrame:
    return metric_table(coverage_plays(df), "coverage", COVERAGE_ORDER, "Coverage")


def man_zone_performance(df: pd.DataFrame) -> pd.DataFrame:
    plays = df[df["is_pass_play"] & df["man_zone"].notna()]
    return metric_table(plays, "man_zone", ["Man", "Zone"], "Scheme")


def coverage_usage(scope: Scope) -> pd.DataFrame:
    def shares(df: pd.DataFrame) -> pd.Series:
        return coverage_plays(df)["coverage"].value_counts(normalize=True)

    league = shares(scope.league)
    out = pd.DataFrame({"League %": league})
    if scope.has_team:
        out.insert(0, f"{scope.label} %", shares(scope.team))
    out = out.fillna(0.0)
    order = [c for c in COVERAGE_ORDER if c in out.index] + [
        c for c in out.index if c not in COVERAGE_ORDER
    ]
    return out.loc[order].rename_axis("Coverage").reset_index()


def team_coverage_profile(league: pd.DataFrame) -> pd.DataFrame:
    plays = league[league["is_pass_play"] & league["man_zone"].notna()]
    if plays.empty:
        return pd.DataFrame(columns=["Team", "Zone %", "Man %", "EPA/Dropback", "Dropbacks"])
    grouped = plays.groupby("defteam")
    out = pd.DataFrame({
        "Zone %": grouped["man_zone"].apply(lambda s: (s == "Zone").mean()),
        "Man %": grouped["man_zone"].apply(lambda s: (s == "Man").mean()),
        "EPA/Dropback": grouped["epa"].mean(),
        "Dropbacks": grouped.size().astype(float),
    })
    return out.rename_axis("Team").reset_index().sort_values("EPA/Dropback")


# --- 3rd down --------------------------------------------------------------------

def distance_bucket(ydstogo: pd.Series) -> pd.Series:
    return pd.cut(ydstogo, bins=[0, 2, 6, 10, np.inf], labels=DISTANCE_BUCKETS)


def third_down_plays(df: pd.DataFrame) -> pd.DataFrame:
    plays = df[df["down"] == 3].copy()
    plays["distance"] = distance_bucket(plays["ydstogo"])
    plays["converted"] = plays["third_down_converted"].fillna(0)
    return plays


def conversion_table(plays: pd.DataFrame, by: str, label: str, order: list[str] | None = None
                     ) -> pd.DataFrame:
    if plays.empty:
        return pd.DataFrame(columns=[label, "Attempts", "Conv %", "EPA/Play", "Yds/Play"])
    grouped = plays.groupby(by, observed=True)
    out = pd.DataFrame({
        "Attempts": grouped.size().astype(float),
        "Conv %": grouped["converted"].mean(),
        "EPA/Play": grouped["epa"].mean(),
        "Yds/Play": grouped["yards_gained"].mean(),
    }).rename_axis(label).reset_index()
    if order:
        out[label] = pd.Categorical(out[label], categories=order, ordered=True)
        out = out.sort_values(label)
        out[label] = out[label].astype(str)
    return out.reset_index(drop=True)


def third_down_by_distance(df: pd.DataFrame) -> pd.DataFrame:
    return conversion_table(third_down_plays(df), "distance", "Distance", DISTANCE_BUCKETS)


def third_down_by_team(league: pd.DataFrame) -> pd.DataFrame:
    out = conversion_table(third_down_plays(league), "defteam", "Team")
    return out.sort_values("Conv %").reset_index(drop=True)


def third_down_by_coverage(df: pd.DataFrame) -> pd.DataFrame:
    plays = third_down_plays(df)
    plays = plays[plays["is_pass_play"] & plays["coverage"].notna()]
    out = conversion_table(plays, "coverage", "Coverage", None)
    if out.empty:
        return out
    order = {c: i for i, c in enumerate(COVERAGE_ORDER)}
    return out.sort_values("Coverage", key=lambda s: s.map(order)).reset_index(drop=True)


def third_down_by_play_type(df: pd.DataFrame) -> pd.DataFrame:
    plays = third_down_plays(df)
    plays["Play Type"] = np.where(plays["is_pass_play"], "Pass", "Run")
    return conversion_table(plays, "Play Type", "Play Type", ["Pass", "Run"])


def third_down_by_blitz(df: pd.DataFrame) -> pd.DataFrame:
    plays = third_down_plays(df)
    plays = plays[plays["blitz"].notna()].copy()
    plays["Pressure Call"] = np.where(plays["blitz"] == 1, "Blitz", "No Blitz")
    return conversion_table(plays, "Pressure Call", "Pressure Call", ["Blitz", "No Blitz"])


# --- 4th & 1 ---------------------------------------------------------------------

def fourth_and_one_plays(df: pd.DataFrame) -> pd.DataFrame:
    plays = df[(df["down"] == 4) & (df["ydstogo"] == 1)].copy()
    plays["converted"] = plays["fourth_down_converted"].fillna(0)
    plays["stopped"] = 1 - plays["converted"]
    plays["Play Type"] = np.where(plays["is_pass_play"], "Pass", "Run")
    return plays


def fourth_and_one_by_team(league: pd.DataFrame) -> pd.DataFrame:
    plays = fourth_and_one_plays(league)
    if plays.empty:
        return pd.DataFrame(columns=["Team", "Faced", "Stops", "Stop %", "EPA/Play"])
    grouped = plays.groupby("defteam")
    out = pd.DataFrame({
        "Faced": grouped.size().astype(float),
        "Stops": grouped["stopped"].sum(),
        "Stop %": grouped["stopped"].mean(),
        "EPA/Play": grouped["epa"].mean(),
    }).rename_axis("Team").reset_index()
    return out.sort_values(["Stop %", "Faced"], ascending=[False, False]).reset_index(drop=True)


def fourth_and_one_breakdown(df: pd.DataFrame, by: str, label: str) -> pd.DataFrame:
    plays = fourth_and_one_plays(df)
    plays = plays[plays[by].notna()]
    if plays.empty:
        return pd.DataFrame(columns=[label, "Faced", "Stop %", "EPA/Play"])
    grouped = plays.groupby(by, observed=True)
    out = pd.DataFrame({
        "Faced": grouped.size().astype(float),
        "Stop %": grouped["stopped"].mean(),
        "EPA/Play": grouped["epa"].mean(),
    }).rename_axis(label).reset_index()
    if by == "box_count":
        out[label] = out[label].astype(int).astype(str)
    return out


def fourth_and_one_log(df: pd.DataFrame) -> pd.DataFrame:
    plays = fourth_and_one_plays(df)
    cols = {
        "season": "Season", "week": "Week", "posteam": "Offense", "defteam": "Defense",
        "Play Type": "Type", "box_count": "Box", "coverage": "Coverage", "epa": "EPA",
        "desc": "Description",
    }
    plays = plays.assign(Result=np.where(plays["stopped"] == 1, "STOP", "Converted"))
    keep = [c for c in cols if c in plays.columns]
    out = plays[keep + ["Result"]].rename(columns=cols)
    order = ["Season", "Week", "Offense", "Defense", "Type", "Result", "Box", "Coverage",
             "EPA", "Description"]
    out = out[[c for c in order if c in out.columns]]
    return out.sort_values(["Season", "Week"], ascending=False).reset_index(drop=True)


# --- Blitz -----------------------------------------------------------------------

def blitz_plays(df: pd.DataFrame) -> pd.DataFrame:
    plays = df[df["blitz"].notna()].copy()
    plays["Pressure Call"] = np.where(plays["blitz"] == 1, "Blitz", "No Blitz")
    return plays


def blitz_vs_base(df: pd.DataFrame) -> pd.DataFrame:
    return metric_table(blitz_plays(df), "Pressure Call", ["Blitz", "No Blitz"])


def rusher_bucket(rushers: pd.Series) -> pd.Series:
    return pd.cut(rushers, bins=[0, 3, 4, 5, 6, np.inf], labels=RUSHER_BUCKETS)


def blitz_by_rushers(df: pd.DataFrame) -> pd.DataFrame:
    plays = df[df["pass_rushers"].notna()].copy()
    plays["Rushers"] = rusher_bucket(plays["pass_rushers"]).astype(str)
    return metric_table(plays, "Rushers", RUSHER_BUCKETS)


def blitz_rate_by_down(scope: Scope) -> pd.DataFrame:
    def rates(df: pd.DataFrame) -> pd.Series:
        plays = blitz_plays(df)
        plays = plays[plays["down"].between(1, 4)]
        return plays.groupby("down")["blitz"].mean()

    out = pd.DataFrame({"League %": rates(scope.league)})
    if scope.has_team:
        out.insert(0, f"{scope.label} %", rates(scope.team))
    out.index = [f"{int(d)}{'st' if d == 1 else 'nd' if d == 2 else 'rd' if d == 3 else 'th'}"
                 f" Down" for d in out.index]
    return out.rename_axis("Down").reset_index()


def team_blitz_profile(league: pd.DataFrame) -> pd.DataFrame:
    plays = blitz_plays(league)
    if plays.empty:
        return pd.DataFrame(columns=["Team", "Blitz %", "EPA/Dropback", "Blitz EPA",
                                     "Sack %", "Dropbacks"])
    grouped = plays.groupby("defteam")
    blitz_only = plays[plays["blitz"] == 1].groupby("defteam")["epa"].mean()
    out = pd.DataFrame({
        "Blitz %": grouped["blitz"].mean(),
        "EPA/Dropback": grouped["epa"].mean(),
        "Blitz EPA": blitz_only,
        "Sack %": grouped["sack"].mean(),
        "Dropbacks": grouped.size().astype(float),
    })
    return out.rename_axis("Team").reset_index().sort_values("Blitz %", ascending=False)


# --- Players ---------------------------------------------------------------------

EVENT_COLUMNS = ["play", "player_id", "name", "team", "stat", "value"]
STAT_NAMES = list(PLAYER_STATS)
PLAYER_INFO = ["player_id", "Player", "Pos", "Team"]


def player_events(df: pd.DataFrame) -> pd.DataFrame:
    """One row per defensive credit (tackle, sack, PD, ...) with the play's index.

    Tackles credited to the offense (e.g. after an interception) are dropped."""
    frames = []
    for stat, slots in PLAYER_STATS.items():
        for prefix, weight in slots:
            id_col, name_col, team_col = (f"{prefix}_player_id", f"{prefix}_player_name",
                                          f"{prefix}_team")
            if id_col not in df.columns:
                continue
            rows = df[df[id_col].notna()]
            if team_col in rows.columns:
                rows = rows[rows[team_col].isna() | (rows[team_col] == rows["defteam"])]
            if rows.empty:
                continue
            names = rows[name_col] if name_col in rows.columns else rows[id_col]
            frames.append(pd.DataFrame({
                "play": rows.index, "player_id": rows[id_col].astype(str).to_numpy(),
                "name": names.astype(str).to_numpy(), "team": rows["defteam"].to_numpy(),
                "stat": stat, "value": weight,
            }))
    if not frames:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def on_field(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (play, defender on the field), indexed by the play's index."""
    cols = ["defenders", "defender_names", "defender_positions", "defteam", "epa", "success"]
    plays = df.loc[df["defenders"].str.len() > 0, cols]
    if plays.empty:
        return pd.DataFrame(columns=["player_id", "name", "pos", "defteam", "epa", "success"])
    out = plays.explode(["defenders", "defender_names", "defender_positions"])
    return out.rename(columns={"defenders": "player_id", "defender_names": "name",
                               "defender_positions": "pos"})


def player_stats(plays: pd.DataFrame, stopped: pd.Series | None = None) -> pd.DataFrame:
    """Per-defender box score over `plays`.

    Snaps / Stop % / EPA/Play use on-field participation data (when loaded); a *stop* is a
    play where `stopped` is true (default: unsuccessful offensive play) and *Stops* counts
    such plays where the player made the tackle or sack."""
    if stopped is None:
        stopped = plays["success"].fillna(1) == 0
    stopped = stopped.reindex(plays.index).fillna(False).astype(bool)
    events = player_events(plays)
    field = on_field(plays)
    if events.empty and field.empty:
        return pd.DataFrame(columns=[*PLAYER_INFO, "Snaps", "Tackles", "Stops", *STAT_NAMES,
                                     "Stop %", "EPA/Play"])

    counts = events.pivot_table(index="player_id", columns="stat", values="value",
                                aggfunc="sum", fill_value=0.0)
    out = counts.reindex(columns=STAT_NAMES, fill_value=0.0)
    makers = events[events["stat"].isin(["Solo", "Ast", "Sacks"])
                    & events["play"].map(stopped).fillna(False).astype(bool)]
    stops = makers.groupby("player_id")["play"].nunique().astype(float)
    info = events.groupby("player_id").agg(Player=("name", "last"), Team=("team", "last"))
    info["Pos"] = ""

    if not field.empty:
        field = field.assign(stopped=stopped.loc[field.index].to_numpy(dtype=float))
        grouped = field.groupby("player_id")
        field_stats = pd.DataFrame({
            "Snaps": grouped.size().astype(float),
            "Stop %": grouped["stopped"].mean(),
            "EPA/Play": grouped["epa"].mean(),
        })
        field_info = grouped.agg(Player=("name", "last"), Team=("defteam", "last"),
                                 Pos=("pos", "last"))
        info = field_info.combine_first(info)
    else:
        field_stats = pd.DataFrame(columns=["Snaps", "Stop %", "EPA/Play"], dtype=float)

    out = out.reindex(info.index.union(out.index), fill_value=0.0)
    out["Tackles"] = out["Solo"] + out["Ast"]
    out["Stops"] = stops.reindex(out.index).fillna(0.0)
    out = out.join(field_stats).join(info)
    out = out.rename_axis("player_id").reset_index()
    ordered = [*PLAYER_INFO, "Snaps", "Tackles", "Stops", *STAT_NAMES, "Stop %", "EPA/Play"]
    return out[ordered]


def _player_sort(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return df.sort_values([*by, "Snaps"], ascending=False, na_position="last"
                          ).reset_index(drop=True)


def roster(df: pd.DataFrame) -> pd.DataFrame:
    """Defenders appearing in `df`, most snaps (or tackles) first."""
    stats = player_stats(df)
    return _player_sort(stats, ["Snaps", "Tackles"])[[*PLAYER_INFO, "Snaps", "Tackles"]]


def player_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Dropback box score: plays on the ball plus on-field EPA in man and zone."""
    plays = df[df["is_pass_play"]]
    out = player_stats(plays)
    out["Plays on Ball"] = out["PD"] + out["INT"]
    field = on_field(plays[plays["man_zone"].notna()])
    if not field.empty:
        field = field.assign(man_zone=plays.loc[field.index, "man_zone"].to_numpy())
        split = field.pivot_table(index="player_id", columns="man_zone", values="epa",
                                  aggfunc="mean")
        for scheme in ("Man", "Zone"):
            values = split[scheme] if scheme in split.columns else pd.Series(dtype=float)
            out[f"{scheme} EPA"] = out["player_id"].map(values)
    cols = [*PLAYER_INFO, "Snaps", "Plays on Ball", "PD", "INT", "Tackles", "EPA/Play",
            "Man EPA", "Zone EPA"]
    out = out[[c for c in cols if c in out.columns]]
    return _player_sort(out, ["Plays on Ball", "PD"])


def player_third_down(df: pd.DataFrame) -> pd.DataFrame:
    plays = third_down_plays(df)
    out = player_stats(plays, plays["converted"] == 0)
    cols = [*PLAYER_INFO, "Snaps", "Stops", "Tackles", "Sacks", "PD", "INT", "Stop %",
            "EPA/Play"]
    return _player_sort(out[cols], ["Stops", "Tackles"])


def player_fourth_and_one(df: pd.DataFrame) -> pd.DataFrame:
    plays = fourth_and_one_plays(df)
    out = player_stats(plays, plays["stopped"] == 1)
    cols = [*PLAYER_INFO, "Snaps", "Stops", "Tackles", "TFL", "Sacks", "Stop %", "EPA/Play"]
    return _player_sort(out[cols], ["Stops", "Tackles"])


def player_pass_rush(df: pd.DataFrame) -> pd.DataFrame:
    """Dropback pass-rush box score, with sacks / QB hits split out on blitzes."""
    plays = df[df["is_pass_play"]]
    out = player_stats(plays)
    blitz = player_events(plays[plays["blitz"] == 1])
    for stat in ("Sacks", "QB Hits"):
        totals = blitz[blitz["stat"] == stat].groupby("player_id")["value"].sum()
        out[f"Blitz {stat}"] = out["player_id"].map(totals).fillna(0.0)
    events = player_events(plays)
    sacked = set(events.loc[events["stat"] == "Sacks", "play"])
    hits = events[(events["stat"] == "QB Hits") & ~events["play"].isin(sacked)]
    out["Non-Sack Hits"] = out["player_id"].map(
        hits.groupby("player_id")["value"].sum()).fillna(0.0)
    cols = [*PLAYER_INFO, "Snaps", "Sacks", "QB Hits", "Non-Sack Hits", "Blitz Sacks",
            "Blitz QB Hits", "TFL", "FF", "EPA/Play"]
    out = out[cols]
    out = out[(out["Sacks"] > 0) | (out["QB Hits"] > 0) | (out["Snaps"] > 0)]
    return _player_sort(out, ["Sacks", "QB Hits"])
