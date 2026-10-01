"""Pure pandas analytics for each tracker tab."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import COVERAGE_ORDER

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


@dataclass(frozen=True)
class Scope:
    """League-wide plays (all filters but team) and the selected defense's plays."""

    league: pd.DataFrame
    team: pd.DataFrame
    team_name: str

    @property
    def has_team(self) -> bool:
        return self.team_name != ALL


def apply_filters(df: pd.DataFrame, filters: Filters) -> Scope:
    mask = pd.Series(True, index=df.index)
    if filters.seasons:
        mask &= df["season"].isin(filters.seasons)
    if filters.season_type != ALL:
        mask &= df["season_type"] == filters.season_type
    mask &= df["week"].between(filters.week_min, filters.week_max) | df["week"].isna()
    league = df[mask]
    team = league[league["defteam"] == filters.team] if filters.team != ALL else league
    return Scope(league=league, team=team, team_name=filters.team)


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
        out.insert(0, f"{scope.team_name} %", shares(scope.team))
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
        out.insert(0, f"{scope.team_name} %", rates(scope.team))
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
