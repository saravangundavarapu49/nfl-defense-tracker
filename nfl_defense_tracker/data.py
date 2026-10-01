"""Loading, merging and normalizing nflverse play-by-play data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SUPPORTED_SUFFIXES = (".csv", ".csv.gz", ".parquet", ".pq")

PBP_COLUMNS = [
    "game_id", "play_id", "season", "season_type", "week", "posteam", "defteam",
    "qtr", "down", "ydstogo", "yardline_100", "play_type", "pass", "rush",
    "qb_scramble", "epa", "success", "yards_gained", "air_yards", "first_down",
    "third_down_converted", "third_down_failed", "fourth_down_converted",
    "fourth_down_failed", "sack", "qb_hit", "interception", "complete_pass",
    "incomplete_pass", "pass_attempt", "touchdown", "desc",
    # Present when the file was pre-merged with participation/FTN data.
    "defense_coverage_type", "defense_man_zone_type", "number_of_pass_rushers",
    "defenders_in_box", "was_pressure", "defense_personnel", "time_to_throw",
    "n_blitzers", "n_pass_rushers", "n_defense_box", "is_play_action",
    "is_screen_pass",
]
PARTICIPATION_COLUMNS = [
    "defense_coverage_type", "defense_man_zone_type", "number_of_pass_rushers",
    "defenders_in_box", "was_pressure", "defense_personnel", "time_to_throw",
]
FTN_COLUMNS = [
    "n_blitzers", "n_pass_rushers", "n_defense_box", "is_play_action",
    "is_screen_pass",
]

COVERAGE_LABELS = {
    "COVER_0": "Cover 0",
    "COVER_1": "Cover 1",
    "COVER_2": "Cover 2",
    "2_MAN": "2-Man",
    "COVER_3": "Cover 3",
    "COVER_4": "Cover 4",
    "COVER_6": "Cover 6",
    "COVER_9": "Cover 9",
    "COMBO": "Combo",
    "BLOWN": "Blown",
    "PREVENT": "Prevent",
}
COVERAGE_ORDER = list(COVERAGE_LABELS.values())
MAN_ZONE_LABELS = {"MAN_COVERAGE": "Man", "ZONE_COVERAGE": "Zone"}


class DataLoadError(Exception):
    """Raised when a file cannot be read or recognized."""


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    name = path.name.lower()
    if not path.exists():
        raise DataLoadError(f"File not found: {path}")
    try:
        if name.endswith((".parquet", ".pq")):
            return pd.read_parquet(path)
        if name.endswith((".csv", ".csv.gz")):
            return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise DataLoadError(f"Could not read {path.name}: {exc}") from exc
    raise DataLoadError(
        f"Unsupported file type: {path.name} (expected {', '.join(SUPPORTED_SUFFIXES)})"
    )


def detect_kind(df: pd.DataFrame) -> str:
    """Classify an nflverse table as 'pbp', 'participation' or 'ftn'."""
    cols = set(df.columns)
    if {"defteam", "down", "play_id"} <= cols and ("game_id" in cols):
        return "pbp"
    if {"nflverse_game_id", "play_id"} <= cols and (
        {"defense_coverage_type", "number_of_pass_rushers"} & cols
    ):
        return "participation"
    if {"nflverse_game_id", "nflverse_play_id"} <= cols and "n_blitzers" in cols:
        return "ftn"
    raise DataLoadError(
        "Unrecognized file. Expected nflverse play-by-play, pbp_participation "
        "or ftn_charting data."
    )


def _keep(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[c for c in columns if c in df.columns]].copy()


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _to_bool_float(series: pd.Series) -> pd.Series:
    mapped = series.map(
        lambda v: 1.0 if v in (True, 1, "1", "TRUE", "True", "true")
        else 0.0 if v in (False, 0, "0", "FALSE", "False", "false")
        else np.nan
    )
    return mapped.astype(float)


@dataclass
class LoadedFile:
    path: Path
    kind: str
    rows: int


@dataclass
class DataStore:
    """Holds uploaded nflverse tables and produces a merged, normalized play table."""

    pbp: list[pd.DataFrame] = field(default_factory=list)
    participation: list[pd.DataFrame] = field(default_factory=list)
    ftn: list[pd.DataFrame] = field(default_factory=list)
    files: list[LoadedFile] = field(default_factory=list)

    def add_file(self, path: str | Path) -> LoadedFile:
        df = read_table(path)
        return self.add_frame(df, Path(path))

    def add_frame(self, df: pd.DataFrame, path: Path) -> LoadedFile:
        kind = detect_kind(df)
        if kind == "pbp":
            self.pbp.append(_keep(df, PBP_COLUMNS))
        elif kind == "participation":
            part = _keep(df, ["nflverse_game_id", "play_id", *PARTICIPATION_COLUMNS])
            self.participation.append(part)
        else:
            ftn = _keep(df, ["nflverse_game_id", "nflverse_play_id", *FTN_COLUMNS])
            self.ftn.append(ftn)
        loaded = LoadedFile(path=path, kind=kind, rows=len(df))
        self.files.append(loaded)
        return loaded

    def clear(self) -> None:
        self.pbp.clear()
        self.participation.clear()
        self.ftn.clear()
        self.files.clear()

    def build(self) -> pd.DataFrame:
        if not self.pbp:
            raise DataLoadError(
                "Load at least one nflverse play-by-play file (play_by_play_YYYY)."
            )
        pbp = pd.concat(self.pbp, ignore_index=True)
        pbp["play_id"] = _to_num(pbp["play_id"])
        pbp = pbp.drop_duplicates(["game_id", "play_id"])

        if self.participation:
            part = pd.concat(self.participation, ignore_index=True)
            part["play_id"] = _to_num(part["play_id"])
            part = part.drop_duplicates(["nflverse_game_id", "play_id"])
            part = part.rename(columns={"nflverse_game_id": "game_id"})
            pbp = pbp.drop(columns=[c for c in PARTICIPATION_COLUMNS if c in pbp.columns])
            pbp = pbp.merge(part, on=["game_id", "play_id"], how="left")

        if self.ftn:
            ftn = pd.concat(self.ftn, ignore_index=True)
            ftn["nflverse_play_id"] = _to_num(ftn["nflverse_play_id"])
            ftn = ftn.drop_duplicates(["nflverse_game_id", "nflverse_play_id"])
            ftn = ftn.rename(
                columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"}
            )
            pbp = pbp.drop(columns=[c for c in FTN_COLUMNS if c in pbp.columns])
            pbp = pbp.merge(ftn, on=["game_id", "play_id"], how="left")

        return normalize(pbp)


def normalize(pbp: pd.DataFrame) -> pd.DataFrame:
    """Restrict to scrimmage plays and derive the defensive analysis columns."""
    df = pbp.copy()
    for col in ("season", "week", "down", "ydstogo", "pass", "rush", "epa", "success",
                "yards_gained", "first_down", "third_down_converted", "third_down_failed",
                "fourth_down_converted", "fourth_down_failed", "sack", "qb_hit",
                "interception", "complete_pass", "incomplete_pass", "pass_attempt",
                "qb_scramble", "touchdown"):
        if col in df.columns:
            df[col] = _to_num(df[col])
        else:
            df[col] = np.nan
    if "season_type" not in df.columns:
        df["season_type"] = "REG"

    df = df[df["play_type"].isin(["pass", "run"]) & df["defteam"].notna()].copy()
    df["dropback"] = (df["pass"] == 1).astype(float)
    df["is_pass_play"] = df["dropback"] == 1

    if "defense_coverage_type" in df.columns:
        raw = df["defense_coverage_type"].astype("string").str.strip().str.upper()
        df["coverage"] = raw.map(COVERAGE_LABELS).where(raw.notna(), np.nan)
        unknown = raw.notna() & (raw != "") & df["coverage"].isna()
        df.loc[unknown, "coverage"] = raw[unknown].str.replace("_", " ").str.title()
    else:
        df["coverage"] = np.nan

    if "defense_man_zone_type" in df.columns:
        mz = df["defense_man_zone_type"].astype("string").str.strip().str.upper()
        df["man_zone"] = mz.map(MAN_ZONE_LABELS)
    else:
        df["man_zone"] = np.nan

    rushers = pd.Series(np.nan, index=df.index)
    if "number_of_pass_rushers" in df.columns:
        rushers = _to_num(df["number_of_pass_rushers"])
    if "n_pass_rushers" in df.columns:
        rushers = rushers.where(rushers > 0, _to_num(df["n_pass_rushers"]))
    rushers = rushers.where((rushers > 0) & df["is_pass_play"])
    df["pass_rushers"] = rushers

    blitzers = (
        _to_num(df["n_blitzers"]) if "n_blitzers" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    blitzers = blitzers.where(df["is_pass_play"])
    df["blitzers"] = blitzers
    blitz = pd.Series(np.nan, index=df.index)
    blitz = blitz.where(blitzers.isna(), (blitzers > 0).astype(float))
    fallback = blitz.isna() & rushers.notna()
    blitz[fallback] = (rushers[fallback] >= 5).astype(float)
    df["blitz"] = blitz

    box = pd.Series(np.nan, index=df.index)
    if "defenders_in_box" in df.columns:
        box = _to_num(df["defenders_in_box"])
    if "n_defense_box" in df.columns:
        box = box.where(box > 0, _to_num(df["n_defense_box"]))
    df["box_count"] = box.where(box > 0)

    if "was_pressure" in df.columns:
        df["pressure"] = _to_bool_float(df["was_pressure"]).where(df["is_pass_play"])
    else:
        df["pressure"] = np.nan

    return df.reset_index(drop=True)


@dataclass(frozen=True)
class DataCapabilities:
    coverage: bool
    man_zone: bool
    blitz: bool
    blitz_source: str
    pressure: bool
    box: bool


def capabilities(df: pd.DataFrame) -> DataCapabilities:
    has_blitzers = df["blitzers"].notna().any()
    has_rushers = df["pass_rushers"].notna().any()
    if has_blitzers:
        source = "FTN blitzer counts"
    elif has_rushers:
        source = "5+ pass rushers"
    else:
        source = "unavailable"
    return DataCapabilities(
        coverage=bool(df["coverage"].notna().any()),
        man_zone=bool(df["man_zone"].notna().any()),
        blitz=bool(df["blitz"].notna().any()),
        blitz_source=source,
        pressure=bool(df["pressure"].notna().any()),
        box=bool(df["box_count"].notna().any()),
    )
