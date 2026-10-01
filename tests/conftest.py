import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def pbp() -> pd.DataFrame:
    """Small synthetic play-by-play table shaped like nflverse play_by_play_YYYY."""
    rows = []
    play_id = 1
    for game, (off, deff) in enumerate([("KC", "BAL"), ("BAL", "KC"), ("BUF", "BAL")]):
        game_id = f"2023_0{game + 1}_{off}_{deff}"
        for i in range(20):
            down = [1, 2, 3, 4][i % 4]
            ydstogo = 1 if down == 4 else [2, 5, 8, 12][i % 4]
            is_pass = i % 3 != 0
            converted = i % 2 == 0
            rows.append({
                "game_id": game_id, "play_id": float(play_id), "season": 2023,
                "season_type": "REG", "week": game + 1, "posteam": off, "defteam": deff,
                "down": down, "ydstogo": ydstogo, "play_type": "pass" if is_pass else "run",
                "pass": int(is_pass), "rush": int(not is_pass), "epa": 0.5 if converted else -0.5,
                "success": int(converted), "yards_gained": 6 if converted else 1,
                "first_down": int(converted), "complete_pass": int(is_pass and converted),
                "incomplete_pass": int(is_pass and not converted), "pass_attempt": int(is_pass),
                "sack": int(is_pass and i == 5), "interception": 0, "qb_hit": 0,
                "third_down_converted": int(down == 3 and converted),
                "third_down_failed": int(down == 3 and not converted),
                "fourth_down_converted": int(down == 4 and converted),
                "fourth_down_failed": int(down == 4 and not converted),
                "desc": f"play {play_id}",
            })
            play_id += 1
        rows.append({"game_id": game_id, "play_id": float(play_id), "season": 2023,
                     "season_type": "REG", "week": game + 1, "posteam": off, "defteam": deff,
                     "down": 4, "ydstogo": 10, "play_type": "punt", "pass": 0, "rush": 0,
                     "epa": 0.0, "desc": "punt"})
        play_id += 1
    return pd.DataFrame(rows)


@pytest.fixture
def participation(pbp: pd.DataFrame) -> pd.DataFrame:
    coverages = ["COVER_1", "COVER_3", "2_MAN", None]
    out = pbp[["game_id", "play_id"]].rename(columns={"game_id": "nflverse_game_id"}).copy()
    out["possession_team"] = "X"
    out["defense_coverage_type"] = [coverages[i % 4] for i in range(len(out))]
    out["defense_man_zone_type"] = out["defense_coverage_type"].map(
        {"COVER_1": "MAN_COVERAGE", "2_MAN": "MAN_COVERAGE", "COVER_3": "ZONE_COVERAGE"}
    ).fillna("")
    out["number_of_pass_rushers"] = [float([4, 5, 6, 0][i % 4]) for i in range(len(out))]
    out["defenders_in_box"] = [6 + i % 3 for i in range(len(out))]
    out["was_pressure"] = [i % 2 == 0 for i in range(len(out))]
    return out


@pytest.fixture
def ftn(pbp: pd.DataFrame) -> pd.DataFrame:
    out = pbp[["game_id", "play_id"]].rename(
        columns={"game_id": "nflverse_game_id", "play_id": "nflverse_play_id"}
    ).copy()
    out["n_blitzers"] = [[0, 0, 1, 0, 2][i % 5] for i in range(len(out))]
    out["n_pass_rushers"] = 4.0
    out["n_defense_box"] = "7"
    out["is_play_action"] = False
    out["is_screen_pass"] = False
    out.loc[out.index[-3:], "n_blitzers"] = np.nan
    return out
