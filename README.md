# NFL Defensive Analytics Tracker

A local desktop app (Python + CustomTkinter + Matplotlib) for exploring team **and
individual player** defensive performance in
[nflverse](https://github.com/nflverse/nflverse-data) play-by-play data.
Charts render natively inside the window (Matplotlib `FigureCanvasTkAgg`) with hover
tooltips plus zoom / pan / save via the embedded toolbar.

## Quick start (Windows)

1. On GitHub click the green **Code** button -> **Download ZIP**.
2. In your Downloads folder, right-click the zip -> **Extract All...** -> **Extract**.
3. In the extracted folder, double-click **`Start-Tracker-Windows.bat`**.
   - If Python is missing it opens the download page: install it (tick
     **"Add python.exe to PATH"**) and double-click the file again.
   - The first start installs the libraries (a few minutes); later starts are instant.
   - If Windows shows "Windows protected your PC", click **More info** -> **Run anyway**.
4. In the app click **Upload CSV / Parquet...** and pick your nflverse files
   (see [Getting data](#getting-data)).

## Features

- **Local file uploader** - pick one or many `.csv`, `.csv.gz` or `.parquet` files. The
  file type is detected from its columns and the tables are merged on game/play id:
  | File (nflverse release) | Required | Adds |
  | --- | --- | --- |
  | `play_by_play_YYYY` | yes | downs, distance, EPA, success, conversions, player credits (tackles, sacks, QB hits, PDs, INTs, FFs) |
  | `pbp_participation_YYYY` | for coverage tabs & player snaps | coverage shell, man/zone, pass rushers, box count, pressure, the 11 defenders on the field |
  | `ftn_charting_YYYY` | optional | blitzer counts (preferred blitz definition), box count |

  Multiple seasons can be loaded at once. A play-by-play file that was already joined with
  participation data (e.g. `nflreadr::load_participation(include_pbp = TRUE)`) also works.
- **Filters** (sidebar, apply to every tab): defense, **player**, season,
  regular season / playoffs, week range.
  - The player box lists the selected defense's defenders by snaps; type part of a name and
    press Enter to search (works with Defense = All too).
  - With a player selected, every team chart/KPI is recomputed over the snaps where that
    player was on the field (or, without participation data, plays he was credited on), and
    the player is highlighted on each Players leaderboard.
- **Tabs** - each has KPI cards (including a player leader), five chart views, a sortable
  table and CSV export:
  1. **Coverages & Zones** - coverage usage (team vs league), EPA allowed by coverage,
     man vs zone splits, league zone-rate vs EPA scatter, **Players**: plays on the ball
     (PD + INT) with on-field EPA in man and in zone.
  2. **3rd Down Efficiency** - conversion rate allowed by distance, league ranking,
     by coverage, run/pass and blitz splits, **Players**: 3rd-down stops, tackles, sacks,
     PDs, on-field stop rate.
  3. **4th & 1** - league stop-rate ranking, run vs pass, defenders in the box, full play
     log, **Players**: 4th & 1 stops, tackles, TFLs, on-field stop rate.
  4. **Blitz Defense** - blitz vs no blitz, EPA by number of pass rushers, blitz rate by
     down, league blitz-rate vs EPA scatter, **Players**: pass-rush leaders (sacks, QB hits,
     sacks / hits on blitzes).
- Players leaderboards show the top 15 as a stacked bar chart (hover for the full stat
  line); the table below lists every defender and can be sorted or exported.
- Dark / light appearance toggle; HiDPI scaling on Linux.

Definitions: scrimmage plays only (`play_type` in pass/run). *Blitz* = FTN `n_blitzers > 0`
when available, otherwise 5+ pass rushers from participation data (shown in the UI).
*4th & 1* = `down == 4 & ydstogo == 1` go-for-it plays; a *stop* is any play that did not convert.

Player stats:

| Column | Meaning |
| --- | --- |
| Snaps | plays in the current tab/filter with the player on the field (participation data) |
| Tackles | solo + assisted (tackles credited to the offense, e.g. after an INT, are ignored) |
| Sacks | full sacks + 0.5 per half sack |
| QB Hits / Non-Sack Hits | nflverse `qb_hit_*` credits (these include most sacks) / hits on plays without a sack |
| PD, INT, TFL, FF | passes defensed, interceptions, tackles for loss, forced fumbles |
| Stops | plays the player tackled or sacked on that ended in a defensive stop (failed 3rd / 4th down; unsuccessful play elsewhere) |
| Stop %, EPA/Play | defensive stop rate and EPA allowed while the player was on the field |
| Man EPA / Zone EPA | on-field EPA per dropback in man / zone coverage |

Player names, positions and teams come from the participation file (last team seen for
traded players); with play-by-play only, Snaps/Stop %/EPA are blank but box-score stats work.

## Setup

Python 3.10+ with Tk is required (on Debian/Ubuntu: `sudo apt install python3-tk`;
the python.org installers for Windows/macOS include Tk).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m nfl_defense_tracker       # optionally pass file paths to preload them
```

### Getting data

Download from the [nflverse-data releases](https://github.com/nflverse/nflverse-data/releases), e.g. for 2023:

```bash
mkdir -p data && cd data
base=https://github.com/nflverse/nflverse-data/releases/download
curl -LO $base/pbp/play_by_play_2023.parquet
curl -LO $base/pbp_participation/pbp_participation_2023.parquet
curl -LO $base/ftn_charting/ftn_charting_2023.parquet
```

Then use **Upload CSV / Parquet...** and select all three files (or
`python -m nfl_defense_tracker data/*.parquet`).

Note: nflverse participation data (coverage, rushers) is only published for some seasons
(currently 2016-2023); FTN charting starts in 2022. Tabs that need it explain what is missing.

## Development

```bash
pip install pytest ruff
python -m pytest
ruff check .
```

Layout: `data.py` (loading/merging/normalizing), `analytics.py` (pure pandas team and
player metrics),
`charts.py` (Matplotlib figures + hover metadata), `views.py` (tab/KPI definitions),
`app.py` (CustomTkinter UI).
