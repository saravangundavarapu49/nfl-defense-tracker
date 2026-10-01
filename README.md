# NFL Defensive Analytics Tracker

A local desktop app (Python + CustomTkinter + Matplotlib) for exploring defensive
performance in [nflverse](https://github.com/nflverse/nflverse-data) play-by-play data.
Charts render natively inside the window (Matplotlib `FigureCanvasTkAgg`) with hover
tooltips plus zoom / pan / save via the embedded toolbar.

## Features

- **Local file uploader** - pick one or many `.csv`, `.csv.gz` or `.parquet` files. The
  file type is detected from its columns and the tables are merged on game/play id:
  | File (nflverse release) | Required | Adds |
  | --- | --- | --- |
  | `play_by_play_YYYY` | yes | downs, distance, EPA, success, conversions |
  | `pbp_participation_YYYY` | for coverage tabs | coverage shell, man/zone, pass rushers, box count, pressure |
  | `ftn_charting_YYYY` | optional | blitzer counts (preferred blitz definition), box count |

  Multiple seasons can be loaded at once. A play-by-play file that was already joined with
  participation data (e.g. `nflreadr::load_participation(include_pbp = TRUE)`) also works.
- **Filters** (sidebar, apply to every tab): defense, season, regular season / playoffs, week range.
- **Tabs** - each has KPI cards, four chart views, a sortable table and CSV export:
  1. **Coverages & Zones** - coverage usage (team vs league), EPA allowed by coverage,
     man vs zone splits, league zone-rate vs EPA scatter.
  2. **3rd Down Efficiency** - conversion rate allowed by distance, league ranking,
     by coverage, run/pass and blitz splits.
  3. **4th & 1** - league stop-rate ranking, run vs pass, defenders in the box, full play log.
  4. **Blitz Defense** - blitz vs no blitz, EPA by number of pass rushers, blitz rate by
     down, league blitz-rate vs EPA scatter.
- Dark / light appearance toggle; HiDPI scaling on Linux.

Definitions: scrimmage plays only (`play_type` in pass/run). *Blitz* = FTN `n_blitzers > 0`
when available, otherwise 5+ pass rushers from participation data (shown in the UI).
*4th & 1* = `down == 4 & ydstogo == 1` go-for-it plays; a *stop* is any play that did not convert.

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

Layout: `data.py` (loading/merging/normalizing), `analytics.py` (pure pandas metrics),
`charts.py` (Matplotlib figures + hover metadata), `views.py` (tab/KPI definitions),
`app.py` (CustomTkinter UI).
