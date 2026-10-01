"""Matplotlib figure builders with hover metadata for the embedded canvases."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from matplotlib.artist import Artist
from matplotlib.figure import Figure

Formatter = Callable[[float], str]

TEAM_COLOR = "#1f6aa5"
LEAGUE_COLOR = "#8a8f98"
HIGHLIGHT_COLOR = "#e8a33d"
GOOD_COLOR = "#3fa34d"
BAD_COLOR = "#d1495b"
PALETTE = [TEAM_COLOR, LEAGUE_COLOR, HIGHLIGHT_COLOR, GOOD_COLOR, BAD_COLOR, "#7b5ea7"]


def pct(value: float) -> str:
    return "-" if value is None or np.isnan(value) else f"{value * 100:.1f}%"


def dec(value: float) -> str:
    return "-" if value is None or np.isnan(value) else f"{value:+.3f}"


def num(value: float) -> str:
    return "-" if value is None or np.isnan(value) else f"{value:,.0f}"


def num2(value: float) -> str:
    return "-" if value is None or np.isnan(value) else f"{value:.2f}"


@dataclass(frozen=True)
class Theme:
    background: str
    panel: str
    text: str
    muted: str
    grid: str


DARK = Theme(background="#242424", panel="#2b2b2b", text="#e6e6e6", muted="#a0a0a0",
             grid="#3d3d3d")
LIGHT = Theme(background="#ebebeb", panel="#f7f7f7", text="#1d1d1d", muted="#5c5c5c",
              grid="#d0d0d0")


@dataclass
class HoverTarget:
    """An artist plus the tooltip text shown for it (one per point for scatters)."""

    artist: Artist
    labels: list[str]


@dataclass
class ChartResult:
    figure: Figure
    hovers: list[HoverTarget] = field(default_factory=list)


def _figure(theme: Theme) -> tuple[Figure, object]:
    fig = Figure(figsize=(9, 4.6), dpi=100, facecolor=theme.background)
    ax = fig.add_subplot(111)
    _style_axes(ax, theme)
    return fig, ax


def _style_axes(ax, theme: Theme) -> None:
    ax.set_facecolor(theme.panel)
    for spine in ax.spines.values():
        spine.set_color(theme.grid)
    ax.tick_params(colors=theme.text, labelsize=9)
    ax.yaxis.label.set_color(theme.text)
    ax.xaxis.label.set_color(theme.text)
    ax.title.set_color(theme.text)
    ax.grid(axis="y", color=theme.grid, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def _legend(ax, theme: Theme) -> None:
    legend = ax.legend(facecolor=theme.panel, edgecolor=theme.grid, fontsize=9)
    for text in legend.get_texts():
        text.set_color(theme.text)


def _finish(fig: Figure, ax, title: str, subtitle: str, theme: Theme) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=22,
                 color=theme.text)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=theme.muted)
    fig.tight_layout()


def message(text: str, theme: Theme) -> ChartResult:
    fig = Figure(figsize=(9, 4.6), dpi=100, facecolor=theme.background)
    ax = fig.add_subplot(111)
    ax.set_facecolor(theme.background)
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12, color=theme.muted,
            wrap=True, transform=ax.transAxes)
    return ChartResult(fig)


def grouped_bar(
    categories: Sequence[str],
    series: dict[str, Sequence[float]],
    *,
    title: str,
    ylabel: str,
    fmt: Formatter,
    theme: Theme,
    subtitle: str = "",
    counts: Sequence[float] | None = None,
    signed_colors: bool = False,
) -> ChartResult:
    """Side-by-side bars. `signed_colors` colors a single series green/red by sign
    (negative is good for a defense, e.g. EPA allowed)."""
    fig, ax = _figure(theme)
    hovers: list[HoverTarget] = []
    x = np.arange(len(categories))
    n = max(len(series), 1)
    width = 0.8 / n
    for i, (name, values) in enumerate(series.items()):
        vals = np.asarray(values, dtype=float)
        offset = (i - (n - 1) / 2) * width
        if signed_colors and n == 1:
            colors = [GOOD_COLOR if v <= 0 else BAD_COLOR for v in np.nan_to_num(vals)]
        else:
            colors = PALETTE[i % len(PALETTE)]
        bars = ax.bar(x + offset, np.nan_to_num(vals), width * 0.92, label=name,
                      color=colors)
        for j, (rect, value) in enumerate(zip(bars, vals, strict=False)):
            label = f"{categories[j]}\n{name}: {fmt(value)}"
            if counts is not None and i == 0:
                label += f"\nPlays: {num(counts[j])}"
            hovers.append(HoverTarget(rect, [label]))
            if len(categories) * n <= 24:
                ax.annotate(fmt(value), (rect.get_x() + rect.get_width() / 2,
                                         rect.get_height()),
                            ha="center", va="bottom" if value >= 0 else "top",
                            fontsize=8, color=theme.text, xytext=(0, 2 if value >= 0 else -2),
                            textcoords="offset points")
    ax.set_xticks(x, categories, rotation=0 if len(categories) <= 8 else 35,
                  ha="center" if len(categories) <= 8 else "right")
    ax.set_ylabel(ylabel)
    ax.axhline(0, color=theme.muted, linewidth=0.8)
    if fmt is pct:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    if n > 1:
        _legend(ax, theme)
    _finish(fig, ax, title, subtitle, theme)
    return ChartResult(fig, hovers)


def ranked_bar(
    teams: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    ylabel: str,
    fmt: Formatter,
    theme: Theme,
    highlight: str | None = None,
    subtitle: str = "",
    extra: Sequence[str] | None = None,
) -> ChartResult:
    """One bar per team, ordered as given, with the selected team highlighted."""
    fig, ax = _figure(theme)
    vals = np.asarray(values, dtype=float)
    colors = [HIGHLIGHT_COLOR if t == highlight else TEAM_COLOR for t in teams]
    bars = ax.bar(np.arange(len(teams)), np.nan_to_num(vals), color=colors, width=0.75)
    avg = float(np.nanmean(vals)) if len(vals) else np.nan
    if not np.isnan(avg):
        ax.axhline(avg, color=LEAGUE_COLOR, linestyle="--", linewidth=1,
                   label=f"League avg {fmt(avg)}")
        _legend(ax, theme)
    hovers = []
    for i, rect in enumerate(bars):
        label = f"#{i + 1} {teams[i]}: {fmt(vals[i])}"
        if extra is not None:
            label += f"\n{extra[i]}"
        hovers.append(HoverTarget(rect, [label]))
    ax.set_xticks(np.arange(len(teams)), teams, rotation=90, fontsize=8)
    ax.set_ylabel(ylabel)
    if fmt is pct:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    _finish(fig, ax, title, subtitle, theme)
    return ChartResult(fig, hovers)


def team_scatter(
    teams: Sequence[str],
    x: Sequence[float],
    y: Sequence[float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    xfmt: Formatter,
    yfmt: Formatter,
    theme: Theme,
    highlight: str | None = None,
    subtitle: str = "",
    invert_y: bool = True,
) -> ChartResult:
    """League scatter with median quadrant lines. `invert_y` puts low EPA (good) on top."""
    fig, ax = _figure(theme)
    ax.grid(axis="x", color=theme.grid, linewidth=0.6, alpha=0.8)
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    colors = [HIGHLIGHT_COLOR if t == highlight else TEAM_COLOR for t in teams]
    sizes = [140 if t == highlight else 60 for t in teams]
    points = ax.scatter(xs, ys, c=colors, s=sizes, edgecolors=theme.text, linewidths=0.4,
                        zorder=3)
    for t, xv, yv in zip(teams, xs, ys, strict=False):
        ax.annotate(t, (xv, yv), xytext=(4, 4), textcoords="offset points", fontsize=7,
                    color=theme.text, fontweight="bold" if t == highlight else "normal")
    if len(xs):
        ax.axvline(np.nanmedian(xs), color=LEAGUE_COLOR, linestyle=":", linewidth=1)
        ax.axhline(np.nanmedian(ys), color=LEAGUE_COLOR, linestyle=":", linewidth=1)
    if invert_y:
        ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xfmt is pct:
        ax.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    labels = [f"{t}\n{xlabel}: {xfmt(xv)}\n{ylabel}: {yfmt(yv)}"
              for t, xv, yv in zip(teams, xs, ys, strict=False)]
    _finish(fig, ax, title, subtitle, theme)
    return ChartResult(fig, [HoverTarget(points, labels)])


def player_bar(
    players: Sequence[str],
    series: dict[str, Sequence[float]],
    *,
    title: str,
    xlabel: str,
    fmt: Formatter,
    theme: Theme,
    subtitle: str = "",
    details: Sequence[str] | None = None,
    highlight: int | None = None,
) -> ChartResult:
    """Horizontal leaderboard (first player on top) with stacked series segments.
    `highlight` is the row index of the selected player."""
    fig, ax = _figure(theme)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=theme.grid, linewidth=0.6, alpha=0.8)
    y = np.arange(len(players))[::-1]
    left = np.zeros(len(players))
    hovers: list[HoverTarget] = []
    for i, (name, values) in enumerate(series.items()):
        vals = np.nan_to_num(np.asarray(values, dtype=float))
        bars = ax.barh(y, vals, left=left, height=0.72, label=name,
                       color=PALETTE[i % len(PALETTE)])
        for j, rect in enumerate(bars):
            if highlight == j:
                rect.set_edgecolor(HIGHLIGHT_COLOR)
                rect.set_linewidth(2)
            label = f"{players[j]}\n{name}: {fmt(vals[j])}"
            if details is not None:
                label += f"\n{details[j]}"
            hovers.append(HoverTarget(rect, [label]))
        left += vals
    for j, total in enumerate(left):
        ax.annotate(fmt(total), (total, y[j]), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=8, color=theme.text)
    ax.set_yticks(y, players, fontsize=8)
    if highlight is not None:
        tick = ax.get_yticklabels()[highlight]
        tick.set_color(HIGHLIGHT_COLOR)
        tick.set_fontweight("bold")
    ax.set_xlabel(xlabel)
    ax.margins(x=0.08)
    if len(series) > 1:
        _legend(ax, theme)
    _finish(fig, ax, title, subtitle, theme)
    return ChartResult(fig, hovers)
