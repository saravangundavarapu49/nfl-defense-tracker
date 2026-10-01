"""CustomTkinter desktop UI for the NFL defensive analytics tracker."""

from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

import customtkinter as ctk
import pandas as pd
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.collections import PathCollection

from . import analytics as an
from . import charts as ch
from .data import DataCapabilities, DataLoadError, DataStore, capabilities
from .views import TABS, TabSpec, ViewResult, format_cell

FILE_TYPES = [
    ("nflverse data", "*.csv *.csv.gz *.parquet *.pq"),
    ("CSV", "*.csv *.csv.gz"),
    ("Parquet", "*.parquet *.pq"),
    ("All files", "*.*"),
]
SEASON_TYPES = {"All": an.ALL, "Regular": "REG", "Playoffs": "POST"}
MAX_TABLE_ROWS = 2000
MAX_PLAYER_CHOICES = 40


def current_theme() -> ch.Theme:
    return ch.DARK if ctk.get_appearance_mode() == "Dark" else ch.LIGHT


class ChartHover:
    """Shows a tooltip when the mouse is over a bar or scatter point."""

    def __init__(self, canvas: FigureCanvasTkAgg, result: ch.ChartResult, theme: ch.Theme):
        self.canvas = canvas
        self.result = result
        fig = result.figure
        self.annotation = None
        if fig.axes and result.hovers:
            self.annotation = fig.axes[0].annotate(
                "", xy=(0, 0), xycoords="figure pixels", xytext=(14, 14),
                textcoords="offset points", fontsize=9, color=theme.text, zorder=10,
                bbox={"boxstyle": "round,pad=0.4", "fc": theme.panel, "ec": ch.HIGHLIGHT_COLOR,
                      "alpha": 0.95},
            )
            self.annotation.set_visible(False)
            canvas.mpl_connect("motion_notify_event", self.on_move)

    def on_move(self, event: MouseEvent) -> None:
        assert self.annotation is not None
        text = None
        if event.inaxes is not None:
            for target in self.result.hovers:
                hit, details = target.artist.contains(event)
                if not hit:
                    continue
                if isinstance(target.artist, PathCollection):
                    text = target.labels[int(details["ind"][0])]
                else:
                    text = target.labels[0]
                break
        if text is None:
            if self.annotation.get_visible():
                self.annotation.set_visible(False)
                self.canvas.draw_idle()
            return
        bbox = self.result.figure.bbox
        right = event.x > bbox.width * 0.7
        top = event.y > bbox.height * 0.55
        self.annotation.xy = (event.x, event.y)
        self.annotation.set_text(text)
        self.annotation.set_horizontalalignment("right" if right else "left")
        self.annotation.set_verticalalignment("top" if top else "bottom")
        self.annotation.xyann = (-14 if right else 14, -14 if top else 14)
        self.annotation.set_visible(True)
        self.canvas.draw_idle()


class DataTable(ctk.CTkFrame):
    """Sortable ttk.Treeview for the active view's summary table."""

    def __init__(self, master: tk.Misc):
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(self, show="headings", style="Tracker.Treeview", height=8)
        yscroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        xscroll = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self.frame = pd.DataFrame()
        self.sort_state: tuple[str, bool] | None = None

    def show(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)
        self.sort_state = None
        self._render(self.frame)

    def _render(self, frame: pd.DataFrame) -> None:
        self.tree.delete(*self.tree.get_children())
        columns = list(frame.columns)
        self.tree["columns"] = columns
        for col in columns:
            arrow = ""
            if self.sort_state and self.sort_state[0] == col:
                arrow = " ▲" if self.sort_state[1] else " ▼"
            self.tree.heading(col, text=col + arrow, command=lambda c=col: self.sort_by(c))
            wide = col == "Description"
            scale = self.winfo_fpixels("1i") / 96
            width = 520 if wide else 180 if col == "Player" else 105
            self.tree.column(col, width=int(width * scale), minwidth=60,
                             anchor="w" if wide or frame[col].dtype == object else "center",
                             stretch=not wide)
        for i, row in enumerate(frame.head(MAX_TABLE_ROWS).itertuples(index=False)):
            values = [format_cell(c, v) for c, v in zip(columns, row, strict=False)]
            self.tree.insert("", "end", values=values, tags=("odd" if i % 2 else "even",))

    def sort_by(self, column: str) -> None:
        ascending = not (self.sort_state and self.sort_state == (column, True))
        self.sort_state = (column, ascending)
        self._render(self.frame.sort_values(column, ascending=ascending, na_position="last"))


class KpiCard(ctk.CTkFrame):
    def __init__(self, master: tk.Misc, label: str, value: str, detail: str):
        super().__init__(master, corner_radius=8)
        ctk.CTkLabel(self, text=label.upper(), font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=("gray35", "gray65")).pack(anchor="w", padx=12, pady=(8, 0))
        size = 20 if len(value) <= 10 else 15 if len(value) <= 16 else 12
        ctk.CTkLabel(self, text=value, font=ctk.CTkFont(size=size, weight="bold")).pack(
            anchor="w", padx=12)
        ctk.CTkLabel(self, text=detail or " ", font=ctk.CTkFont(size=10),
                     text_color=("gray40", "gray60")).pack(anchor="w", padx=12, pady=(0, 6))


class AnalyticsPanel(ctk.CTkFrame):
    """One tab: KPI cards, a chart selector, the embedded chart and its table."""

    def __init__(self, master: tk.Misc, spec: TabSpec, on_export):
        super().__init__(master, fg_color="transparent")
        self.spec = spec
        self.on_export = on_export
        self.canvas: FigureCanvasTkAgg | None = None
        self.toolbar: NavigationToolbar2Tk | None = None
        self.hover: ChartHover | None = None
        self.scope: an.Scope | None = None
        self.caps: DataCapabilities | None = None
        self.result: ViewResult | None = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=3)
        self.grid_rowconfigure(3, weight=2)

        self.kpi_row = ctk.CTkFrame(self, fg_color="transparent")
        self.kpi_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        views = list(spec.views)
        self.selector = ctk.CTkSegmentedButton(controls, values=views,
                                               command=lambda _v: self.render())
        self.selector.set(views[0])
        self.selector.pack(side="left")
        ctk.CTkButton(controls, text="Export table CSV", width=130,
                      command=self.export).pack(side="right")

        self.chart_frame = ctk.CTkFrame(self, corner_radius=8)
        self.chart_frame.grid(row=2, column=0, sticky="nsew")
        self.placeholder = ctk.CTkLabel(
            self.chart_frame, text="Upload nflverse play-by-play files to begin.",
            font=ctk.CTkFont(size=15), text_color=("gray40", "gray60"))
        self.placeholder.pack(expand=True)

        self.table = DataTable(self)
        self.table.grid(row=3, column=0, sticky="nsew", pady=(8, 0))

    def update_data(self, scope: an.Scope, caps: DataCapabilities) -> None:
        self.scope = scope
        self.caps = caps
        self.render()

    def render(self) -> None:
        if self.scope is None or self.caps is None:
            return
        for child in self.kpi_row.winfo_children():
            child.destroy()
        kpis = self.spec.kpis(self.scope, self.caps)
        for i, kpi in enumerate(kpis):
            self.kpi_row.grid_columnconfigure(i, weight=1, uniform="kpi")
            KpiCard(self.kpi_row, kpi.label, kpi.value, kpi.detail).grid(
                row=0, column=i, sticky="ew", padx=(0 if i == 0 else 6, 0))

        builder = self.spec.views[self.selector.get()]
        theme = current_theme()
        self.result = builder(self.scope, self.caps, theme)
        self._show_chart(self.result.chart, theme)
        self.table.show(self.result.table)

    def _show_chart(self, result: ch.ChartResult, theme: ch.Theme) -> None:
        self.placeholder.pack_forget()
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        if self.toolbar is not None:
            self.toolbar.destroy()
        self.canvas = FigureCanvasTkAgg(result.figure, master=self.chart_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.chart_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True, padx=4, pady=4)
        self.hover = ChartHover(self.canvas, result, theme)
        self.canvas.draw_idle()

    def export(self) -> None:
        if self.result is None or self.result.table.empty:
            messagebox.showinfo("Export", "There is no table to export yet.")
            return
        self.on_export(self.result.table, f"{self.spec.title} - {self.selector.get()}")


class TrackerApp(ctk.CTk):
    def __init__(self, initial_files: list[str] | None = None):
        super().__init__()
        self.title("NFL Defensive Analytics Tracker")
        self.geometry("1440x920")
        self.minsize(1100, 720)
        self._apply_linux_dpi_scaling()

        self.store = DataStore()
        self.plays: pd.DataFrame | None = None
        self.caps: DataCapabilities | None = None
        self.loading = False
        self.results: queue.Queue = queue.Queue()
        self.dirty: set[str] = set()
        self.player_ids: dict[str, str] = {}
        self.roster_key: tuple | None = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._style_treeview()
        self._build_sidebar()
        self._build_main()

        if initial_files:
            self.after(200, lambda: self.load_files(initial_files))

    def _apply_linux_dpi_scaling(self) -> None:
        """CustomTkinter only auto-scales on Windows/macOS; match the X11 DPI on Linux."""
        if not sys.platform.startswith("linux"):
            return
        factor = self.winfo_fpixels("1i") / 96
        if factor > 1.1:
            ctk.set_widget_scaling(factor)
            ctk.set_window_scaling(factor)

    # --- layout ------------------------------------------------------------------

    def _build_sidebar(self) -> None:
        side = ctk.CTkScrollableFrame(self, width=270, corner_radius=0)
        side.grid(row=0, column=0, rowspan=2, sticky="nsew")
        pad = {"padx": 16, "sticky": "ew"}
        ctk.CTkLabel(side, text="NFL Defense Tracker",
                     font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, pady=(18, 0), **pad)
        ctk.CTkLabel(side, text="nflverse play-by-play analytics",
                     text_color=("gray40", "gray60")).grid(row=1, pady=(0, 12), **pad)

        self.upload_btn = ctk.CTkButton(side, text="Upload CSV / Parquet...", height=36,
                                        command=self.choose_files)
        self.upload_btn.grid(row=2, pady=(0, 6), **pad)
        ctk.CTkButton(side, text="Clear data", fg_color="transparent", border_width=1,
                      text_color=("gray10", "gray90"), command=self.clear_data).grid(
            row=3, pady=(0, 10), **pad)

        self.files_box = ctk.CTkTextbox(side, height=110, font=ctk.CTkFont(size=11),
                                        wrap="none")
        self.files_box.grid(row=4, pady=(0, 6), **pad)
        self.status_box = ctk.CTkLabel(side, text="", justify="left", anchor="w",
                                       font=ctk.CTkFont(size=11), wraplength=240)
        self.status_box.grid(row=5, pady=(0, 12), **pad)
        self._set_files_text()

        ctk.CTkLabel(side, text="FILTERS", font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=6, pady=(6, 4), **pad)

        ctk.CTkLabel(side, text="Defense", anchor="w").grid(row=7, **pad)
        self.team_menu = ctk.CTkOptionMenu(side, values=[an.ALL],
                                           command=lambda _v: self.refresh())
        self.team_menu.grid(row=8, pady=(0, 8), **pad)

        ctk.CTkLabel(side, text="Player (type to search)", anchor="w").grid(row=9, **pad)
        self.player_box = ctk.CTkComboBox(side, values=[an.ALL],
                                          command=lambda _v: self.refresh())
        self.player_box.set(an.ALL)
        self.player_box.bind("<KeyRelease>", self.on_player_typed)
        self.player_box.grid(row=10, pady=(0, 8), **pad)

        ctk.CTkLabel(side, text="Season", anchor="w").grid(row=11, **pad)
        self.season_menu = ctk.CTkOptionMenu(side, values=[an.ALL],
                                             command=lambda _v: self.refresh())
        self.season_menu.grid(row=12, pady=(0, 8), **pad)

        ctk.CTkLabel(side, text="Season type", anchor="w").grid(row=13, **pad)
        self.type_buttons = ctk.CTkSegmentedButton(side, values=list(SEASON_TYPES),
                                                   command=lambda _v: self.refresh())
        self.type_buttons.set("All")
        self.type_buttons.grid(row=14, pady=(0, 8), **pad)

        weeks = [str(w) for w in range(1, 23)]
        ctk.CTkLabel(side, text="Weeks", anchor="w").grid(row=15, **pad)
        week_row = ctk.CTkFrame(side, fg_color="transparent")
        week_row.grid(row=16, pady=(0, 8), **pad)
        week_row.grid_columnconfigure((0, 2), weight=1)
        self.week_min = ctk.CTkOptionMenu(week_row, values=weeks, width=90,
                                          command=lambda _v: self.refresh())
        self.week_min.set("1")
        self.week_min.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(week_row, text="to").grid(row=0, column=1, padx=6)
        self.week_max = ctk.CTkOptionMenu(week_row, values=weeks, width=90,
                                          command=lambda _v: self.refresh())
        self.week_max.set("22")
        self.week_max.grid(row=0, column=2, sticky="ew")

        ctk.CTkButton(side, text="Reset filters", fg_color="transparent", border_width=1,
                      text_color=("gray10", "gray90"), command=self.reset_filters).grid(
            row=17, pady=(4, 16), **pad)

        ctk.CTkLabel(side, text="Appearance", anchor="w").grid(row=18, **pad)
        appearance = ctk.CTkSegmentedButton(side, values=["Dark", "Light"],
                                            command=self.set_appearance)
        appearance.set("Dark")
        appearance.grid(row=19, pady=(0, 16), **pad)

    def _build_main(self) -> None:
        self.tabview = ctk.CTkTabview(self, command=self.on_tab_change)
        self.tabview.grid(row=0, column=1, sticky="nsew", padx=12, pady=(4, 0))
        self.panels: dict[str, AnalyticsPanel] = {}
        for spec in TABS:
            tab = self.tabview.add(spec.title)
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)
            panel = AnalyticsPanel(tab, spec, self.export_table)
            panel.grid(row=0, column=0, sticky="nsew")
            self.panels[spec.title] = panel
        self.status = ctk.CTkLabel(self, text="Ready", anchor="w", font=ctk.CTkFont(size=11),
                                   text_color=("gray35", "gray65"))
        self.status.grid(row=1, column=1, sticky="ew", padx=16, pady=(2, 6))

    def _style_treeview(self) -> None:
        style = ttk.Style(self)
        style.theme_use("default")
        dark = ctk.get_appearance_mode() == "Dark"
        bg, fg, alt, head = (("#2b2b2b", "#e6e6e6", "#323232", "#1f6aa5") if dark
                             else ("#ffffff", "#1d1d1d", "#f1f1f1", "#1f6aa5"))
        family = tkfont.nametofont("TkDefaultFont").actual()["family"]
        self.table_font = tkfont.Font(family=family, size=10)
        rowheight = self.table_font.metrics("linespace") + int(self.winfo_fpixels("1i") / 16)
        style.configure("Tracker.Treeview", background=bg, fieldbackground=bg,
                        foreground=fg, rowheight=rowheight, borderwidth=0, font=self.table_font)
        style.configure("Tracker.Treeview.Heading", background=head, foreground="white",
                        relief="flat", font=(family, 10, "bold"))
        style.map("Tracker.Treeview", background=[("selected", "#e8a33d")],
                  foreground=[("selected", "black")])
        style.map("Tracker.Treeview.Heading", background=[("active", "#144870")])
        self.row_colors = (bg, alt)
        for panel in getattr(self, "panels", {}).values():
            panel.table.tree.tag_configure("even", background=bg)
            panel.table.tree.tag_configure("odd", background=alt)

    # --- data loading ------------------------------------------------------------

    def choose_files(self) -> None:
        paths = filedialog.askopenfilenames(title="Select nflverse files", filetypes=FILE_TYPES)
        if paths:
            self.load_files(list(paths))

    def load_files(self, paths: list[str]) -> None:
        if self.loading:
            return
        self.loading = True
        self.upload_btn.configure(state="disabled", text="Loading...")
        self.set_status(f"Loading {len(paths)} file(s)...")
        threading.Thread(target=self._load_worker, args=(paths,), daemon=True).start()
        self.after(100, self._poll_load)

    def _load_worker(self, paths: list[str]) -> None:
        errors = []
        for path in paths:
            try:
                self.store.add_file(path)
            except DataLoadError as exc:
                errors.append(str(exc))
        try:
            plays = self.store.build() if self.store.pbp else None
        except DataLoadError as exc:
            errors.append(str(exc))
            plays = None
        self.results.put((plays, errors))

    def _poll_load(self) -> None:
        try:
            plays, errors = self.results.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_load)
            return
        self.loading = False
        self.upload_btn.configure(state="normal", text="Upload CSV / Parquet...")
        if errors:
            messagebox.showwarning("Some files were not loaded", "\n\n".join(errors))
        self.plays = plays
        self.caps = capabilities(plays) if plays is not None else None
        self._set_files_text()
        if plays is None:
            self.set_status("Waiting for a play-by-play file.")
            return
        self._populate_filters(plays)
        self.refresh()

    def clear_data(self) -> None:
        self.store.clear()
        self.plays = None
        self.caps = None
        self._set_files_text()
        self.team_menu.configure(values=[an.ALL])
        self.season_menu.configure(values=[an.ALL])
        self.player_ids = {}
        self.roster_key = None
        self.player_box.configure(values=[an.ALL])
        self.player_box.set(an.ALL)
        for panel in self.panels.values():
            panel.scope = None
        self.set_status("Data cleared.")

    def _set_files_text(self) -> None:
        self.files_box.configure(state="normal")
        self.files_box.delete("1.0", "end")
        if not self.store.files:
            self.files_box.insert("end", "No files loaded.\n\nAccepted: nflverse\n"
                                         " play_by_play_YYYY\n pbp_participation_YYYY\n"
                                         " ftn_charting_YYYY")
        for f in self.store.files:
            self.files_box.insert("end", f"[{f.kind}] {f.path.name} ({f.rows:,})\n")
        self.files_box.configure(state="disabled")
        if self.caps is None or self.plays is None:
            self.status_box.configure(text="")
            return
        yes = "available"
        no = "missing"
        self.status_box.configure(text=(
            f"Scrimmage plays: {len(self.plays):,}\n"
            f"Coverage data: {yes if self.caps.coverage else no}\n"
            f"Blitz data: {self.caps.blitz_source}\n"
            f"Pressure data: {yes if self.caps.pressure else no}\n"
            f"Box counts: {yes if self.caps.box else no}\n"
            f"Player stats: {yes if self.caps.players else no}\n"
            f"On-field defenders: {yes if self.caps.on_field else no}"))

    def _populate_filters(self, plays: pd.DataFrame) -> None:
        teams = sorted(plays["defteam"].dropna().unique().tolist())
        seasons = sorted({int(s) for s in plays["season"].dropna().unique()}, reverse=True)
        self.team_menu.configure(values=[an.ALL, *teams])
        if self.team_menu.get() not in teams:
            self.team_menu.set(an.ALL)
        season_values = [an.ALL, *[str(s) for s in seasons]]
        self.season_menu.configure(values=season_values)
        if self.season_menu.get() not in season_values:
            self.season_menu.set(str(seasons[0]) if seasons else an.ALL)

    # --- filtering & rendering ---------------------------------------------------

    def current_filters(self) -> an.Filters:
        season = self.season_menu.get()
        lo, hi = int(self.week_min.get()), int(self.week_max.get())
        label = self.player_box.get()
        return an.Filters(
            seasons=() if season == an.ALL else (int(season),),
            season_type=SEASON_TYPES[self.type_buttons.get()],
            team=self.team_menu.get(),
            week_min=min(lo, hi),
            week_max=max(lo, hi),
            player_id=self.player_ids.get(label, an.ALL),
            player_name=label.split(" (")[0] if label in self.player_ids else "",
        )

    def _update_roster(self, filters: an.Filters) -> an.Filters:
        """Refresh the player picker for the current defense/season/week filters and drop
        a selected player who no longer appears in them."""
        key = (filters.seasons, filters.season_type, filters.team, filters.week_min,
               filters.week_max)
        if key == self.roster_key or self.plays is None:
            return filters
        self.roster_key = key
        defense = an.apply_filters(self.plays, replace(filters, player_id=an.ALL)).team
        players = an.roster(defense)
        self.player_ids = {}
        for row in players.itertuples(index=False):
            pos = f"{row.Pos}, " if row.Pos else ""
            label = f"{row.Player} ({pos}{row.Team})"
            if label in self.player_ids:
                label = f"{label} [{row.player_id}]"
            self.player_ids[label] = row.player_id
        self.player_box.configure(values=[an.ALL, *list(self.player_ids)[:MAX_PLAYER_CHOICES]])
        if self.player_box.get() not in self.player_ids:
            self.player_box.set(an.ALL)
            return replace(filters, player_id=an.ALL, player_name="")
        return filters

    def on_player_typed(self, event: tk.Event) -> None:
        text = self.player_box.get().strip().lower()
        if event.keysym == "Return":
            matches = [lbl for lbl in self.player_ids if text in lbl.lower()]
            exact = [lbl for lbl in matches if lbl.lower().startswith(text)]
            if text in ("", an.ALL.lower()):
                self.player_box.set(an.ALL)
            elif matches:
                self.player_box.set((exact or matches)[0])
            else:
                self.set_status(f"No defender matches '{text}'.")
                return
            self.refresh()
            return
        matches = [lbl for lbl in self.player_ids if text in lbl.lower()] if text else list(
            self.player_ids)
        self.player_box.configure(values=[an.ALL, *matches[:MAX_PLAYER_CHOICES]])

    def refresh(self) -> None:
        if self.plays is None or self.caps is None:
            return
        filters = self._update_roster(self.current_filters())
        scope = an.apply_filters(self.plays, filters)
        for panel in self.panels.values():
            panel.scope = scope
            panel.caps = self.caps
        self.dirty = set(self.panels)
        self._render_active()
        who = filters.team if filters.team != an.ALL else "all defenses"
        if scope.has_player:
            who = f"{scope.player_name} on the field ({who})"
        self.set_status(f"{len(scope.team):,} plays for {who} "
                        f"({len(scope.league):,} league plays in filter). Hover charts for "
                        "details; use the toolbar to zoom, pan or save.")

    def _render_active(self) -> None:
        name = self.tabview.get()
        if name in self.dirty:
            self.dirty.discard(name)
            self.panels[name].render()

    def on_tab_change(self) -> None:
        self._render_active()

    def reset_filters(self) -> None:
        self.team_menu.set(an.ALL)
        self.player_box.set(an.ALL)
        self.type_buttons.set("All")
        self.week_min.set("1")
        self.week_max.set("22")
        if self.plays is not None:
            self._populate_filters(self.plays)
        self.refresh()

    def set_appearance(self, mode: str) -> None:
        ctk.set_appearance_mode(mode)
        self._style_treeview()
        if self.plays is not None:
            self.dirty = set(self.panels)
            self._render_active()

    def export_table(self, table: pd.DataFrame, title: str) -> None:
        default = title.lower().replace(" & ", "_").replace(" - ", "_").replace(" ", "_")
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=default,
                                            filetypes=[("CSV", "*.csv")])
        if path:
            table.to_csv(path, index=False)
            self.set_status(f"Exported {len(table):,} rows to {Path(path).name}")

    def set_status(self, text: str) -> None:
        self.status.configure(text=text)


def run(initial_files: list[str] | None = None) -> None:
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    app = TrackerApp(initial_files)
    app.mainloop()
