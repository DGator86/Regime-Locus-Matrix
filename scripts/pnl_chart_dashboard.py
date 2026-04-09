#!/usr/bin/env python3
"""
Professional realized + open-MTM P/L terminal: P/L tabs, **daily activity**, **universe**, and optional **IBKR** live snapshot.

**Logs:** ``trade_log.csv`` (options), ``equity_trade_log.csv`` (equities), ``universe_trade_plans.json``.

Closed trades: ``closed=1``, one count per ``plan_id`` (last exit snapshot by time).
Open MTM: sum of ``unrealized_pnl`` on the latest row per ``plan_id`` where still open.
**Open reason** on the Activity tab comes from ``decision.rationale`` in the universe JSON when ``plan_id`` matches.

Examples::

    python scripts/pnl_chart_dashboard.py
    python scripts/pnl_chart_dashboard.py --refresh-ms 5000
    python scripts/pnl_chart_dashboard.py --universe-plans data/processed/universe_trade_plans.json
    python scripts/pnl_chart_dashboard.py --ibkr
    python scripts/pnl_chart_dashboard.py --ibkr --ibkr-refresh-ms 15000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("TkAgg")

import tkinter as tk
from tkinter import ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import dates as mdates
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPTIONS_LOG = ROOT / "data" / "processed" / "trade_log.csv"
DEFAULT_EQUITY_LOG = ROOT / "data" / "processed" / "equity_trade_log.csv"
DEFAULT_UNIVERSE_PLANS = ROOT / "data" / "processed" / "universe_trade_plans.json"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"

# Terminal palette (Bloomberg / IDE-adjacent)
BG = "#0d1117"
BG_ELEV = "#161b22"
BORDER = "#30363d"
TEXT = "#e6edf3"
TEXT_DIM = "#8b949e"
PROFIT = "#3fb950"
LOSS = "#f85149"
NEUTRAL = "#79c0ff"
ACCENT = "#58a6ff"
GRID = "#21262d"


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _is_equity_schema(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    return "action" in rows[0] and "quantity" in rows[0]


def realized_trades_dataframe(rows: list[dict[str, str]]) -> pd.DataFrame:
    closed = [r for r in rows if str(r.get("closed", "")).strip() == "1"]
    if not closed:
        return pd.DataFrame(columns=["exit_ts", "plan_id", "symbol", "pnl", "signal"])

    df = pd.DataFrame(closed)
    df["exit_ts"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df["pnl"] = pd.to_numeric(df.get("unrealized_pnl"), errors="coerce").fillna(0.0)
    df["plan_id"] = df["plan_id"].astype(str)
    df["symbol"] = df.get("symbol", "").astype(str)
    df["signal"] = df.get("signal", "").astype(str)
    df = df.dropna(subset=["exit_ts"])
    df = df.sort_values("exit_ts").groupby("plan_id", as_index=False).last()
    df = df.sort_values("exit_ts").reset_index(drop=True)
    return df[["exit_ts", "plan_id", "symbol", "pnl", "signal"]]


def open_mtm_usd(rows: list[dict[str, str]]) -> float:
    """Sum latest unrealized P/L across plans that are not closed."""
    if not rows:
        return 0.0
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df["closed"] = df["closed"].astype(str).str.strip()
    df["pnl"] = pd.to_numeric(df.get("unrealized_pnl"), errors="coerce").fillna(0.0)
    df["plan_id"] = df["plan_id"].astype(str)
    df = df.dropna(subset=["ts"])
    latest = df.sort_values("ts").groupby("plan_id", as_index=False).last()
    open_only = latest[latest["closed"] != "1"]
    return float(open_only["pnl"].sum())


def load_plan_meta_from_universe(path: Path) -> dict[str, dict[str, str]]:
    """Map ``plan_id`` → rationale / strategy / regime from ``universe_trade_plans.json``."""
    out: dict[str, dict[str, str]] = {}
    if not path.is_file():
        return out
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return out

    def ingest(section: object) -> None:
        if not isinstance(section, list):
            return
        for item in section:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("plan_id") or "").strip()
            if not pid:
                continue
            dec = item.get("decision") if isinstance(item.get("decision"), dict) else {}
            pipe = item.get("pipeline") if isinstance(item.get("pipeline"), dict) else {}
            rk = str(dec.get("regime_key") or pipe.get("regime_key") or "")
            out[pid] = {
                "rationale": str(dec.get("rationale") or ""),
                "strategy_name": str(dec.get("strategy_name") or ""),
                "regime_key": rk,
                "action": str(dec.get("action") or ""),
                "symbol": str(item.get("symbol") or ""),
            }

    # Later sections win on duplicate plan_id (active_ranked overrides results)
    ingest(data.get("results") or [])
    ingest(data.get("active_ranked") or [])
    return out


def _ts_local_str(ts: pd.Timestamp, tz: Any) -> str:
    if pd.isna(ts):
        return ""
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    return ts.tz_convert(tz).strftime("%Y-%m-%d %H:%M")


def lifecycles_from_log(rows: list[dict[str, str]], book: str) -> pd.DataFrame:
    """Per ``plan_id``: first row time, first exit time, open flag, exit details from last closed row."""
    if not rows:
        return pd.DataFrame(
            columns=[
                "book",
                "plan_id",
                "symbol",
                "first_ts",
                "first_close_ts",
                "is_open",
                "strategy_log",
                "exit_signal",
                "realized_pnl",
                "mtm_now",
            ]
        )
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    df["plan_id"] = df["plan_id"].astype(str)
    df["symbol"] = df.get("symbol", "").astype(str)
    df["is_closed_row"] = df["closed"].astype(str).str.strip() == "1"
    df["signal"] = df.get("signal", "").astype(str)
    df["strat"] = df.get("strategy", "").astype(str)

    out_rows: list[dict[str, object]] = []
    for pid, g in df.groupby("plan_id"):
        g = g.sort_values("ts")
        first_ts = g["ts"].iloc[0]
        last = g.iloc[-1]
        is_open = str(last.get("closed", "")).strip() != "1"
        strat = str(g["strat"].iloc[0] or g["strat"].iloc[-1] or "")
        closed_part = g.loc[g["is_closed_row"]]
        if len(closed_part):
            first_close_ts = closed_part["ts"].min()
            last_c = closed_part.iloc[-1]
            exit_sig = str(last_c.get("signal", "") or "")
            realized = float(pd.to_numeric(last_c.get("unrealized_pnl"), errors="coerce") or 0.0)
        else:
            first_close_ts = pd.NaT
            exit_sig = ""
            realized = 0.0
        mtm_now = float(pd.to_numeric(last.get("unrealized_pnl"), errors="coerce") or 0.0)
        out_rows.append(
            {
                "book": book,
                "plan_id": pid,
                "symbol": str(g["symbol"].iloc[0]),
                "first_ts": first_ts,
                "first_close_ts": first_close_ts,
                "is_open": is_open,
                "strategy_log": strat,
                "exit_signal": exit_sig,
                "realized_pnl": realized,
                "mtm_now": mtm_now,
            }
        )
    return pd.DataFrame(out_rows)


def _local_day(ts: pd.Timestamp) -> date | None:
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    return ts.tz_convert(_local_tzinfo()).date()


def split_activity_for_day(life: pd.DataFrame, day: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Opened on ``day``, closed on ``day``, and still open (latest log)."""
    if life.empty:
        return life.iloc[:0], life.iloc[:0], life.iloc[:0]
    opened_mask = life["first_ts"].apply(_local_day) == day
    closed_mask = life["first_close_ts"].apply(_local_day) == day
    holding_mask = life["is_open"]
    return life.loc[opened_mask], life.loc[closed_mask], life.loc[holding_mask]


def _parse_iso_date(s: str) -> date | None:
    try:
        return date.fromisoformat(s.strip()[:10])
    except ValueError:
        return None


def _file_brief(path: Path) -> str:
    if not path.is_file():
        return "—"
    try:
        sz = path.stat().st_size
    except OSError:
        return "?"
    if sz < 1024:
        return f"{sz} B"
    if sz < 1024 * 1024:
        return f"{sz // 1024} KB"
    return f"{sz // (1024 * 1024)} MB"


def universe_symbol_data_rows(symbols: list[str]) -> list[tuple[str, str, str, str, str, str]]:
    """Per symbol: presence/size of key datasets on disk."""
    rows = []
    sym_set = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
    for sym in sym_set:
        bars = DATA_RAW / f"bars_{sym}.csv"
        chain = DATA_RAW / f"option_chain_{sym}.csv"
        feats = DATA_PROC / f"features_{sym}.csv"
        ff = DATA_PROC / f"forecast_features_{sym}.csv"
        rows.append(
            (
                sym,
                _file_brief(bars),
                _file_brief(chain),
                _file_brief(feats),
                _file_brief(ff),
                _file_brief(DATA_PROC / f"backtest_equity_{sym}.csv"),
            )
        )
    return rows


def _local_tzinfo():
    return datetime.now().astimezone().tzinfo


def _exit_local_date(s: pd.Timestamp) -> date:
    if s.tzinfo is None:
        s = s.tz_localize(timezone.utc)
    return s.tz_convert(_local_tzinfo()).date()


def aggregates(df: pd.DataFrame) -> tuple[float, float, float, int]:
    if df.empty:
        return 0.0, 0.0, 0.0, 0

    today = datetime.now().astimezone().date()
    iso = today.isocalendar()
    year_w, week_w = iso.year, iso.week

    daily = 0.0
    weekly = 0.0
    for _, row in df.iterrows():
        d = _exit_local_date(row["exit_ts"])
        pnl = float(row["pnl"])
        if d == today:
            daily += pnl
        yw = d.isocalendar()
        if yw.year == year_w and yw.week == week_w:
            weekly += pnl

    all_time = float(df["pnl"].sum())
    return daily, weekly, all_time, len(df)


def daily_realized_by_date(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["day", "pnl"])
    x = df.copy()
    x["day"] = x["exit_ts"].apply(_exit_local_date)
    g = x.groupby("day", as_index=False)["pnl"].sum().sort_values("day")
    return g


def performance_stats(df: pd.DataFrame) -> dict[str, float | int]:
    out: dict[str, float | int] = {
        "n": 0,
        "wins": 0,
        "losses": 0,
        "breakeven": 0,
        "win_rate": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "best_trade": 0.0,
        "worst_trade": 0.0,
        "best_day": 0.0,
        "worst_day": 0.0,
        "profit_factor": 0.0,
    }
    if df.empty:
        return out

    n = len(df)
    pnl = df["pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    flat = pnl[pnl == 0]

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss_abs = float(abs(losses.sum())) if len(losses) else 0.0
    if gross_loss_abs > 1e-9:
        pf = gross_profit / gross_loss_abs
    else:
        pf = float("inf") if gross_profit > 1e-9 else 0.0

    by_day = daily_realized_by_date(df)
    best_d = float(by_day["pnl"].max()) if len(by_day) else 0.0
    worst_d = float(by_day["pnl"].min()) if len(by_day) else 0.0

    out.update(
        {
            "n": n,
            "wins": int((pnl > 0).sum()),
            "losses": int((pnl < 0).sum()),
            "breakeven": int((pnl == 0).sum()),
            "win_rate": 100.0 * len(wins) / n if n else 0.0,
            "avg_win": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "best_trade": float(pnl.max()),
            "worst_trade": float(pnl.min()),
            "best_day": best_d,
            "worst_day": worst_d,
            "profit_factor": pf,
        }
    )
    return out


def cumulative_series(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["exit_ts", "pnl", "cum"])
    out = df.sort_values("exit_ts").copy()
    out["cum"] = out["pnl"].cumsum()
    return out


def _money_color(v: float) -> str:
    if v > 1e-6:
        return PROFIT
    if v < -1e-6:
        return LOSS
    return TEXT


def _fmt_money(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}${v:,.2f}"


def _apply_ttk_dark(style: ttk.Style) -> None:
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure(".", background=BG, foreground=TEXT, fieldbackground=BG_ELEV)
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=TEXT_DIM, font=("Segoe UI", 9))
    style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 11, "bold"))
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", background=BG_ELEV, foreground=TEXT_DIM, padding=(14, 8))
    style.map(
        "TNotebook.Tab",
        background=[("selected", BORDER)],
        foreground=[("selected", TEXT)],
    )
    style.configure("TButton", background=BORDER, foreground=TEXT, padding=(12, 6))
    style.map("TButton", background=[("active", ACCENT)])
    style.configure(
        "Treeview",
        background=BG_ELEV,
        fieldbackground=BG_ELEV,
        foreground=TEXT,
        borderwidth=0,
        rowheight=22,
    )
    style.configure("Treeview.Heading", background=BORDER, foreground=TEXT, font=("Segoe UI", 9, "bold"))
    style.map("Treeview", background=[("selected", ACCENT)], foreground=[("selected", TEXT)])
    style.configure("TLabelframe", background=BG, foreground=TEXT_DIM)
    style.configure("TLabelframe.Label", background=BG, foreground=ACCENT, font=("Segoe UI", 9, "bold"))


def _trunc_one_line(s: str, n: int = 140) -> str:
    t = " ".join(str(s).split())
    return (t[: n - 1] + "…") if len(t) > n else t


def _make_tree_frame(
    parent: tk.Widget,
    columns: tuple[str, ...],
    headings: tuple[str, ...],
    widths: tuple[int, ...],
    *,
    height: int = 5,
) -> tuple[ttk.Treeview, tk.Frame]:
    fr = tk.Frame(parent, bg=BG)
    ys = ttk.Scrollbar(fr, orient=tk.VERTICAL)
    xs = ttk.Scrollbar(fr, orient=tk.HORIZONTAL)
    tv = ttk.Treeview(
        fr,
        columns=columns,
        show="headings",
        height=height,
        yscrollcommand=ys.set,
        xscrollcommand=xs.set,
    )
    ys.config(command=tv.yview)
    xs.config(command=tv.xview)
    for c, h, w in zip(columns, headings, widths):
        tv.heading(c, text=h)
        tv.column(c, width=w, minwidth=40)
    tv.grid(row=0, column=0, sticky="nsew")
    ys.grid(row=0, column=1, sticky="ns")
    xs.grid(row=1, column=0, sticky="ew")
    fr.rowconfigure(0, weight=1)
    fr.columnconfigure(0, weight=1)
    return tv, fr


class KPICard(tk.Frame):
    def __init__(self, parent: tk.Widget, title: str) -> None:
        super().__init__(parent, bg=BG_ELEV, highlightbackground=BORDER, highlightthickness=1)
        self._title = title
        tk.Label(self, text=title.upper(), bg=BG_ELEV, fg=TEXT_DIM, font=("Segoe UI", 8)).pack(
            anchor=tk.W, padx=12, pady=(10, 2)
        )
        self._value = tk.Label(
            self,
            text="—",
            bg=BG_ELEV,
            fg=TEXT,
            font=("Consolas", 17, "bold"),
        )
        self._value.pack(anchor=tk.W, padx=12, pady=(0, 10))

    def set_value(self, amount: float, *, neutral_ok: bool = False) -> None:
        self._value.config(text=_fmt_money(amount), fg=_money_color(amount) if not neutral_ok else TEXT)


class TradingTab(tk.Frame):
    def __init__(self, parent: tk.Widget, *, tab_name: str) -> None:
        super().__init__(parent, bg=BG)
        self._tab_name = tab_name

        kpi_row = tk.Frame(self, bg=BG)
        kpi_row.pack(fill=tk.X, padx=10, pady=(10, 6))
        for i in range(4):
            kpi_row.columnconfigure(i, weight=1, uniform="kpi")

        self._kpi_daily = KPICard(kpi_row, "Today (realized)")
        self._kpi_daily.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._kpi_week = KPICard(kpi_row, "This week (realized)")
        self._kpi_week.grid(row=0, column=1, sticky="nsew", padx=6)
        self._kpi_all = KPICard(kpi_row, "All-time (realized)")
        self._kpi_all.grid(row=0, column=2, sticky="nsew", padx=6)
        self._kpi_open = KPICard(kpi_row, "Open MTM (now)")
        self._kpi_open.grid(row=0, column=3, sticky="nsew", padx=(6, 0))

        stats = tk.Frame(self, bg=BG_ELEV, highlightbackground=BORDER, highlightthickness=1)
        stats.pack(fill=tk.X, padx=10, pady=6)
        self._stats_var = tk.StringVar(value="")
        tk.Label(
            stats,
            textvariable=self._stats_var,
            bg=BG_ELEV,
            fg=TEXT_DIM,
            font=("Consolas", 10),
            justify=tk.LEFT,
            anchor=tk.W,
        ).pack(fill=tk.X, padx=12, pady=10)

        self.fig = Figure(figsize=(10, 5.8), dpi=100, facecolor=BG)
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.08, hspace=0.22)
        self.ax_eq = self.fig.add_subplot(2, 1, 1, facecolor=BG_ELEV)
        self.ax_bar = self.fig.add_subplot(2, 1, 2, facecolor=BG_ELEV, sharex=self.ax_eq)

        for ax in (self.ax_eq, self.ax_bar):
            ax.tick_params(colors=TEXT_DIM, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(BORDER)
            ax.grid(True, color=GRID, linewidth=0.6, alpha=0.85)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().configure(bg=BG)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def _style_axes(self) -> None:
        self.ax_eq.set_title("Cumulative realized P/L", color=TEXT, fontsize=11, fontweight="bold", pad=8)
        self.ax_eq.set_ylabel("USD", color=TEXT_DIM, fontsize=9)
        self.ax_bar.set_ylabel("Daily $", color=TEXT_DIM, fontsize=9)
        self.ax_bar.set_xlabel("Date (local)", color=TEXT_DIM, fontsize=9)

    def refresh(self, df: pd.DataFrame, raw_rows: list[dict[str, str]]) -> None:
        d, w, a, n = aggregates(df)
        mtm = open_mtm_usd(raw_rows)
        st = performance_stats(df)

        self._kpi_daily.set_value(d)
        self._kpi_week.set_value(w)
        self._kpi_all.set_value(a)
        self._kpi_open.set_value(mtm)

        pf = float(st["profit_factor"])
        pf_s = "∞" if math.isinf(pf) else f"{pf:.2f}" if st["n"] else "—"
        self._stats_var.set(
            f"Trades closed: {st['n']}  ·  W / L / flat: {st['wins']} / {st['losses']} / {st['breakeven']}  ·  "
            f"Win rate: {st['win_rate']:.1f}%  ·  Avg win {_fmt_money(float(st['avg_win']))}  ·  "
            f"Avg loss {_fmt_money(float(st['avg_loss']))}  ·  Profit factor: {pf_s}  ·  "
            f"Best day {_fmt_money(float(st['best_day']))}  ·  Worst day {_fmt_money(float(st['worst_day']))}  ·  "
            f"Best trade {_fmt_money(float(st['best_trade']))}  ·  Worst trade {_fmt_money(float(st['worst_trade']))}"
        )

        self.ax_eq.clear()
        self.ax_bar.clear()
        self._style_axes()

        cum = cumulative_series(df)
        tz = _local_tzinfo()

        if cum.empty:
            self.ax_eq.text(
                0.5,
                0.55,
                "No closed trades yet.\nOpen positions still show MTM above.",
                ha="center",
                va="center",
                transform=self.ax_eq.transAxes,
                color=TEXT_DIM,
                fontsize=11,
            )
            self.ax_eq.set_xticks([])
            self.ax_eq.set_yticks([])
        else:
            x = cum["exit_ts"].dt.tz_convert(tz)
            y = cum["cum"]
            final = float(y.iloc[-1])
            line_color = PROFIT if final >= 0 else LOSS
            self.ax_eq.plot(x, y, color=line_color, linewidth=2.2, solid_capstyle="round")
            self.ax_eq.fill_between(x, y, 0, where=(y >= 0), color=PROFIT, alpha=0.12, interpolate=True)
            self.ax_eq.fill_between(x, y, 0, where=(y < 0), color=LOSS, alpha=0.12, interpolate=True)
            self.ax_eq.axhline(0, color=TEXT_DIM, linewidth=0.7, linestyle="--", alpha=0.5)
            self.ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%b %d", tz=tz))
            self.ax_eq.xaxis.set_major_locator(mdates.AutoDateLocator())

        daily = daily_realized_by_date(df)
        if len(daily) > 0:
            days_naive = pd.to_datetime(daily["day"])
            if days_naive.dt.tz is None:
                days = days_naive.dt.tz_localize(tz)
            else:
                days = days_naive.dt.tz_convert(tz)
            heights = daily["pnl"].astype(float)
            colors = [PROFIT if h >= 0 else LOSS for h in heights]
            self.ax_bar.bar(days, heights, width=0.7, color=colors, edgecolor=BORDER, linewidth=0.4, alpha=0.9)
            self.ax_bar.axhline(0, color=TEXT_DIM, linewidth=0.6, alpha=0.4)
            self.ax_bar.xaxis.set_major_formatter(mdates.DateFormatter("%b %d", tz=tz))
            self.ax_bar.xaxis.set_major_locator(mdates.AutoDateLocator())
        else:
            self.ax_bar.text(0.5, 0.5, "No daily realized bars yet", ha="center", va="center", transform=self.ax_bar.transAxes, color=TEXT_DIM)

        self.fig.autofmt_xdate()
        self.canvas.draw()


class DailyActivityTab(tk.Frame):
    """Opened / closed / holding from trade logs + open reason from universe JSON (``plan_id`` match)."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, bg=BG)
        top = tk.Frame(self, bg=BG)
        top.pack(fill=tk.X, padx=8, pady=8)
        tk.Label(top, text="Calendar day (local):", bg=BG, fg=TEXT_DIM, font=("Segoe UI", 10)).pack(side=tk.LEFT)
        self._date_var = tk.StringVar(value=date.today().isoformat())
        ttk.Entry(top, textvariable=self._date_var, width=12).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Apply date", command=self._on_apply).pack(side=tk.LEFT, padx=(0, 12))
        self._summary = tk.StringVar(value="")
        tk.Label(top, textvariable=self._summary, bg=BG, fg=TEXT, font=("Consolas", 10)).pack(side=tk.LEFT)

        self._opened_tv, f1 = _make_tree_frame(
            self,
            ("book", "symbol", "plan_id", "opened", "strategy", "open_reason"),
            ("Book", "Symbol", "Plan ID", "Opened (local)", "Strategy (log)", "Open reason (universe)"),
            (70, 70, 200, 130, 100, 360),
            height=5,
        )
        lf1 = ttk.LabelFrame(self, text="Opened on this day")
        lf1.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        f1.pack(in_=lf1, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._closed_tv, f2 = _make_tree_frame(
            self,
            ("book", "symbol", "plan_id", "closed", "exit", "pnl", "open_reason"),
            ("Book", "Symbol", "Plan ID", "Closed (local)", "Exit signal", "Realized $", "Open reason (universe)"),
            (70, 70, 200, 130, 110, 90, 320),
            height=5,
        )
        lf2 = ttk.LabelFrame(self, text="Closed on this day")
        lf2.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        f2.pack(in_=lf2, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._hold_tv, f3 = _make_tree_frame(
            self,
            ("book", "symbol", "plan_id", "opened", "strategy", "open_reason", "mtm"),
            ("Book", "Symbol", "Plan ID", "Opened (local)", "Strategy (log)", "Open reason (universe)", "MTM $"),
            (70, 70, 200, 130, 100, 320, 80),
            height=5,
        )
        lf3 = ttk.LabelFrame(self, text="Holding now (latest log row not closed)")
        lf3.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        f3.pack(in_=lf3, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._apply_cb: object | None = None

    def set_apply_callback(self, cb: object) -> None:
        self._apply_cb = cb

    def _on_apply(self) -> None:
        if callable(self._apply_cb):
            self._apply_cb()

    def refresh(
        self,
        opt_rows: list[dict[str, str]],
        eq_rows: list[dict[str, str]],
        plan_meta: dict[str, dict[str, str]],
    ) -> None:
        day = _parse_iso_date(self._date_var.get()) or date.today()
        tz = _local_tzinfo()
        lo = lifecycles_from_log(opt_rows, "OPT")
        le = lifecycles_from_log(eq_rows, "EQ")
        life = pd.concat([lo, le], ignore_index=True) if not lo.empty or not le.empty else pd.DataFrame()
        opened, closed, holding = split_activity_for_day(life, day)
        self._summary.set(
            f"Day {day.isoformat()}  ·  opened={len(opened)}  closed={len(closed)}  holding_now={len(holding)}"
        )

        def reason(pid: str) -> str:
            m = plan_meta.get(pid) or {}
            return _trunc_one_line(m.get("rationale", "") or "—")

        for tv, df, kind in (
            (self._opened_tv, opened, "o"),
            (self._closed_tv, closed, "c"),
            (self._hold_tv, holding, "h"),
        ):
            tv.delete(*tv.get_children())
            if df.empty:
                continue
            for _, r in df.iterrows():
                pid = str(r["plan_id"])
                op_r = reason(pid)
                if kind == "o":
                    tv.insert(
                        "",
                        tk.END,
                        values=(
                            r["book"],
                            r["symbol"],
                            pid,
                            _ts_local_str(r["first_ts"], tz),
                            _trunc_one_line(str(r["strategy_log"] or "—"), 80),
                            op_r,
                        ),
                    )
                elif kind == "c":
                    tv.insert(
                        "",
                        tk.END,
                        values=(
                            r["book"],
                            r["symbol"],
                            pid,
                            _ts_local_str(r["first_close_ts"], tz),
                            str(r["exit_signal"] or "—"),
                            f"{float(r['realized_pnl']):,.2f}",
                            op_r,
                        ),
                    )
                else:
                    tv.insert(
                        "",
                        tk.END,
                        values=(
                            r["book"],
                            r["symbol"],
                            pid,
                            _ts_local_str(r["first_ts"], tz),
                            _trunc_one_line(str(r["strategy_log"] or "—"), 80),
                            op_r,
                            f"{float(r['mtm_now']):,.2f}",
                        ),
                    )


class UniverseDataTab(tk.Frame):
    """``universe_trade_plans.json``: per-symbol pipeline / decision + on-disk data files."""

    def __init__(self, parent: tk.Widget, plans_path: Path) -> None:
        super().__init__(parent, bg=BG)
        self._path = plans_path
        meta = tk.Frame(self, bg=BG)
        meta.pack(fill=tk.X, padx=8, pady=6)
        self._meta_var = tk.StringVar(value="")
        tk.Label(meta, textvariable=self._meta_var, bg=BG, fg=TEXT_DIM, font=("Consolas", 9), justify=tk.LEFT).pack(
            anchor=tk.W
        )

        cols = (
            "symbol",
            "status",
            "action",
            "strategy",
            "regime",
            "plan_id",
            "rank",
            "skip",
            "rationale",
        )
        heads = (
            "Symbol",
            "Status",
            "Action",
            "Strategy",
            "Regime",
            "Plan ID",
            "Rank",
            "Skip / note",
            "Rationale (trunc)",
        )
        widths = (60, 90, 70, 110, 120, 180, 60, 200, 280)
        self._univ_tv, uf = _make_tree_frame(self, cols, heads, widths, height=12)
        lf = ttk.LabelFrame(self, text="Universe run — results & active_ranked")
        lf.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        uf.pack(in_=lf, fill=tk.BOTH, expand=True, padx=4, pady=4)

        tk.Label(self, text="Full rationale (selected row):", bg=BG, fg=TEXT_DIM, font=("Segoe UI", 9)).pack(
            anchor=tk.W, padx=10
        )
        self._rat_text = tk.Text(self, height=5, bg=BG_ELEV, fg=TEXT, insertbackground=TEXT, wrap=tk.WORD)
        self._rat_text.pack(fill=tk.X, padx=10, pady=(0, 6))
        self._univ_tv.bind("<<TreeviewSelect>>", self._on_select_univ)

        dcols = ("symbol", "bars", "chain", "features", "forecast", "bt_equity")
        dheads = ("Symbol", "bars_*.csv", "option_chain_*.csv", "features_*.csv", "forecast_features_*.csv", "backtest_equity_*.csv")
        dwidths = (70, 90, 90, 90, 110, 110)
        self._disk_tv, df = _make_tree_frame(self, dcols, dheads, dwidths, height=8)
        lf2 = ttk.LabelFrame(self, text="Data on disk (processed + raw)")
        lf2.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        df.pack(in_=lf2, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._rationales: dict[str, str] = {}

    def _on_select_univ(self, _evt: object) -> None:
        sel = self._univ_tv.selection()
        self._rat_text.delete("1.0", tk.END)
        if not sel:
            return
        vals = self._univ_tv.item(sel[0], "values")
        if not vals:
            return
        sym = str(vals[0])
        full = self._rationales.get(sym, "")
        self._rat_text.insert(tk.END, full or "(no rationale stored for this symbol in JSON)")

    def refresh(self) -> None:
        self._rationales.clear()
        self._univ_tv.delete(*self._univ_tv.get_children())
        self._disk_tv.delete(*self._disk_tv.get_children())
        self._rat_text.delete("1.0", tk.END)

        if not self._path.is_file():
            self._meta_var.set(f"(missing) {self._path}")
            return

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            self._meta_var.set(f"Invalid JSON: {e}")
            return

        gen = str(data.get("generated_at_utc", ""))
        req = data.get("symbols_requested") or []
        req_s = ", ".join(str(x) for x in req) if isinstance(req, list) else str(req)
        self._meta_var.set(f"generated_at_utc={gen}\nsymbols_requested: {req_s}")

        by_sym: dict[str, dict] = {}
        for item in data.get("results") or []:
            if isinstance(item, dict) and str(item.get("symbol", "")).strip():
                s = str(item["symbol"]).strip().upper()
                by_sym[s] = item
        for item in data.get("active_ranked") or []:
            if isinstance(item, dict) and str(item.get("symbol", "")).strip():
                s = str(item["symbol"]).strip().upper()
                by_sym[s] = item

        if isinstance(req, list) and req:
            ordered_syms = [str(s).strip().upper() for s in req if str(s).strip()]
        else:
            ordered_syms = sorted(by_sym.keys())

        for sym in ordered_syms:
            item = by_sym.get(sym)
            if item is None:
                self._univ_tv.insert(
                    "",
                    tk.END,
                    iid=f"missing:{sym}",
                    values=(sym, "—", "", "", "", "", "", "(no row in last JSON run)", ""),
                )
                self._rationales[sym] = ""
                continue
            dec = item.get("decision") if isinstance(item.get("decision"), dict) else {}
            pipe = item.get("pipeline") if isinstance(item.get("pipeline"), dict) else {}
            rat = str(dec.get("rationale") or "")
            self._rationales[sym] = rat
            rk = str(dec.get("regime_key") or pipe.get("regime_key") or "")
            try:
                rk_s = float(item.get("rank_score") or 0)
                rk_disp = f"{rk_s:.4f}" if str(item.get("status")) == "active" else ""
            except (TypeError, ValueError):
                rk_disp = str(item.get("rank_score") or "")
            skip = str(item.get("skip_reason") or "")
            pid = str(item.get("plan_id") or "")
            st = str(item.get("status") or "")
            self._univ_tv.insert(
                "",
                tk.END,
                iid=f"row:{sym}:{pid or st}",
                values=(
                    sym,
                    st,
                    str(dec.get("action", "")),
                    str(dec.get("strategy_name", "")),
                    _trunc_one_line(rk, 60),
                    pid,
                    rk_disp,
                    _trunc_one_line(skip, 120),
                    _trunc_one_line(rat, 200),
                ),
            )

        for row in universe_symbol_data_rows(ordered_syms):
            self._disk_tv.insert("", tk.END, values=row)


class IbkrLiveTab(tk.Frame):
    """Read-only positions + account summary via TWS / IB Gateway (``ibkr_snapshot``)."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, bg=BG)
        self._root = parent.winfo_toplevel()
        self._busy = False

        top = tk.Frame(self, bg=BG)
        top.pack(fill=tk.X, padx=10, pady=(10, 6))
        ttk.Button(top, text="Refresh IBKR snapshot", command=self.request_refresh).pack(side=tk.LEFT)
        self._conn_var = tk.StringVar(value="")
        tk.Label(top, textvariable=self._conn_var, bg=BG, fg=TEXT_DIM, font=("Consolas", 9)).pack(
            side=tk.LEFT, padx=(16, 0)
        )
        self._msg_var = tk.StringVar(
            value="Requires TWS or IB Gateway + ibapi. Set IBKR_DASHBOARD_CLIENT_ID to avoid clashing with other API clients."
        )
        tk.Label(
            self,
            textvariable=self._msg_var,
            bg=BG,
            fg=TEXT_DIM,
            font=("Segoe UI", 9),
            wraplength=920,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=10, pady=(0, 6))

        lf_sum = ttk.LabelFrame(self, text="Account summary")
        lf_sum.pack(fill=tk.BOTH, expand=False, padx=10, pady=4)
        cols_s = ("account", "tag", "value", "ccy")
        heads_s = ("Account", "Tag", "Value", "Ccy")
        w_s = (100, 200, 140, 56)
        self._sum_tv, sum_fr = _make_tree_frame(lf_sum, cols_s, heads_s, w_s, height=7)
        sum_fr.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        lf_pos = ttk.LabelFrame(self, text="Positions")
        lf_pos.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        cols_p = ("account", "symbol", "sec", "qty", "avg", "exch", "ccy")
        heads_p = ("Account", "Contract", "Type", "Qty", "Avg cost", "Exch", "Ccy")
        w_p = (88, 240, 48, 72, 104, 64, 44)
        self._pos_tv, pos_fr = _make_tree_frame(lf_pos, cols_p, heads_p, w_p, height=16)
        pos_fr.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    def request_refresh(self) -> None:
        if self._busy:
            return
        self._busy = True
        self._msg_var.set("Fetching snapshot…")
        root = self._root

        def work() -> None:
            snap: Any = None
            err: str | None = None
            try:
                from rlm.data.ibkr_snapshot import fetch_ibkr_account_snapshot

                snap = fetch_ibkr_account_snapshot(timeout_sec=25.0)
            except ImportError as e:
                err = str(e)
            except Exception as e:
                err = str(e)
            root.after(0, lambda s=snap, m=err: self._apply_snapshot(s, m))

        threading.Thread(target=work, daemon=True).start()

    def _apply_snapshot(self, snap: Any, err: str | None) -> None:
        self._busy = False
        for tv in (self._sum_tv, self._pos_tv):
            for iid in tv.get_children():
                tv.delete(iid)
        if err:
            self._msg_var.set(err)
            self._conn_var.set("")
            return
        if snap is None:
            self._msg_var.set("No data.")
            return
        self._conn_var.set(f"{snap.host}:{snap.port}  clientId={snap.client_id}")
        self._msg_var.set(
            f"OK — {len(snap.account_summary)} summary row(s), {len(snap.positions)} position line(s)."
        )
        for r in snap.account_summary:
            self._sum_tv.insert("", tk.END, values=(r.account, r.tag, r.value, r.currency))
        for r in snap.positions:
            disp = r.local_symbol if r.local_symbol else r.symbol
            self._pos_tv.insert(
                "",
                tk.END,
                values=(
                    r.account,
                    disp,
                    r.sec_type,
                    f"{r.position:g}",
                    f"{r.avg_cost:.4f}",
                    r.exchange,
                    r.currency,
                ),
            )


class PnLChartDashboard:
    def __init__(
        self,
        *,
        options_log: Path,
        equity_log: Path,
        universe_plans: Path,
        refresh_ms: int | None,
        ibkr_enabled: bool = False,
        ibkr_refresh_ms: int | None = None,
    ) -> None:
        self._options_log = options_log
        self._equity_log = equity_log
        self._universe_path = universe_plans
        self._refresh_ms = refresh_ms
        self._ibkr_refresh_ms = ibkr_refresh_ms
        self._ibkr_tab: IbkrLiveTab | None = None

        self.root = tk.Tk()
        self.root.title("P/L Command — RLM")
        self.root.geometry("1080x780")
        self.root.minsize(900, 640)
        self.root.configure(bg=BG)

        style = ttk.Style(self.root)
        _apply_ttk_dark(style)

        header = tk.Frame(self.root, bg=BG, height=52)
        header.pack(fill=tk.X, padx=12, pady=(12, 0))
        tk.Label(
            header,
            text="P/L COMMAND",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI", 16, "bold"),
        ).pack(side=tk.LEFT)
        tk.Label(
            header,
            text="  P/L · daily activity · universe & data files  ·  local TZ",
            bg=BG,
            fg=TEXT_DIM,
            font=("Segoe UI", 10),
        ).pack(side=tk.LEFT)

        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self._tab_opt = tk.Frame(nb, bg=BG)
        self._tab_eq = tk.Frame(nb, bg=BG)
        self._tab_activity = tk.Frame(nb, bg=BG)
        self._tab_univ = tk.Frame(nb, bg=BG)
        nb.add(self._tab_opt, text="  OPTIONS  ")
        nb.add(self._tab_eq, text="  EQUITIES  ")
        nb.add(self._tab_activity, text="  ACTIVITY  ")
        nb.add(self._tab_univ, text="  UNIVERSE  ")
        if ibkr_enabled:
            self._tab_ibkr = tk.Frame(nb, bg=BG)
            nb.add(self._tab_ibkr, text="  IBKR  ")

        self._pnl_options = TradingTab(self._tab_opt, tab_name="options")
        self._pnl_options.pack(fill=tk.BOTH, expand=True)
        self._pnl_equity = TradingTab(self._tab_eq, tab_name="equities")
        self._pnl_equity.pack(fill=tk.BOTH, expand=True)
        self._activity_tab = DailyActivityTab(self._tab_activity)
        self._activity_tab.pack(fill=tk.BOTH, expand=True)
        self._activity_tab.set_apply_callback(self._refresh_all)
        self._universe_tab = UniverseDataTab(self._tab_univ, self._universe_path)
        self._universe_tab.pack(fill=tk.BOTH, expand=True)
        if ibkr_enabled:
            self._ibkr_tab = IbkrLiveTab(self._tab_ibkr)
            self._ibkr_tab.pack(fill=tk.BOTH, expand=True)

        foot = tk.Frame(self.root, bg=BG)
        foot.pack(fill=tk.X, padx=12, pady=(0, 10))
        self._status = tk.StringVar(value="")
        ttk.Button(foot, text="Refresh  (F5)", command=self._refresh_all).pack(side=tk.LEFT)
        tk.Label(foot, textvariable=self._status, bg=BG, fg=TEXT_DIM, font=("Consolas", 9)).pack(
            side=tk.LEFT, padx=(16, 0)
        )

        self.root.bind("<F5>", lambda e: self._refresh_all())
        self.root.bind("<Control-r>", lambda e: self._refresh_all())

        self._refresh_all()
        if self._refresh_ms and self._refresh_ms >= 2000:
            self._schedule()
        if self._ibkr_tab is not None and self._ibkr_refresh_ms and self._ibkr_refresh_ms >= 10000:
            self.root.after(600, self._ibkr_poll_loop)

    def _ibkr_poll_loop(self) -> None:
        if self._ibkr_tab is None:
            return
        self._ibkr_tab.request_refresh()
        if self._ibkr_refresh_ms and self._ibkr_refresh_ms >= 10000:
            self.root.after(self._ibkr_refresh_ms, self._ibkr_poll_loop)

    def _load_options_bundle(self) -> tuple[pd.DataFrame, list[dict[str, str]]]:
        rows = _load_csv(self._options_log)
        if not rows or _is_equity_schema(rows):
            return (
                pd.DataFrame(columns=["exit_ts", "plan_id", "symbol", "pnl", "signal"]),
                [],
            )
        return realized_trades_dataframe(rows), rows

    def _load_equity_bundle(self) -> tuple[pd.DataFrame, list[dict[str, str]]]:
        rows = _load_csv(self._equity_log)
        if not rows or not _is_equity_schema(rows):
            return (
                pd.DataFrame(columns=["exit_ts", "plan_id", "symbol", "pnl", "signal"]),
                [],
            )
        return realized_trades_dataframe(rows), rows

    def _raw_option_rows(self) -> list[dict[str, str]]:
        rows = _load_csv(self._options_log)
        if not rows or _is_equity_schema(rows):
            return []
        return rows

    def _raw_equity_rows(self) -> list[dict[str, str]]:
        rows = _load_csv(self._equity_log)
        if not rows or not _is_equity_schema(rows):
            return []
        return rows

    def _refresh_all(self) -> None:
        try:
            dfo, ro = self._load_options_bundle()
            self._pnl_options.refresh(dfo, ro)
        except OSError:
            pass
        try:
            dfe, re = self._load_equity_bundle()
            self._pnl_equity.refresh(dfe, re)
        except OSError:
            pass
        try:
            meta = load_plan_meta_from_universe(self._universe_path)
            self._activity_tab.refresh(self._raw_option_rows(), self._raw_equity_rows(), meta)
        except (OSError, ValueError, KeyError):
            pass
        try:
            self._universe_tab.refresh()
        except (OSError, json.JSONDecodeError):
            pass
        self._status.set(f"Last refresh  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def _schedule(self) -> None:
        self._refresh_all()
        self.root.after(self._refresh_ms, self._schedule)

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--options-log", type=Path, default=DEFAULT_OPTIONS_LOG)
    ap.add_argument("--equity-log", type=Path, default=DEFAULT_EQUITY_LOG)
    ap.add_argument("--universe-plans", type=Path, default=DEFAULT_UNIVERSE_PLANS, help="universe_trade_plans.json")
    ap.add_argument(
        "--refresh-ms",
        type=int,
        default=0,
        help="Auto-refresh interval in ms (0 = manual only; min 2000 if set)",
    )
    ap.add_argument(
        "--ibkr",
        action="store_true",
        help="Show IBKR tab: live positions + account summary (TWS/Gateway + ibapi)",
    )
    ap.add_argument(
        "--ibkr-refresh-ms",
        type=int,
        default=0,
        help="Auto-refresh IBKR tab in ms (0 = manual only; min 10000 if set)",
    )
    args = ap.parse_args()
    opt = args.options_log if args.options_log.is_absolute() else ROOT / args.options_log
    eq = args.equity_log if args.equity_log.is_absolute() else ROOT / args.equity_log
    uni = args.universe_plans if args.universe_plans.is_absolute() else ROOT / args.universe_plans
    refresh = args.refresh_ms if args.refresh_ms >= 2000 else None
    ibkr_ms = args.ibkr_refresh_ms if args.ibkr_refresh_ms >= 10000 else None
    app = PnLChartDashboard(
        options_log=opt,
        equity_log=eq,
        universe_plans=uni,
        refresh_ms=refresh,
        ibkr_enabled=bool(args.ibkr),
        ibkr_refresh_ms=ibkr_ms,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
