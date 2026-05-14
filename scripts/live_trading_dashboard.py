#!/usr/bin/env python3
"""
Single-window dashboard: options dry-run log, equity log, and multi-timeframe regime.

- **Options** rows come from ``data/processed/trade_log.csv`` (monitor / dry-run).
  Open positions are **one row per symbol** (newest log line per underlying; intraday
  ``plan_id`` churn no longer fills the table).
- **Equity** rows come from ``data/processed/equity_trade_log.csv`` (same per-symbol rule).
- **Algo (ROEE)** tab reads ``universe_trade_plans.json`` ``results``: action, strategy,
  rationale, skip_reason, pipeline scores — select a row for full detail.
- **5m / 30m / 1h regime** uses the same factor → state-matrix → forecast path as
  ``run_universe_options_pipeline.py``, with IBKR historical bars per bar size.

Requires TWS / IB Gateway for regime panels (separate client id from other scripts).

Examples::

    python scripts/live_trading_dashboard.py
    python scripts/live_trading_dashboard.py --symbol QQQ --no-auto-regime

Log and regime auto-refresh intervals are scheduled on **wall-clock-aligned** boundaries
(multiples of the chosen period in ms since the Unix epoch), so cadence stays in step with
system time instead of drifting with Tk event-loop latency.

Environment
-----------
``IBKR_DASHBOARD_CLIENT_ID`` — API client id for this app (default **88**).
Other IBKR vars follow ``rlm.data.ibkr_stocks`` (host, port, etc.).
"""

from __future__ import annotations

import csv
import json
import os
import sys
import threading
import time
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DEFAULT_OPTIONS_LOG = ROOT / "data" / "processed" / "trade_log.csv"
DEFAULT_EQUITY_LOG = ROOT / "data" / "processed" / "equity_trade_log.csv"
DEFAULT_PLANS = ROOT / "data" / "processed" / "universe_trade_plans.json"

# (label, bar_size, duration, move_window, vol_window) — aligned with universe pipeline hints.
_REGIME_TIMEFRAMES: tuple[tuple[str, str, str, int, int], ...] = (
    ("5m", "5 mins", "10 D", 390, 390),
    ("30m", "30 mins", "60 D", 130, 130),
    ("1h", "1 hour", "180 D", 100, 100),
)

_DIRECTION_COLORS = {
    "bull": "#1a7f37",
    "bear": "#cf222e",
    "range": "#0969da",
    "transition": "#bc4c00",
    "unknown": "#656d76",
}


def _delay_ms_to_next_system_tick(period_ms: int) -> int:
    """Ms to wait until the next boundary aligned with ``time.time()`` (epoch ms).

    Repeating ``after(delay, ...)`` with a fixed delay lets Tk processing jitter accumulate;
    aligning to absolute multiples of ``period_ms`` keeps the cadence locked to the system clock.
    """
    period_ms = max(1, int(period_ms))
    now_ms = int(time.time() * 1000)
    rem = now_ms % period_ms
    if rem == 0:
        return period_ms
    return period_ms - rem


def _dashboard_client_id() -> int:
    return int(os.environ.get("IBKR_DASHBOARD_CLIENT_ID") or "88")


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _parse_ts_utc(raw: str) -> datetime | None:
    s = (raw or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _latest_per_plan(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Last row in file order per ``plan_id`` (legacy helper)."""
    seen: dict[str, dict[str, str]] = {}
    for row in rows:
        pid = row.get("plan_id") or ""
        seen[pid] = row
    return list(seen.values())


def _latest_open_row_per_symbol(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """One row per underlying: newest ``timestamp_utc`` among open (``closed`` != 1) rows.

    The monitor appends a line every poll; pipeline reruns create new ``plan_id`` strings,
    so ``_latest_per_plan`` would list every intraday plan. Traders usually want the current
    book keyed by symbol.
    """
    best: dict[str, tuple[datetime, dict[str, str]]] = {}
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    for row in rows:
        if str(row.get("closed", "")).strip() == "1":
            continue
        sym = (row.get("symbol") or "").strip().upper()
        if not sym:
            continue
        ts = _parse_ts_utc(row.get("timestamp_utc", "")) or epoch
        prev = best.get(sym)
        if prev is None or ts >= prev[0]:
            best[sym] = (ts, row)
    out = [t[1] for t in best.values()]
    out.sort(key=lambda r: (r.get("symbol") or "").upper())
    return out


def _is_equity_schema(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    cols = set(rows[0].keys())
    return "action" in cols and "quantity" in cols


def _load_plans_summary(path: Path) -> str:
    if not path.is_file():
        return f"(no plans file: {path.name})"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "(invalid JSON in plans file)"
    ranked = data.get("active_ranked") or []
    if not ranked:
        gen = data.get("generated_at_utc", "")
        return f"No active_ranked plans. generated_at_utc={gen}"
    lines = []
    for item in ranked[:12]:
        if not isinstance(item, dict):
            continue
        sym = item.get("symbol", "?")
        rk = item.get("regime_key", item.get("pipeline", {}).get("regime_key", ""))
        st = item.get("status", "")
        lines.append(f"  {sym}  [{st}]  {rk}")
    if len(ranked) > 12:
        lines.append(f"  … +{len(ranked) - 12} more")
    return "\n".join(lines) if lines else "(empty active_ranked)"


def _load_algo_decision_rows(path: Path) -> list[dict[str, object]]:
    """Per-symbol rows from ``universe_trade_plans.json`` ``results`` for ROEE / entry visibility."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    results = data.get("results") or []
    out: list[dict[str, object]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        sym = str(item.get("symbol", "")).strip().upper()
        if not sym:
            continue
        dec = item.get("decision") if isinstance(item.get("decision"), dict) else {}
        pipe = item.get("pipeline") if isinstance(item.get("pipeline"), dict) else {}
        meta = dec.get("metadata") if isinstance(dec.get("metadata"), dict) else {}
        conf = meta.get("confidence")
        conf_s = ""
        try:
            if conf is not None and conf != "":
                conf_s = f"{float(conf):.4f}"
        except (TypeError, ValueError):
            pass
        sf = dec.get("size_fraction")
        sf_s = ""
        try:
            if sf is not None and sf != "":
                sf_s = f"{float(sf):.4f}"
        except (TypeError, ValueError):
            pass
        rationale = str(dec.get("rationale", "") or "")
        lines_detail = [
            f"symbol={sym}",
            f"status={item.get('status', '')}",
            f"skip_reason={item.get('skip_reason', '')}",
            f"plan_id={item.get('plan_id', '')}",
            f"run_at_utc={item.get('run_at_utc', '')}",
            "",
            f"decision.action={dec.get('action', '')}",
            f"decision.strategy_name={dec.get('strategy_name', '')}",
            f"decision.regime_key={dec.get('regime_key', '')}",
            f"decision.size_fraction={sf_s}",
            f"decision.metadata.confidence={conf_s}",
            "",
            f"pipeline.regime_key={pipe.get('regime_key', '')}",
            f"pipeline.close={pipe.get('close', '')}",
            f"pipeline.sigma={pipe.get('sigma', '')}",
            f"pipeline S_D,S_V,S_L,S_G={pipe.get('S_D', '')}, {pipe.get('S_V', '')}, "
            f"{pipe.get('S_L', '')}, {pipe.get('S_G', '')}",
            "",
            "rationale:",
            rationale or "(none)",
        ]
        out.append(
            {
                "symbol": sym,
                "status": str(item.get("status", "")),
                "action": str(dec.get("action", "")),
                "strategy": str(dec.get("strategy_name", "")),
                "regime_key": str(dec.get("regime_key", "") or pipe.get("regime_key", "")),
                "skip_reason": str(item.get("skip_reason", "") or ""),
                "rationale_short": (rationale[:100] + "…") if len(rationale) > 100 else rationale,
                "detail": "\n".join(lines_detail),
            }
        )
    out.sort(key=lambda r: str(r["symbol"]))
    return out


def compute_regime_timeframes(
    symbol: str,
    *,
    attach_vix: bool = True,
    client_id: int | None = None,
) -> dict[str, dict[str, str]]:
    """Return per-label dict with direction_regime, regime_key, bar_time (last bar timestamp)."""
    from rlm.features.factors.pipeline import FactorPipeline

    from rlm.data.ibkr_stocks import fetch_historical_stock_bars
    from rlm.data.bars_enrichment import prepare_bars_for_factors
    from rlm.forecasting.engines import ForecastPipeline
    from rlm.features.scoring.state_matrix import classify_state_matrix

    sym = symbol.strip().upper()
    if not sym:
        raise ValueError("Empty symbol")

    cid = client_id if client_id is not None else _dashboard_client_id()
    out: dict[str, dict[str, str]] = {}

    for label, bar_size, duration, move_w, vol_w in _REGIME_TIMEFRAMES:
        bars = fetch_historical_stock_bars(
            sym,
            duration=duration,
            bar_size=bar_size,
            timeout_sec=120.0,
            client_id=cid,
        )
        if bars.empty:
            out[label] = {
                "direction_regime": "unknown",
                "regime_key": "",
                "bar_time": "",
                "detail": "no bars",
            }
            continue

        df = bars.sort_values("timestamp").set_index("timestamp")
        df = prepare_bars_for_factors(df, option_chain=None, underlying=sym, attach_vix=attach_vix)
        feats = FactorPipeline().run(df)
        feats = classify_state_matrix(feats)
        forecast = ForecastPipeline(move_window=move_w, vol_window=vol_w)
        scored = forecast.run(feats)
        last = scored.iloc[-1]
        ts = last.name
        ts_s = ""
        if isinstance(ts, pd.Timestamp):
            ts_s = ts.isoformat()
        dr = str(last.get("direction_regime", "unknown"))
        rk = str(last.get("regime_key", ""))
        out[label] = {
            "direction_regime": dr,
            "regime_key": rk,
            "bar_time": ts_s,
            "detail": f"{bar_size} · {duration}",
        }
    return out


class LiveTradingDashboard:
    def __init__(self, *, auto_regime: bool = True) -> None:
        self.root = tk.Tk()
        self.root.title("RLM — dry-run, equity & regimes")
        self.root.geometry("1100x720")
        self.root.minsize(880, 560)
        self._auto_regime = auto_regime

        self._options_log = DEFAULT_OPTIONS_LOG
        self._equity_log = DEFAULT_EQUITY_LOG
        self._plans_path = DEFAULT_PLANS

        self._symbol_var = tk.StringVar(value=os.environ.get("DASHBOARD_SYMBOL") or "SPY")
        self._status_var = tk.StringVar(value="Load logs to begin.")
        self._regime_busy = threading.Lock()
        self._regime_running = False
        self._algo_detail_by_iid: dict[str, str] = {}

        self._build_ui()
        self._schedule_log_refresh()
        if self._auto_regime:
            rperiod = max(30_000, int(self._regime_interval_ms.get()))
            self.root.after(_delay_ms_to_next_system_tick(rperiod), self._tick_regime_then_reschedule)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Regime symbol:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(top, textvariable=self._symbol_var, width=8).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(top, text="Refresh regimes now", command=self._start_regime_fetch).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(top, textvariable=self._status_var).pack(side=tk.LEFT, padx=(8, 0))

        regime_row = ttk.Frame(self.root, padding=(6, 0, 6, 6))
        regime_row.pack(fill=tk.X)
        self._regime_labels: dict[str, tuple[tk.Label, ttk.Label]] = {}
        for col, (label, _, _, _, _) in enumerate(_REGIME_TIMEFRAMES):
            fr = ttk.LabelFrame(regime_row, text=f" {label} ", padding=8)
            fr.grid(row=0, column=col, padx=4, sticky=tk.NSEW)
            regime_row.columnconfigure(col, weight=1)
            dir_l = tk.Label(fr, text="—", font=("Segoe UI", 18, "bold"), fg=_DIRECTION_COLORS["unknown"])
            dir_l.pack()
            sub = ttk.Label(fr, text="(connect TWS / Gateway for IBKR)", font=("Segoe UI", 9), wraplength=320)
            sub.pack()
            self._regime_labels[label] = (dir_l, sub)

        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        self._tab_options = ttk.Frame(nb)
        self._tab_equity = ttk.Frame(nb)
        self._tab_algo = ttk.Frame(nb)
        self._tab_plans = ttk.Frame(nb)
        nb.add(self._tab_options, text="Options dry-run")
        nb.add(self._tab_equity, text="Equity")
        nb.add(self._tab_algo, text="Algo (ROEE)")
        nb.add(self._tab_plans, text="Plans snapshot")

        opts = (
            "symbol",
            "plan_id",
            "as_of_utc",
            "strategy",
            "closed",
            "entry",
            "mark",
            "uPnL",
            "uPnL%",
            "signal",
            "dte/qty",
        )
        self._tree_options = self._make_tree(self._tab_options, opts)

        eqcols = (
            "symbol",
            "plan_id",
            "as_of_utc",
            "dir",
            "closed",
            "qty",
            "entry",
            "mark",
            "uPnL",
            "uPnL%",
            "signal",
            "note",
        )
        self._tree_equity = self._make_tree(self._tab_equity, eqcols)

        algo_outer = ttk.Frame(self._tab_algo)
        algo_outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        algo_cols = (
            "symbol",
            "status",
            "action",
            "strategy",
            "regime_key",
            "skip_reason",
            "rationale",
        )
        algo_split = ttk.Panedwindow(algo_outer, orient=tk.VERTICAL)
        algo_split.pack(fill=tk.BOTH, expand=True)
        tree_fr = ttk.Frame(algo_split)
        detail_fr = ttk.LabelFrame(algo_split, text="Selected — full entry / pipeline detail", padding=4)
        algo_split.add(tree_fr, weight=3)
        algo_split.add(detail_fr, weight=2)
        ys_a = ttk.Scrollbar(tree_fr, orient=tk.VERTICAL)
        xs_a = ttk.Scrollbar(tree_fr, orient=tk.HORIZONTAL)
        self._tree_algo = ttk.Treeview(
            tree_fr,
            columns=algo_cols,
            show="headings",
            yscrollcommand=ys_a.set,
            xscrollcommand=xs_a.set,
        )
        ys_a.config(command=self._tree_algo.yview)
        xs_a.config(command=self._tree_algo.xview)
        for c in algo_cols:
            self._tree_algo.heading(c, text=c)
            self._tree_algo.column(c, width=100, minwidth=40)
        self._tree_algo.column("symbol", width=64)
        self._tree_algo.column("strategy", width=130)
        self._tree_algo.column("regime_key", width=200)
        self._tree_algo.column("skip_reason", width=160)
        self._tree_algo.column("rationale", width=260)
        self._tree_algo.grid(row=0, column=0, sticky=tk.NSEW)
        ys_a.grid(row=0, column=1, sticky=tk.NS)
        xs_a.grid(row=1, column=0, sticky=tk.EW)
        tree_fr.rowconfigure(0, weight=1)
        tree_fr.columnconfigure(0, weight=1)
        self._algo_detail_text = tk.Text(detail_fr, height=12, wrap=tk.WORD, font=("Consolas", 10))
        ys_d = ttk.Scrollbar(detail_fr, command=self._algo_detail_text.yview)
        self._algo_detail_text.configure(yscrollcommand=ys_d.set)
        self._algo_detail_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ys_d.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree_algo.bind("<<TreeviewSelect>>", self._on_algo_select)

        plans_fr = ttk.Frame(self._tab_plans, padding=6)
        plans_fr.pack(fill=tk.BOTH, expand=True)
        self._plans_text = tk.Text(plans_fr, height=18, wrap=tk.WORD, font=("Consolas", 10))
        ys = ttk.Scrollbar(plans_fr, command=self._plans_text.yview)
        self._plans_text.configure(yscrollcommand=ys.set)
        self._plans_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ys.pack(side=tk.RIGHT, fill=tk.Y)

        bot = ttk.Frame(self.root, padding=6)
        bot.pack(fill=tk.X)
        self._log_interval_ms = tk.IntVar(value=5000)
        self._regime_interval_ms = tk.IntVar(value=120_000)
        ttk.Label(bot, text="Log period (ms, wall-clock aligned):").pack(side=tk.LEFT)
        ttk.Spinbox(bot, from_=2000, to=120_000, increment=1000, width=8, textvariable=self._log_interval_ms).pack(
            side=tk.LEFT, padx=(4, 16)
        )
        ttk.Label(bot, text="Regime period (ms, wall-clock aligned):").pack(side=tk.LEFT)
        ttk.Spinbox(
            bot,
            from_=30_000,
            to=600_000,
            increment=10_000,
            width=8,
            textvariable=self._regime_interval_ms,
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(bot, text=f"IBKR client id={_dashboard_client_id()}").pack(side=tk.RIGHT)

    def _make_tree(self, parent: ttk.Frame, columns: tuple[str, ...]) -> ttk.Treeview:
        fr = ttk.Frame(parent)
        fr.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        ys = ttk.Scrollbar(fr, orient=tk.VERTICAL)
        xs = ttk.Scrollbar(fr, orient=tk.HORIZONTAL)
        tree = ttk.Treeview(fr, columns=columns, show="headings", yscrollcommand=ys.set, xscrollcommand=xs.set)
        ys.config(command=tree.yview)
        xs.config(command=tree.xview)
        for c in columns:
            tree.heading(c, text=c)
            tree.column(c, width=88, minwidth=48)
        if "symbol" in columns:
            tree.column("symbol", width=72)
        if "strategy" in columns:
            tree.column("strategy", width=140)
        if "signal" in columns:
            tree.column("signal", width=160)
        if "note" in columns:
            tree.column("note", width=200)
        if "dte/qty" in columns:
            tree.column("dte/qty", width=64)
        if "plan_id" in columns:
            tree.column("plan_id", width=148)
        if "as_of_utc" in columns:
            tree.column("as_of_utc", width=132)
        tree.grid(row=0, column=0, sticky=tk.NSEW)
        ys.grid(row=0, column=1, sticky=tk.NS)
        xs.grid(row=1, column=0, sticky=tk.EW)
        fr.rowconfigure(0, weight=1)
        fr.columnconfigure(0, weight=1)
        return tree

    def _clear_tree(self, tree: ttk.Treeview) -> None:
        tree.delete(*tree.get_children())

    def _fmt_num(self, x: str) -> str:
        try:
            return f"{float(x):,.2f}"
        except (TypeError, ValueError):
            return str(x)

    def _fmt_as_of_utc(self, raw: str) -> str:
        ts = _parse_ts_utc(raw)
        if ts is None:
            return (raw or "")[:22]
        return ts.strftime("%Y-%m-%d %H:%M UTC")

    def _refresh_logs(self) -> None:
        self._reload_options_table()
        self._reload_equity_table()
        self._reload_algo_table()
        self._reload_plans_text()

    def _reload_plans_text(self) -> None:
        self._plans_text.delete("1.0", tk.END)
        self._plans_text.insert(tk.END, _load_plans_summary(self._plans_path))

    def _reload_algo_table(self) -> None:
        self._algo_detail_by_iid.clear()
        self._clear_tree(self._tree_algo)
        self._algo_detail_text.delete("1.0", tk.END)
        rows = _load_algo_decision_rows(self._plans_path)
        for i, r in enumerate(rows):
            iid = f"algo_{i}"
            self._tree_algo.insert(
                "",
                tk.END,
                iid=iid,
                values=(
                    r["symbol"],
                    r["status"],
                    r["action"],
                    r["strategy"],
                    r["regime_key"],
                    r["skip_reason"],
                    r["rationale_short"],
                ),
            )
            self._algo_detail_by_iid[iid] = str(r["detail"])
        kids = self._tree_algo.get_children()
        if kids:
            self._tree_algo.selection_set(kids[0])
            self._tree_algo.focus(kids[0])
            self._on_algo_select()

    def _on_algo_select(self, _event: object | None = None) -> None:
        sel = self._tree_algo.selection()
        if not sel:
            return
        block = self._algo_detail_by_iid.get(sel[0], "")
        self._algo_detail_text.delete("1.0", tk.END)
        self._algo_detail_text.insert(tk.END, block)

    def _reload_options_table(self) -> None:
        self._clear_tree(self._tree_options)
        rows = _load_csv_rows(self._options_log)
        if not rows or _is_equity_schema(rows):
            return
        for r in _latest_open_row_per_symbol(rows):
            self._tree_options.insert(
                "",
                tk.END,
                values=(
                    r.get("symbol", ""),
                    r.get("plan_id", ""),
                    self._fmt_as_of_utc(r.get("timestamp_utc", "")),
                    r.get("strategy", ""),
                    r.get("closed", ""),
                    self._fmt_num(r.get("entry_debit", "")),
                    self._fmt_num(r.get("current_mark", "")),
                    self._fmt_num(r.get("unrealized_pnl", "")),
                    self._fmt_num(r.get("unrealized_pnl_pct", "")),
                    r.get("signal", ""),
                    r.get("dte", ""),
                ),
            )

    def _reload_equity_table(self) -> None:
        self._clear_tree(self._tree_equity)
        rows = _load_csv_rows(self._equity_log)
        if not rows:
            return
        if not _is_equity_schema(rows):
            return
        for r in _latest_open_row_per_symbol(rows):
            self._tree_equity.insert(
                "",
                tk.END,
                values=(
                    r.get("symbol", ""),
                    r.get("plan_id", ""),
                    self._fmt_as_of_utc(r.get("timestamp_utc", "")),
                    r.get("strategy", ""),
                    r.get("closed", ""),
                    r.get("quantity", ""),
                    self._fmt_num(r.get("entry_debit", "")),
                    self._fmt_num(r.get("current_mark", "")),
                    self._fmt_num(r.get("unrealized_pnl", "")),
                    self._fmt_num(r.get("unrealized_pnl_pct", "")),
                    r.get("signal", ""),
                    (r.get("note", "") or "")[:80],
                ),
            )

    def _set_regime_ui(self, payload: dict[str, dict[str, str]] | None, err: str | None) -> None:
        if err:
            self._status_var.set(err[:200])
            for label in self._regime_labels:
                d, s = self._regime_labels[label]
                d.config(text="err", fg=_DIRECTION_COLORS["unknown"])
                s.config(text=err[:120])
            return
        assert payload is not None
        self._status_var.set(f"Regimes OK — {self._symbol_var.get().strip().upper()}")
        for label, data in payload.items():
            d_l, s_l = self._regime_labels[label]
            dr = data.get("direction_regime", "unknown")
            color = _DIRECTION_COLORS.get(dr, _DIRECTION_COLORS["unknown"])
            d_l.config(text=dr.upper(), fg=color)
            rk = data.get("regime_key", "")
            bt = data.get("bar_time", "")
            det = data.get("detail", "")
            s_l.config(text=f"{det}\n{rk}\n{bt}")

    def _start_regime_fetch(self) -> None:
        if not self._regime_busy.acquire(blocking=False):
            return
        if self._regime_running:
            self._regime_busy.release()
            return
        self._regime_running = True
        sym = self._symbol_var.get().strip().upper()
        self._status_var.set(f"Fetching regimes for {sym}…")

        def work() -> None:
            err: str | None = None
            payload: dict[str, dict[str, str]] | None = None
            try:
                payload = compute_regime_timeframes(sym)
            except ImportError as e:
                err = f"ibapi missing: {e}"
            except Exception as e:
                err = str(e)

            def done() -> None:
                self._set_regime_ui(payload, err)
                self._regime_running = False
                self._regime_busy.release()

            self.root.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def _schedule_log_refresh(self) -> None:
        try:
            self._refresh_logs()
        except OSError:
            pass
        period = max(2000, int(self._log_interval_ms.get()))
        self.root.after(_delay_ms_to_next_system_tick(period), self._schedule_log_refresh)

    def _tick_regime_then_reschedule(self) -> None:
        self._start_regime_fetch()
        period = max(30_000, int(self._regime_interval_ms.get()))
        self.root.after(_delay_ms_to_next_system_tick(period), self._tick_regime_then_reschedule)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default=None, help="Default ticker for regime panels")
    ap.add_argument(
        "--no-auto-regime",
        action="store_true",
        help="Only refresh regimes when you click the button (still loads logs on a timer)",
    )
    ap.add_argument("--options-log", type=Path, default=None)
    ap.add_argument("--equity-log", type=Path, default=None)
    ap.add_argument("--plans", type=Path, default=None)
    args = ap.parse_args()

    app = LiveTradingDashboard(auto_regime=not args.no_auto_regime)
    if args.symbol:
        app._symbol_var.set(args.symbol.strip().upper())
    if args.options_log:
        app._options_log = args.options_log if args.options_log.is_absolute() else ROOT / args.options_log
    if args.equity_log:
        app._equity_log = args.equity_log if args.equity_log.is_absolute() else ROOT / args.equity_log
    if args.plans:
        app._plans_path = args.plans if args.plans.is_absolute() else ROOT / args.plans

    app.run()


if __name__ == "__main__":
    main()
