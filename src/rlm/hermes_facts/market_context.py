"""Pre-formatted trade + regime context for Hermes tools, no LLM."""

from __future__ import annotations

import csv as _csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from rlm.utils.market_hours import session_label

_STALE_THRESHOLD_MINUTES = 30


def _fmt_score(v: object) -> str:
    try:
        return f"{float(v):.4f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "?"


def _fmt_pct(v: object, decimals: int = 1) -> str:
    try:
        return f"{float(v) * 100:.{decimals}f}%"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "?"


def _fmt_ret(v: object) -> str:
    try:
        val = float(v) * 100  # type: ignore[arg-type]
        sign = "+" if val >= 0 else ""
        return f"{sign}{val:.2f}%"
    except (TypeError, ValueError):
        return "?"


def plans_data_age(root: Path) -> tuple[str, float]:
    path = root / "data" / "processed" / "universe_trade_plans.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        gen_str = data.get("generated_at_utc", "")
        gen_dt = datetime.fromisoformat(gen_str.replace("Z", "+00:00"))
        age = (datetime.now(tz=timezone.utc) - gen_dt).total_seconds() / 60
        return gen_str, round(age, 1)
    except Exception:
        return "", 0.0


def _read_plans(root: Path) -> str:
    path = root / "data" / "processed" / "universe_trade_plans.json"
    if not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    results = data.get("results") or []
    active = [r for r in results if r.get("status") == "active"]
    if not active:
        return "No active plans."
    generated_at = data.get("generated_at_utc", "?")
    stale_tag = ""
    try:
        gen_dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        age_min = (datetime.now(tz=timezone.utc) - gen_dt).total_seconds() / 60
        if age_min > _STALE_THRESHOLD_MINUTES:
            stale_tag = f" [STALE: {age_min:.0f}min old — scores may not reflect current market]"
    except Exception:
        pass
    lines: list[str] = [
        f"Generated: {generated_at}{stale_tag}",
        f"Active plans: {len(active)}",
    ]
    active.sort(key=lambda x: float(x.get("rank_score") or 0), reverse=True)
    for r in active[:15]:
        sym = r.get("symbol", "?")
        strat = r.get("strategy", "?")
        score = r.get("rank_score")
        pid = r.get("plan_id", "?")
        regime = r.get("regime", r.get("regime_label", "?"))
        conf = r.get("regime_confidence", r.get("confidence"))
        kron = r.get("kronos_return_forecast")
        lines.append(
            f"  • {sym} [{strat}] score={_fmt_score(score)} regime={regime} "
            f"conf={_fmt_pct(conf)} kronos={_fmt_ret(kron)} id={pid}"
        )
    return "\n".join(lines)


def _read_latest_artifact(root: Path) -> str:
    artifacts_dir = root / "data" / "artifacts" / "runs"
    if not artifacts_dir.is_dir():
        return ""
    try:
        files = sorted(artifacts_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not files:
            return ""
        latest = json.loads(files[0].read_text(encoding="utf-8"))
        lines: list[str] = [f"Run: {files[0].name}"]
        for key in (
            "regime",
            "regime_label",
            "regime_confidence",
            "kronos_return_forecast",
            "kronos_confidence",
            "kronos_regime_agreement",
            "kronos_transition_flag",
            "roee_strategy",
            "roee_confidence",
        ):
            val = latest.get(key)
            if val is not None:
                lines.append(f"  {key}: {val}")
        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception:
        return ""


def _read_walkforward_performance(root: Path, windows: int = 3) -> str:
    processed_dir = root / "data" / "processed"
    if not processed_dir.is_dir():
        return ""

    all_lines: list[str] = []

    def _collect(path: Path, label: str) -> None:
        try:
            with path.open(encoding="utf-8") as f:
                rows = list(_csv.DictReader(f))
            if not rows or "win_rate" not in rows[0]:
                return
            recent = rows[-windows:]
            all_lines.append(f"{label} (last {len(recent)} OOS windows):")
            for r in recent:
                try:
                    wr = f"{float(r['win_rate']):.0%}"
                    sh = f"{float(r['sharpe']):.1f}"
                    pnl = f"{float(r['avg_trade_pnl_pct']):+.1f}%"
                    n = int(float(r.get("num_trades", 0)))
                    oos_end = str(r.get("oos_end", "?"))[:10]
                    safe_frac = r.get("regime_safety_fraction", "")
                    safe = r.get("regime_safety_passed", "?")
                    safe_str = f"{safe} ({float(safe_frac):.0%})" if safe_frac else str(safe)
                    all_lines.append(
                        f"  OOS {oos_end}: win={wr} sharpe={sh} "
                        f"avg_pnl={pnl} trades={n} safe={safe_str}"
                    )
                except (ValueError, KeyError):
                    continue
        except Exception:
            pass

    for path in sorted(processed_dir.glob("walkforward_summary_*.csv")):
        sym = path.stem.replace("walkforward_summary_", "")
        _collect(path, sym)

    generic = processed_dir / "walkforward_summary.csv"
    if generic.is_file():
        _collect(generic, "universe")

    return "\n".join(all_lines) if all_lines else ""


def _read_walkforward_oos_aggregate(root: Path) -> str:
    proc = root / "data" / "processed"
    if not proc.is_dir():
        return ""
    snap_path = proc / "walkforward_universe_latest.json"
    snap_line = ""
    if snap_path.is_file():
        try:
            snap: dict = json.loads(snap_path.read_text(encoding="utf-8"))
            cs = snap.get("cross_symbol_mean_window_sharpe")
            if cs is not None:
                snap_line = (
                    f"Latest universe walk-forward batch ({str(snap.get('ts_utc', ''))[:19]}): "
                    f"mean OOS window sharpe={float(cs):.4f}, "
                    f"ok={snap.get('symbols_ok')}/{snap.get('symbols_attempted')}, "
                    f"OOS windows={snap.get('total_oos_windows')}"
                )
        except Exception:
            snap_line = ""

    pat = re.compile(r"^walkforward_summary_([A-Za-z0-9\.\-]+)\.csv$")
    per_symbol: list[tuple[str, dict[str, float]]] = []
    for p in sorted(proc.glob("walkforward_summary_*.csv")):
        m = pat.match(p.name)
        if not m or p.name == "walkforward_summary_universe_all_windows.csv":
            continue
        sym = m.group(1).upper()
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        row: dict[str, float] = {}
        for col in (
            "sharpe",
            "total_return_pct",
            "max_drawdown",
            "profit_factor",
            "num_trades",
        ):
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                row[f"mean_{col}"] = float(s.mean())
        if row:
            per_symbol.append((sym, row))
    if not per_symbol:
        return snap_line

    lines = [f"Symbols with walk-forward summary files: {len(per_symbol)}"]
    if snap_line:
        lines.append(snap_line)
    for src in ("sharpe", "total_return_pct", "max_drawdown"):
        k = f"mean_{src}"
        vals = [d[k] for _s, d in per_symbol if k in d]
        if vals:
            tag = f"mean window {src.replace('_', ' ')}"
            lines.append(f"  cross-symbol avg ({tag}): {float(np.mean(vals)):.4f}")
    if per_symbol:
        by_sharpe = sorted(
            per_symbol,
            key=lambda t: t[1].get("mean_sharpe", float("-inf")),
            reverse=True,
        )
        best = by_sharpe[0]
        worst = by_sharpe[-1]
        lines.append(
            f"  best OOS (by mean window sharpe): {best[0]} mean_sharpe={best[1].get('mean_sharpe', 'n/a')}"
        )
        lines.append(f"  worst: {worst[0]} mean_sharpe={worst[1].get('mean_sharpe', 'n/a')}")
    return "\n".join(lines)


def _read_equity_state(root: Path) -> str:
    path = root / "data" / "processed" / "equity_positions_state.json"
    if not path.is_file():
        return ""
    try:
        data: dict = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    open_pos = [(k, v) for k, v in data.items() if str((v or {}).get("status")) == "open"]
    if not open_pos:
        return "No open equity positions."
    lines = []
    for pid, pos in open_pos[:10]:
        sym = (pos or {}).get("symbol", "?")
        side = (pos or {}).get("side", "?")
        qty = (pos or {}).get("quantity", "?")
        lines.append(f"  • {sym} {side} qty={qty} id={pid}")
    return "\n".join(lines)


def build_trade_and_regime_context(root: Path) -> str:
    """Single string suitable as user context for the research / commander agent."""
    root = root.resolve()
    sections: list[str] = [f"Market State: {session_label()}"]

    plans_text = _read_plans(root)
    if plans_text:
        sections.append(f"=== Active Trade Plans ===\n{plans_text}")

    artifact_text = _read_latest_artifact(root)
    if artifact_text:
        sections.append(f"=== Latest Pipeline Signals ===\n{artifact_text}")

    eq_text = _read_equity_state(root)
    if eq_text:
        sections.append(f"=== Open Equity Positions ===\n{eq_text}")

    wf_text = _read_walkforward_performance(root)
    if wf_text:
        sections.append(f"=== Historical OOS Performance (Walk-Forward) ===\n{wf_text}")

    wf_agg = _read_walkforward_oos_aggregate(root)
    if wf_agg:
        sections.append(f"=== Walk-forward OOS (universe aggregate) ===\n{wf_agg}")

    return "\n\n".join(sections) if sections else "No active plans or signals found."
