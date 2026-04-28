"""
Spock — logical market analyst.

"Fascinating. The probability of a profitable outcome, Captain, is precisely
47.3%… assuming the regime signal is correct."

Reads current trade plans, regime signals, and Kronos forecast artefacts,
then synthesises a concise, probability-weighted analysis of each active plan.
Outputs a structured recommendation list for Kirk to act on.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from rlm.agents.base import LLMClient, Message
from rlm.utils.market_hours import session_label

# -----------------------------------------------------------------------

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

# -----------------------------------------------------------------------
_SPOCK_SYSTEM = """\
You are Spock, the Science Officer and strategic analyst of this trading system.
You operate on pure logic and probability. No emotional language. No hedging phrases.
Structure your response as a numbered list of active trade plans:
  1. SYMBOL | STRATEGY | REGIME | ACTION: [GO / HOLD / ABORT] | RATIONALE: <one sentence, cite numbers>
End with a single LINE: OVERALL RISK POSTURE: [LOW / MODERATE / HIGH / CRITICAL]
Keep each entry under 25 words. Do not repeat data already provided verbatim.
"""


@dataclass
class PlanAnalysis:
    symbol: str
    plan_id: str
    strategy: str
    regime: str
    rank_score: float
    action: str          # GO | HOLD | ABORT
    rationale: str
    raw_plan: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpockBriefing:
    timestamp: str
    plan_analyses: list[PlanAnalysis] = field(default_factory=list)
    overall_risk: str = "UNKNOWN"
    llm_text: str = ""
    context_snapshot: str = ""
    data_generated_at: str = ""
    data_age_minutes: float = 0.0


class SpockAgent:
    def __init__(self, root: Path, llm: Optional[LLMClient] = None) -> None:
        self.root = root
        self.llm = llm or LLMClient()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self) -> SpockBriefing:
        """Read current artefacts and return Spock's briefing."""
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        briefing = SpockBriefing(timestamp=ts)

        gen_at, age_min = self._plans_data_age()
        briefing.data_generated_at = gen_at
        briefing.data_age_minutes = age_min

        context = self._build_context()
        briefing.context_snapshot = context

        if not context.strip() or "no active plans" in context.lower():
            briefing.llm_text = "No active trade plans to analyse at this time. Standing by."
            briefing.overall_risk = "LOW"
            return briefing

        try:
            briefing.llm_text = self.llm.chat(
                [Message("user",
                    f"Analyse these active trade plans and regime signals:\n\n{context}\n\n"
                    "Provide your logical assessment per the format in your instructions.")],
                system=_SPOCK_SYSTEM,
            )
            briefing.overall_risk = self._extract_risk(briefing.llm_text)
        except Exception as exc:
            briefing.llm_text = f"[Spock LLM unavailable: {exc}]\n\n{context}"

        return briefing

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def _build_context(self) -> str:
        sections: list[str] = [f"Market State: {session_label()}"]

        # 1 — Universe trade plans
        plans_text = self._read_plans()
        if plans_text:
            sections.append(f"=== Active Trade Plans ===\n{plans_text}")

        # 2 — Latest run artefact (regime + Kronos signals)
        artifact_text = self._read_latest_artifact()
        if artifact_text:
            sections.append(f"=== Latest Pipeline Signals ===\n{artifact_text}")

        # 3 — Equity positions
        eq_text = self._read_equity_state()
        if eq_text:
            sections.append(f"=== Open Equity Positions ===\n{eq_text}")

        # 4 — Historical OOS walk-forward performance (recent windows)
        wf_text = self._read_walkforward_performance()
        if wf_text:
            sections.append(f"=== Historical OOS Performance (Walk-Forward) ===\n{wf_text}")

        # 5 — Walk-forward OOS summaries (cross-symbol)
        wf_agg = self._read_walkforward_oos_aggregate()
        if wf_agg:
            sections.append(f"=== Walk-forward OOS (universe aggregate) ===\n{wf_agg}")

        return "\n\n".join(sections) if sections else "No active plans or signals found."

    _STALE_THRESHOLD_MINUTES = 30

    def _plans_data_age(self) -> tuple[str, float]:
        """Return (generated_at_utc_str, age_in_minutes) for the plans file."""
        path = self.root / "data" / "processed" / "universe_trade_plans.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            gen_str = data.get("generated_at_utc", "")
            gen_dt = datetime.fromisoformat(gen_str.replace("Z", "+00:00"))
            age = (datetime.now(tz=timezone.utc) - gen_dt).total_seconds() / 60
            return gen_str, round(age, 1)
        except Exception:
            return "", 0.0

    def _read_plans(self) -> str:
        path = self.root / "data" / "processed" / "universe_trade_plans.json"
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
            if age_min > self._STALE_THRESHOLD_MINUTES:
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

    def _read_latest_artifact(self) -> str:
        artifacts_dir = self.root / "data" / "artifacts" / "runs"
        if not artifacts_dir.is_dir():
            return ""
        try:
            files = sorted(artifacts_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
            if not files:
                return ""
            latest = json.loads(files[0].read_text(encoding="utf-8"))
            lines: list[str] = [f"Run: {files[0].name}"]
            for key in ("regime", "regime_label", "regime_confidence",
                        "kronos_return_forecast", "kronos_confidence",
                        "kronos_regime_agreement", "kronos_transition_flag",
                        "roee_strategy", "roee_confidence"):
                val = latest.get(key)
                if val is not None:
                    lines.append(f"  {key}: {val}")
            return "\n".join(lines) if len(lines) > 1 else ""
        except Exception:
            return ""

    def _read_walkforward_performance(self, windows: int = 5) -> str:
        """Summarise the most recent OOS windows from the walk-forward summary CSV."""
        candidates = [
            self.root / "data" / "processed" / "walkforward_summary.csv",
        ]
        # Prefer symbol-specific files if they have trading data
        for p in (self.root / "data" / "processed").glob("walkforward_summary_*.csv"):
            candidates.insert(0, p)

        import csv as _csv

        for path in candidates:
            if not path.is_file():
                continue
            try:
                with path.open(encoding="utf-8") as f:
                    rows = list(_csv.DictReader(f))
                if not rows:
                    continue
                # Check this file actually has trading metric columns
                if "win_rate" not in rows[0]:
                    continue
                recent = rows[-windows:]
                lines = [f"Last {len(recent)} OOS windows ({path.stem}):"]
                for r in recent:
                    try:
                        wr = f"{float(r['win_rate']):.0%}"
                        sh = f"{float(r['sharpe']):.1f}"
                        pnl = f"{float(r['avg_trade_pnl_pct']):+.1f}%"
                        n = int(float(r.get("num_trades", 0)))
                        oos_end = str(r.get("oos_end", "?"))[:10]
                        safe = r.get("regime_safety_passed", "?")
                        safe_frac = r.get("regime_safety_fraction", "")
                        safe_str = (
                            f"{safe} ({float(safe_frac):.0%})" if safe_frac else str(safe)
                        )
                        lines.append(
                            f"  OOS ending {oos_end}: win={wr} sharpe={sh} "
                            f"avg_pnl={pnl} trades={n} regime_safe={safe_str}"
                        )
                    except (ValueError, KeyError):
                        continue
                if len(lines) > 1:
                    return "\n".join(lines)
            except Exception:
                continue
        return ""

    def _read_walkforward_oos_aggregate(self) -> str:
        proc = self.root / "data" / "processed"
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
            for col in ("sharpe", "total_return_pct", "max_drawdown", "profit_factor", "num_trades"):
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
            lines.append(
                f"  worst: {worst[0]} mean_sharpe={worst[1].get('mean_sharpe', 'n/a')}"
            )
        return "\n".join(lines)

    def _read_equity_state(self) -> str:
        path = self.root / "data" / "processed" / "equity_positions_state.json"
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

    _RISK_PATTERN = re.compile(
        r"OVERALL\s+RISK\s+POSTURE\s*:\s*(CRITICAL|HIGH|MODERATE|LOW)",
        re.IGNORECASE,
    )

    @classmethod
    def _extract_risk(cls, text: str) -> str:
        for line in reversed(text.splitlines()):
            m = cls._RISK_PATTERN.search(line)
            if m:
                return m.group(1).upper()
        return "UNKNOWN"
