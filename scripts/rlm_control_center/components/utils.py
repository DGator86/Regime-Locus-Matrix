"""Paths, loaders, subprocess helpers, and session-friendly utilities."""

from __future__ import annotations

import json
import os
import subprocess
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


def repo_root_from_here(here: Path) -> Path:
    """``here`` = ``scripts/rlm_control_center`` (directory containing ``app.py``)."""
    return here.resolve().parents[1]


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def env_connected_massive() -> bool:
    return bool(os.environ.get("MASSIVE_API_KEY", "").strip())


def env_connected_ibkr() -> bool:
    return bool(os.environ.get("IBKR_HOST", "").strip() or os.environ.get("TWS_HOST", "").strip())


def load_dotenv_if_present(root: Path) -> None:
    try:
        from dotenv import load_dotenv

        p = root / ".env"
        if p.is_file():
            load_dotenv(p)
    except ImportError:
        pass


@st.cache_data(show_spinner=False, ttl=20)
def safe_load_json(path_str: str) -> dict[str, Any] | None:
    p = Path(path_str)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False, ttl=20)
def safe_read_csv(path_str: str) -> pd.DataFrame | None:
    p = Path(path_str)
    if not p.is_file():
        return None
    try:
        return pd.read_csv(p)
    except (OSError, ValueError):
        return None


def flatten_universe_result(row: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested ``pipeline`` / ``decision`` for AgGrid."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if k == "pipeline" and isinstance(v, dict):
            for pk, pv in v.items():
                key = f"p_{pk}"
                out[key] = _short_val(pv, key=key)
        elif k == "decision" and isinstance(v, dict):
            for dk, dv in v.items():
                key = f"d_{dk}"
                out[key] = _short_val(dv, key=key)
        elif isinstance(v, (list, dict)):
            out[k] = _short_val(v, key=k)
        else:
            out[k] = v
    return out


def _short_val(v: Any, *, key: str, max_len: int = 400) -> Any:
    if isinstance(v, (list, dict)):
        s = json.dumps(v, default=str)
        return s if len(s) <= max_len else s[: max_len - 3] + "..."
    return v


def enrich_universe_with_feature_tail(
    udf: pd.DataFrame,
    root: Path,
    *,
    max_extra_cols: int = 24,
) -> pd.DataFrame:
    """
    Left-join each symbol's latest numeric row from ``data/processed/features_{SYM}.csv``.
    New columns are prefixed with ``feat_`` to avoid collisions.
    """
    if udf.empty or "symbol" not in udf.columns:
        return udf
    out = udf.copy()
    for sym in out["symbol"].dropna().astype(str).unique():
        symu = sym.upper()
        p = root / "data" / "processed" / f"features_{symu}.csv"
        if not p.is_file():
            continue
        try:
            fdf = pd.read_csv(p)
        except (OSError, ValueError):
            continue
        if fdf.empty:
            continue
        tail = fdf.iloc[-1]
        num_cols = [c for c in fdf.columns if pd.api.types.is_numeric_dtype(fdf[c])][:max_extra_cols]
        for c in num_cols:
            col_name = f"feat_{c}"
            if col_name not in out.columns:
                out[col_name] = pd.NA
            mask = out["symbol"].astype(str).str.upper() == symu
            try:
                out.loc[mask, col_name] = float(tail.get(c, float("nan")))
            except (TypeError, ValueError):
                out.loc[mask, col_name] = tail.get(c)
    return out


def universe_health_score(payload: dict[str, Any] | None) -> tuple[float, str]:
    """Simple composite 0–100 from active fraction + freshness hint."""
    if not payload:
        return 0.0, "No universe_trade_plans.json loaded"
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return 15.0, "Empty results"
    active = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "active")
    frac = active / max(len(results), 1)
    score = min(100.0, 40.0 + 60.0 * frac)
    gen = str(payload.get("generated_at_utc", ""))
    tail = f" — generated {gen[:19]}…" if gen else ""
    return score, f"{active}/{len(results)} active{tail}"


def run_repo_script(
    root: Path,
    python_exe: str,
    script_rel: str,
    extra_args: list[str],
    *,
    timeout_s: float | None = 3600.0,
) -> tuple[int, str, str]:
    """Run ``python script`` from repo root; return (code, stdout, stderr)."""
    cmd = [python_exe, str(root / script_rel), *extra_args]
    try:
        p = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return int(p.returncode), p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except OSError as e:
        return 1, "", str(e)


def clear_file_caches() -> None:
    """Invalidate JSON/CSV cache (e.g. after pipeline writes new artifacts)."""
    safe_load_json.clear()
    safe_read_csv.clear()


def append_pipeline_log(text: str, *, maxlen: int = 800) -> None:
    dq: deque[str] = st.session_state.setdefault("pipeline_log", deque(maxlen=maxlen))
    for line in text.splitlines():
        dq.append(line)
