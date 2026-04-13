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
    """
    Resolve a path that points to the scripts/rlm_control_center directory and return the repository root two levels above it.
    
    Parameters:
        here (Path): Path to the directory that contains the control-center scripts (expected to be the scripts/rlm_control_center directory).
    
    Returns:
        Path: The repository root directory (two parent levels above `here`).
    """
    return here.resolve().parents[1]


def utc_now_str() -> str:
    """
    Return the current UTC timestamp formatted as "YYYY-MM-DD HH:MM:SS UTC".
    
    Returns:
        str: Current UTC timestamp formatted as "%Y-%m-%d %H:%M:%S UTC".
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def env_connected_massive() -> bool:
    """
    Check whether the MASSIVE API key is present in the environment.
    
    Returns:
        `true` if the environment variable `MASSIVE_API_KEY` exists and is not empty after stripping, `false` otherwise.
    """
    return bool(os.environ.get("MASSIVE_API_KEY", "").strip())


def env_connected_ibkr() -> bool:
    """
    Check whether an Interactive Brokers host is configured via environment variables.
    
    Returns:
        `true` if either the `IBKR_HOST` or `TWS_HOST` environment variable is set to a non-empty value after trimming whitespace, `false` otherwise.
    """
    return bool(os.environ.get("IBKR_HOST", "").strip() or os.environ.get("TWS_HOST", "").strip())


def load_dotenv_if_present(root: Path) -> None:
    """
    Load environment variables from a ".env" file in the given root directory if available.
    
    If the "python-dotenv" package is installed and a file named ".env" exists at `root`, its variables are loaded into the process environment. If the package is not installed or the file is absent, the function does nothing.
    
    Parameters:
        root (Path): Directory to check for a ".env" file.
    """
    try:
        from dotenv import load_dotenv

        p = root / ".env"
        if p.is_file():
            load_dotenv(p)
    except ImportError:
        pass


@st.cache_data(show_spinner=False, ttl=20)
def safe_load_json(path_str: str) -> dict[str, Any] | None:
    """
    Safely load and parse a JSON file from the given filesystem path.
    
    Parameters:
        path_str (str): File path to read as UTF-8 JSON.
    
    Returns:
        dict[str, Any] | None: Parsed JSON object when the file exists and is valid JSON; `None` if the file is missing, unreadable, or contains invalid JSON.
    """
    p = Path(path_str)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False, ttl=20)
def safe_read_csv(path_str: str) -> pd.DataFrame | None:
    """
    Attempt to read a CSV file and return it as a DataFrame.
    
    Parameters:
        path_str (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame | None: DataFrame parsed from the file, or `None` if the path is not a file or reading/parsing fails (e.g., I/O error or CSV parsing error).
    """
    p = Path(path_str)
    if not p.is_file():
        return None
    try:
        return pd.read_csv(p)
    except (OSError, ValueError):
        return None


def flatten_universe_result(row: dict[str, Any]) -> dict[str, Any]:
    """
    Produce a flattened dictionary suitable for AgGrid by expanding nested "pipeline" and "decision" mappings.
    
    Parameters:
        row (dict[str, Any]): A record that may contain nested mappings (notably the keys "pipeline" and "decision"), lists, or dicts.
    
    Returns:
        dict[str, Any]: A new dictionary where:
            - Entries from a "pipeline" mapping are emitted as keys prefixed with "p_" (e.g., "p_name").
            - Entries from a "decision" mapping are emitted as keys prefixed with "d_" (e.g., "d_reason").
            - Values that are lists or dicts are converted to short serialized representations (truncated if too long).
            - Other values are copied unchanged.
    """
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
    """
    Shortens list or dict values by JSON-serializing them and truncating long results.
    
    Parameters:
        v (Any): The value to process. Lists and dicts are serialized; other types are returned unchanged.
        key (str): Contextual key name (accepted for callers but not used by this function).
        max_len (int): Maximum allowed length of the returned string; if the serialized representation exceeds this, it is truncated and "..." is appended.
    
    Returns:
        Any: For non-list/dict inputs, returns the input unchanged. For list or dict inputs, returns a JSON string representation truncated to at most `max_len` characters (with "..." appended when truncated).
    """
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
    Enriches a universe DataFrame by adding the latest numeric feature values for each symbol from per-symbol CSV files.
    
    For each distinct non-null symbol in `udf`, attempts to read root / "data" / "processed" / f"features_{SYMBOL}.csv" (symbol uppercased), takes the last row, and copies up to `max_extra_cols` numeric columns into new columns in the returned DataFrame prefixed with `feat_`. Symbols without a readable file are left unchanged; newly added feature columns are initialized with missing values for rows that do not match a symbol.
    
    Parameters:
        udf (pd.DataFrame): Input universe dataframe that must contain a "symbol" column.
        root (Path): Repository root used to locate per-symbol feature files (expects files at root/data/processed/features_{SYM}.csv).
        max_extra_cols (int): Maximum number of numeric columns to import from each feature file (default 24).
    
    Returns:
        pd.DataFrame: A copy of `udf` with additional `feat_{col}` columns containing the latest per-symbol numeric feature values.
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
    """
    Compute a 0–100 health score and a short status message for a universe payload.
    
    Parameters:
        payload (dict[str, Any] | None): Payload parsed from `universe_trade_plans.json`, expected to contain a
            top-level "results" list of plan records and an optional "generated_at_utc" timestamp string.
            Pass `None` when no payload is loaded.
    
    Returns:
        tuple[float, str]: A (score, status) pair where `score` is between 0.0 and 100.0 reflecting the fraction
        of results with `"status" == "active"`, and `status` is a short summary like `"X/Y active"` with an
        optional `" — generated YYYY-MM-DD HH:MM:SS…"` suffix when a timestamp is present.
        Special cases:
          - When `payload` is falsy, returns (0.0, "No universe_trade_plans.json loaded").
          - When `results` is missing or empty, returns (15.0, "Empty results").
    """
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
    """
    Execute a Python script located under the repository root and capture its outputs.
    
    Parameters:
        root (Path): Repository root used as the current working directory and to resolve script_rel.
        python_exe (str): Path or name of the Python executable to run.
        script_rel (str): Path to the script relative to `root`.
        extra_args (list[str]): Additional command-line arguments passed to the script.
        timeout_s (float | None): Maximum seconds to wait for the process; `None` means no timeout.
    
    Returns:
        tuple[int, str, str]: (exit_code, stdout, stderr).
            - On successful run: `exit_code` is the process return code, `stdout` and `stderr` contain captured text (empty string if none).
            - On timeout: `(124, "", "timeout")`.
            - On OS error (e.g., executable not found): `(1, "", str(e))`.
    """
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
    """
    Append lines of text to a session-scoped pipeline log stored in Streamlit's session_state.
    
    Creates session_state["pipeline_log"] as a collections.deque with the provided maxlen if it does not exist, then splits `text` by newline characters and appends each line to the deque in order.
    
    Parameters:
        text (str): Multiline text to append; each line is added as a separate entry.
        maxlen (int): Maximum number of entries to keep in the session deque when creating it (ignored if the deque already exists).
    """
    dq: deque[str] = st.session_state.setdefault("pipeline_log", deque(maxlen=maxlen))
    for line in text.splitlines():
        dq.append(line)
