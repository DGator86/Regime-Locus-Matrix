"""Canonical three-track layout: large equities, large options, SPY day trade."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_TRACKS_YAML = Path(__file__).resolve().parents[3] / "configs" / "trading_tracks.yaml"

TRACK_LARGE_EQUITIES = "large_equities"
TRACK_LARGE_OPTIONS = "large_options"
TRACK_SPY_DAYTRADE = "spy_daytrade"


@dataclass(frozen=True)
class TrackSpec:
    track_id: str
    label: str
    execution: str
    plans_path: str | None = None
    state_path: str | None = None
    log_path: str | None = None
    symbol: str | None = None
    pipeline_dte_min: int | None = None
    pipeline_dte_max: int | None = None
    short_dte: bool = False
    scalp_dte_min: int | None = None
    scalp_dte_max: int | None = None
    rth_env: str | None = None
    telegram_flag: str | None = None
    systemd_unit: str | None = None

    def resolve_log(self, root: Path) -> Path | None:
        if not self.log_path:
            return None
        p = Path(self.log_path)
        return p if p.is_absolute() else root / p

    def resolve_state(self, root: Path) -> Path | None:
        if not self.state_path:
            return None
        p = Path(self.state_path)
        return p if p.is_absolute() else root / p


def resolve_root() -> Path:
    raw = (os.environ.get("RLM_ROOT") or "").strip()
    if raw:
        return Path(raw).resolve()
    return Path.cwd().resolve()


@lru_cache(maxsize=1)
def _yaml_tracks() -> dict[str, dict[str, Any]]:
    if not _TRACKS_YAML.is_file():
        return {}
    loaded = yaml.safe_load(_TRACKS_YAML.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return {}
    tracks = loaded.get("tracks")
    return tracks if isinstance(tracks, dict) else {}


def load_tracks() -> dict[str, TrackSpec]:
    raw = _yaml_tracks()
    out: dict[str, TrackSpec] = {}
    for tid, blob in raw.items():
        if not isinstance(blob, dict):
            continue
        out[str(tid)] = TrackSpec(
            track_id=str(tid),
            label=str(blob.get("label") or tid),
            execution=str(blob.get("execution") or ""),
            plans_path=blob.get("plans_path"),
            state_path=blob.get("state_path"),
            log_path=blob.get("log_path"),
            symbol=blob.get("symbol"),
            pipeline_dte_min=blob.get("pipeline_dte_min"),
            pipeline_dte_max=blob.get("pipeline_dte_max"),
            short_dte=bool(blob.get("short_dte")),
            scalp_dte_min=blob.get("scalp_dte_min"),
            scalp_dte_max=blob.get("scalp_dte_max"),
            rth_env=blob.get("rth_env"),
            telegram_flag=blob.get("telegram_flag"),
            systemd_unit=blob.get("systemd_unit"),
        )
    return out


def large_options_pipeline_args_from_env() -> list[str]:
    """CLI flags for universe pipeline (large-options swing sleeve)."""
    args: list[str] = []
    if not _env_truthy("RLM_PIPELINE_SHORT_DTE"):
        dte_min = (os.environ.get("RLM_PIPELINE_DTE_MIN") or "7").strip()
        dte_max = (os.environ.get("RLM_PIPELINE_DTE_MAX") or "21").strip()
        if dte_min:
            args.extend(["--dte-min", dte_min])
        if dte_max:
            args.extend(["--dte-max", dte_max])
    else:
        args.append("--short-dte")
        for key, flag in (("RLM_PIPELINE_DTE_MIN", "--dte-min"), ("RLM_PIPELINE_DTE_MAX", "--dte-max")):
            v = (os.environ.get(key) or "").strip()
            if v:
                args.extend([flag, v])
    return args


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def track_health(root: Path) -> dict[str, dict[str, Any]]:
    """Lightweight on-disk health for post-deploy / status."""
    tracks = load_tracks()
    out: dict[str, dict[str, Any]] = {}
    for tid, spec in tracks.items():
        row: dict[str, Any] = {"label": spec.label, "execution": spec.execution}
        log_p = spec.resolve_log(root)
        if log_p is not None:
            row["log_exists"] = log_p.is_file()
            row["log_path"] = str(log_p)
            if log_p.is_file():
                row["log_mtime_utc"] = log_p.stat().st_mtime
        state_p = spec.resolve_state(root)
        if state_p is not None:
            row["state_exists"] = state_p.is_file()
            row["state_path"] = str(state_p)
        flag = spec.telegram_flag
        if flag:
            row["telegram_enabled"] = _env_truthy(flag)
        if tid == TRACK_LARGE_OPTIONS:
            row["pipeline_short_dte"] = _env_truthy("RLM_PIPELINE_SHORT_DTE")
            row["dte_min"] = (os.environ.get("RLM_PIPELINE_DTE_MIN") or "7").strip()
            row["dte_max"] = (os.environ.get("RLM_PIPELINE_DTE_MAX") or "21").strip()
        if tid == TRACK_SPY_DAYTRADE:
            row["symbol"] = (os.environ.get("RLM_CHALLENGE_SYMBOL") or spec.symbol or "SPY").strip()
            row["challenge_interval_sec"] = (os.environ.get("RLM_CHALLENGE_INTERVAL_SEC") or "").strip()
        out[tid] = row
    return out


def print_tracks_banner(root: Path | None = None) -> None:
    root = root or resolve_root()
    tracks = load_tracks()
    print("[tracks] three sleeves:", flush=True)
    for tid in (TRACK_LARGE_EQUITIES, TRACK_LARGE_OPTIONS, TRACK_SPY_DAYTRADE):
        spec = tracks.get(tid)
        if spec is None:
            continue
        log_p = spec.resolve_log(root)
        extra = ""
        if tid == TRACK_LARGE_OPTIONS:
            extra = (
                f" DTE {(os.environ.get('RLM_PIPELINE_DTE_MIN') or '7')}"
                f"-{(os.environ.get('RLM_PIPELINE_DTE_MAX') or '21')}"
                f" short_dte={_env_truthy('RLM_PIPELINE_SHORT_DTE')}"
            )
        if tid == TRACK_SPY_DAYTRADE:
            sym = (os.environ.get("RLM_CHALLENGE_SYMBOL") or spec.symbol or "SPY").strip()
            extra = f" symbol={sym} unit={spec.systemd_unit or 'rlm-challenge-loop'}"
        print(
            f"  · {spec.label}: {spec.execution} log={log_p.name if log_p else '—'}{extra}",
            flush=True,
        )
