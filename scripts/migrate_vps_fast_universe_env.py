#!/usr/bin/env python3
"""Apply VPS fast-universe .env profile (daily primary, parallel Massive, telegram flags).

Idempotent: replaces or appends known keys. Backs up .env before write.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

_FAST_PROFILE: dict[str, str] = {
    "RLM_STOCK_BARS_SOURCE": "eodhd",
    "RLM_ALLOW_DAILY_PRIMARY": "1",
    "RLM_PRIMARY_BAR_SIZE": "1 day",
    "RLM_PRIMARY_DURATION": "30 D",
    "RLM_PIPELINE_ARGS": (
        "--ignore-major-events --event-lookahead-days 0 --no-vix --massive-workers 4 "
        "--market-hours-only --short-dte --dte-min 0 --dte-max 5"
    ),
    "RLM_PIPELINE_SHORT_DTE": "1",
    "RLM_PIPELINE_DTE_MIN": "0",
    "RLM_PIPELINE_DTE_MAX": "5",
    "RLM_PIPELINE_MARKET_HOURS_ONLY": "1",
    "RLM_MONITOR_RTH_ONLY": "1",
    "RLM_EQUITY_RTH_ONLY": "1",
    "RLM_PIPELINE_TIMEOUT_SEC": "2700",
    "RLM_SKIP_FEATURE_CSV": "1",
    "RLM_SKIP_MASTER_CHALLENGE": "1",
    "TELEGRAM_NOTIFY_UNIVERSE": "1",
    "TELEGRAM_NOTIFY_CHALLENGE": "0",
    "RLM_OPTIONS_TRADE_LOG_PATH": "data/processed/options_large_account_trade_log.csv",
    "RLM_SHORT_DTE_SCORING": "1",
    "RLM_OPTIONS_MIN_BUYER_EDGE_PCT": "0.02",
    "RLM_OPTIONS_MAX_SPREAD_PCT_MID": "0.12",
}


def _parse_val(text: str, key: str) -> str | None:
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith(key + "="):
            val = s.split("=", 1)[1].strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            return val
    return None


def _apply_profile(text: str, profile: dict[str, str]) -> tuple[str, list[str]]:
    keys = set(profile)
    out: list[str] = []
    seen: set[str] = set()
    changes: list[str] = []

    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            out.append(line)
            continue
        hit = False
        for key in keys:
            if s.startswith(key + "="):
                old = _parse_val(text, key)
                new = profile[key]
                out.append(f"{key}={new}")
                seen.add(key)
                if old != new:
                    changes.append(f"{key}: {old!r} -> {new!r}")
                hit = True
                break
        if not hit:
            out.append(line)

    for key, val in profile.items():
        if key not in seen:
            out.append(f"{key}={val}")
            changes.append(f"{key}: (added) -> {val!r}")

    new_text = "\n".join(out)
    if text and not text.endswith("\n"):
        new_text += "\n"
    return new_text, changes


def _patch_live_regime_kronos_remote(
    repo_root: Path,
    env_path: Path,
    *,
    dry_run: bool,
) -> list[str]:
    """Turn on live-model Kronos blend when RunPod remote URL is configured."""
    remote = (_parse_val(env_path.read_text(encoding="utf-8"), "RLM_KRONOS_REMOTE_URL") or "").strip()
    if not remote:
        return []
    path = repo_root / "data" / "processed" / "live_regime_model.json"
    if not path.is_file():
        return []
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if blob.get("use_kronos") is True:
        return []
    blob["use_kronos"] = True
    if not dry_run:
        path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return [f"live_regime_model.json use_kronos -> true (remote {remote[:48]}...)"]


def _patch_live_regime_mtf_confirmation(repo_root: Path, *, dry_run: bool) -> list[str]:
    """Enable 5m/15m confirmation bars on live regime JSON when empty."""
    path = repo_root / "data" / "processed" / "live_regime_model.json"
    if not path.is_file():
        return []
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    th = blob.setdefault("timeframe_hierarchy", {})
    cur = th.get("confirmation_bar_sizes") or []
    want = ["5 mins", "15 mins"]
    if cur == want:
        return []
    th["confirmation_bar_sizes"] = want
    th.setdefault("confirmation_duration", "10 D")
    th.setdefault("confirmation_mode", "direction")
    if not dry_run:
        path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return [f"live_regime_model.json confirmation_bar_sizes -> {want!r}"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--env-path",
        type=Path,
        default=Path("/opt/Regime-Locus-Matrix/.env"),
        help="Target .env (default: VPS path)",
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/opt/Regime-Locus-Matrix"),
        help="Repo root for live_regime_model.json patch",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = ap.parse_args()
    p = args.env_path.resolve()
    if not p.is_file():
        print(f"error: {p} not found", flush=True)
        return 2

    text = p.read_text(encoding="utf-8")
    new_text, changes = _apply_profile(text, _FAST_PROFILE)
    if not changes:
        print(f"ok: {p} already matches fast profile", flush=True)
    else:
        for c in changes:
            print(c, flush=True)
        if args.dry_run:
            print("dry-run: no .env write", flush=True)
        else:
            stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = p.with_suffix(f".env.bak.{stamp}")
            shutil.copy2(p, backup)
            p.write_text(new_text, encoding="utf-8")
            print(f"wrote {p} (backup {backup.name})", flush=True)

    repo_root = args.repo_root.resolve()
    for line in _patch_live_regime_kronos_remote(
        repo_root,
        p,
        dry_run=args.dry_run,
    ):
        print(line, flush=True)
    for line in _patch_live_regime_mtf_confirmation(
        repo_root,
        dry_run=args.dry_run,
    ):
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
