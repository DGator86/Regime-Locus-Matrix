#!/usr/bin/env python3
"""Apply VPS fast-universe .env profile (daily primary, parallel Massive, telegram flags).

Idempotent: replaces or appends known keys. Backs up .env before write.
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

_FAST_PROFILE: dict[str, str] = {
    "RLM_STOCK_BARS_SOURCE": "eodhd",
    "RLM_ALLOW_DAILY_PRIMARY": "1",
    "RLM_PIPELINE_ARGS": "--ignore-major-events --event-lookahead-days 0 --no-vix --massive-workers 4",
    "RLM_PIPELINE_TIMEOUT_SEC": "2700",
    "RLM_SKIP_FEATURE_CSV": "1",
    "RLM_SKIP_MASTER_CHALLENGE": "1",
    "TELEGRAM_NOTIFY_UNIVERSE": "1",
    "TELEGRAM_NOTIFY_CHALLENGE": "0",
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--env-path",
        type=Path,
        default=Path("/opt/Regime-Locus-Matrix/.env"),
        help="Target .env (default: VPS path)",
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
        return 0

    for c in changes:
        print(c, flush=True)
    if args.dry_run:
        print("dry-run: no write", flush=True)
        return 0

    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = p.with_suffix(f".env.bak.{stamp}")
    shutil.copy2(p, backup)
    p.write_text(new_text, encoding="utf-8")
    print(f"wrote {p} (backup {backup.name})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
