#!/usr/bin/env python3
"""Point TELEGRAM_BOT_TOKEN at RLM_SYSTEMS_CONTROL_TELEGRAM_BOT_TOKEN when the legacy token is stale."""

from __future__ import annotations

import sys
from pathlib import Path


def sync_env(env_path: Path) -> bool:
    lines = env_path.read_text(encoding="utf-8").splitlines()
    sc: str | None = None
    for line in lines:
        if line.startswith("RLM_SYSTEMS_CONTROL_TELEGRAM_BOT_TOKEN="):
            sc = line.split("=", 1)[1]
            break
    if not sc:
        print("RLM_SYSTEMS_CONTROL_TELEGRAM_BOT_TOKEN not set — nothing to sync", file=sys.stderr)
        return False

    out: list[str] = []
    had_legacy = False
    for line in lines:
        if line.startswith("TELEGRAM_BOT_TOKEN="):
            had_legacy = True
            out.append(f"TELEGRAM_BOT_TOKEN={sc}")
            continue
        out.append(line)
    if not had_legacy:
        out.append(f"TELEGRAM_BOT_TOKEN={sc}")

    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return True


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    env = Path(sys.argv[1]) if len(sys.argv) > 1 else root / ".env"
    if not env.is_file():
        print(f"Missing {env}", file=sys.stderr)
        return 1
    if sync_env(env):
        print(f"Synced TELEGRAM_BOT_TOKEN from systems-control token in {env}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
