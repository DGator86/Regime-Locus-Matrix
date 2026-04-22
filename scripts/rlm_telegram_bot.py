#!/usr/bin/env python3
"""
Minimal Telegram bot for RLM (long-poll, no extra deps — stdlib + python-dotenv).

1. In Telegram, open @BotFather → /newbot (or your existing bot) → copy the **token**
   (keeps it *only* in your server ``.env``, never in git or chat).
2. On the server::

    export TELEGRAM_BOT_TOKEN=...
    # optional: only these numeric user IDs can use the bot
    # export TELEGRAM_ALLOWED_USER_IDS=123456789,987654321
    python3 scripts/rlm_telegram_bot.py

3. Systemd: run like other long-running services, ``WorkingDirectory`` = repo root,
   ``EnvironmentFile`` = ``.env``, ``Restart=always``.

Commands: /start, /help, /status (summary from ``universe_trade_plans.json`` if present)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    p = ROOT / ".env"
    if p.is_file():
        load_dotenv(p)


def _token() -> str:
    t = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    if not t:
        print("Set TELEGRAM_BOT_TOKEN in the environment or .env", file=sys.stderr)
        raise SystemExit(1)
    return t


def _allowed() -> set[int] | None:
    raw = (os.environ.get("TELEGRAM_ALLOWED_USER_IDS") or "").strip()
    if not raw:
        return None
    out: set[int] = set()
    for part in raw.replace(";", ",").split(","):
        p = part.strip()
        if p.isdigit() or (p.startswith("-") and p[1:].isdigit()):
            out.add(int(p))
    return out if out else None


def _api(token: str, method: str, **params: Any) -> dict[str, Any]:
    """POST to Telegram Bot API; form-encoded body."""
    url = f"https://api.telegram.org/bot{token}/{method}"
    body = urlencode({k: v for k, v in params.items() if v is not None}).encode("utf-8")
    req = Request(url, data=body, method="POST", headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urlopen(req, timeout=65) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    if not raw.get("ok"):
        err = raw.get("description", raw)
        raise RuntimeError(f"Telegram API error: {err}")
    return raw["result"]  # type: ignore[no-any-return]


def _get_updates(token: str, offset: int | None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"timeout": 50}
    if offset is not None:
        params["offset"] = offset
    q = urlencode(params)
    url = f"https://api.telegram.org/bot{token}/getUpdates?{q}"
    with urlopen(Request(url, method="GET"), timeout=60) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    if not raw.get("ok"):
        raise RuntimeError(raw.get("description", raw))
    return raw.get("result") or []  # type: ignore[no-any-return]


def _status_text() -> str:
    path = ROOT / "data" / "processed" / "universe_trade_plans.json"
    if not path.is_file():
        return "No `universe_trade_plans.json` yet (run the universe pipeline first)."
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return f"Could not read plans: {e}"
    gen = data.get("generated_at_utc", "?")
    results = data.get("results") or []
    n_active = sum(1 for r in results if r.get("status") == "active")
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    return (
        f"**Plans file**\n"
        f"- generated_at: `{gen}`\n"
        f"- file mtime (UTC): `{mtime}`\n"
        f"- active count: {n_active}\n"
    )


def _handle_message(
    token: str,
    chat_id: int,
    user_id: int,
    text: str,
    allowed: set[int] | None,
) -> None:
    if allowed is not None and user_id not in allowed:
        _api(token, "sendMessage", chat_id=chat_id, text="Not authorized for this bot.")
        return
    t = (text or "").strip()
    if t.startswith("/start"):
        reply = "RLM bot online. Use /status for trade plans summary, /help for commands."
    elif t.startswith("/help"):
        reply = "Commands: /start, /help, /status"
    elif t.startswith("/status"):
        reply = _status_text()
    else:
        reply = "Unknown command. Try /help"
    # Telegram MarkdownV2 is picky; use plain for status file paths
    _api(token, "sendMessage", chat_id=chat_id, text=reply[:4000])


def main() -> int:
    _load_env()
    token = _token()
    allowed = _allowed()
    if allowed is not None:
        print(f"[rlm-telegram] allowed user IDs: {sorted(allowed)}", flush=True)
    else:
        print("[rlm-telegram] TELEGRAM_ALLOWED_USER_IDS not set — any user can talk to the bot", flush=True)
    last_offset: int | None = None
    while True:
        try:
            updates = _get_updates(token, last_offset)
        except Exception as e:
            print(f"[rlm-telegram] getUpdates error: {e}; sleep 5s", flush=True)
            time.sleep(5)
            continue
        if updates:
            umax = max(int(u["update_id"]) for u in updates if isinstance(u.get("update_id"), int))
            last_offset = umax + 1
        for u in updates:
            msg = u.get("message") or u.get("edited_message")
            if not isinstance(msg, dict):
                continue
            from_user = msg.get("from") or {}
            user_id = int(from_user.get("id") or 0)
            chat = msg.get("chat") or {}
            chat_id = int(chat.get("id") or 0)
            text = str(msg.get("text") or "")
            if chat_id and user_id:
                try:
                    _handle_message(token, chat_id, user_id, text, allowed)
                except Exception as e:
                    print(f"[rlm-telegram] handle error: {e}", flush=True)
        time.sleep(0.1)


if __name__ == "__main__":
    raise SystemExit(main())
