"""Telegram helpers for crew / Hermes orchestrator notifications (stdlib only)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path


def _notify_state_path(root: Path) -> Path:
    raw = (os.environ.get("TELEGRAM_STATE_PATH") or "").strip()
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else root / p
    return root / "data" / "processed" / "telegram_notify_state.json"


def resolve_telegram_chat_id(root: Path) -> str:
    """``TELEGRAM_NOTIFY_CHAT_ID`` or ``notify_chat_id`` from bot ``/start`` state file."""
    raw = (
        (os.environ.get("RLM_HERMES_TELEGRAM_CHAT_ID") or "").strip()
        or (os.environ.get("TELEGRAM_NOTIFY_CHAT_ID") or "").strip()
    )
    if raw:
        return raw
    st = _notify_state_path(root)
    if not st.is_file():
        return ""
    try:
        d = json.loads(st.read_text(encoding="utf-8"))
        c = d.get("notify_chat_id")
        if c is not None:
            return str(int(c))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return ""


def telegram_crew_send(text: str, token: str, chat_id: str, *, silent: bool = False) -> bool:
    if not token:
        print(
            "[crew-telegram] RLM_HERMES_TELEGRAM_BOT_TOKEN (or TELEGRAM_BOT_TOKEN) missing — not sending",
            flush=True,
        )
        return False
    if not chat_id:
        print(
            "[crew-telegram] No chat id — set RLM_HERMES_TELEGRAM_CHAT_ID (or TELEGRAM_NOTIFY_CHAT_ID) in .env",
            flush=True,
        )
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    def _post(payload: dict):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=15)

    payload: dict = {
        "chat_id": chat_id,
        "text": text[:4000],
        "parse_mode": "HTML",
        "disable_notification": silent,
    }
    try:
        _post(payload)
        return True
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")[:800]
        except Exception:
            err_body = str(e)
        print(f"[crew-telegram] sendMessage HTTP {e.code}: {err_body}", flush=True)
        if e.code == 400:
            try:
                plain = {
                    "chat_id": chat_id,
                    "text": text[:4000],
                    "disable_notification": silent,
                }
                _post(plain)
                print("[crew-telegram] retry without HTML parse_mode succeeded", flush=True)
                return True
            except Exception as e2:
                print(f"[crew-telegram] retry failed: {e2}", flush=True)
        return False
    except Exception as e:
        print(f"[crew-telegram] sendMessage error: {e}", flush=True)
        return False
