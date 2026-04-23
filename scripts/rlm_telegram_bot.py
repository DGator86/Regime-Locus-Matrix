#!/usr/bin/env python3
"""
Telegram bot for RLM: commands + optional file-driven **push** alerts (options + equity).

**Push alerts** (when ``TELEGRAM_NOTIFY=1`` and ``notify_chat_id`` is set from ``/start`` or
``TELEGRAM_NOTIFY_CHAT_ID``): new **active** ``plan_id`` in ``universe_trade_plans.json``;
opens / take-profit / exit signals in ``trade_log.csv`` (monitor); equity open/close in
``equity_positions_state.json``.

**Commands**: /start, /help, /status, /universe, /portfolio, /balances, /brief (session timer JSON)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any
import urllib.error
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


def _long_poll_timeout_sec() -> int:
    raw = (os.environ.get("TELEGRAM_LONG_POLL_SEC") or "50").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 50
    return max(0, min(50, n))


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


def _resolve_state_path() -> Path:
    raw = (os.environ.get("TELEGRAM_STATE_PATH") or "").strip()
    if raw:
        return Path(raw) if Path(raw).is_absolute() else ROOT / raw
    return ROOT / "data" / "processed" / "telegram_notify_state.json"


def _api(token: str, method: str, **params: Any) -> dict[str, Any]:
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
    to = _long_poll_timeout_sec()
    params: dict[str, Any] = {"timeout": to}
    if offset is not None:
        params["offset"] = offset
    q = urlencode(params)
    url = f"https://api.telegram.org/bot{token}/getUpdates?{q}"
    with urlopen(Request(url, method="GET"), timeout=max(65, to + 15)) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    if not raw.get("ok"):
        raise RuntimeError(raw.get("description", raw))
    return raw.get("result") or []  # type: ignore[no-any-return]


def _handle_message(
    token: str,
    chat_id: int,
    user_id: int,
    text: str,
    allowed: set[int] | None,
) -> None:
    from rlm.notify.telegram_rlm import (
        build_balances_text,
        build_session_brief_text,
        build_status_brief,
        build_universe_and_positions,
        build_universe_report,
    )

    if allowed is not None and user_id not in allowed:
        _api(token, "sendMessage", chat_id=chat_id, text="Not authorized for this bot.")
        return
    t = (text or "").strip()
    t_low = t.lower()
    if t.startswith("/start"):
        st = _resolve_state_path()
        blob: dict[str, Any]
        if st.is_file():
            try:
                blob = json.loads(st.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                blob = {}
        else:
            blob = {}
        blob["notify_chat_id"] = chat_id
        st.parent.mkdir(parents=True, exist_ok=True)
        st.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        reply = (
            "RLM bot online. Push alerts use this chat. Commands: /help /status /universe /portfolio /balances /brief"
        )
    elif t_low.startswith("/help"):
        reply = (
            "/status — plan file summary\n"
            "/universe — ranked active trade ideas\n"
            "/portfolio — universe + open option rows (trade_log) + equity state\n"
            "/balances — IBKR net liq, cash, and STK/OPT position rows (needs Gateway + ibapi)\n"
            "/brief — last session_brief.json (pre/post-close timer run)\n"
            "Push alerts: new active universe plan_id; trade_log open / TP / exit; equity open/close."
        )
    elif t_low.startswith("/status"):
        reply = build_status_brief(ROOT)
    elif t_low.startswith("/portfolio") or t_low.startswith("/positions"):
        reply = build_universe_and_positions(ROOT, max_active=12, max_positions=20)
    elif t_low.startswith("/universe") or t_low.startswith("/report"):
        reply = build_universe_report(ROOT, max_active=12)
    elif t_low.startswith("/balances") or t_low.startswith("/balance"):
        reply = build_balances_text(ROOT)
    elif t_low.startswith("/brief") or t_low.startswith("/session"):
        reply = build_session_brief_text(ROOT)
    else:
        reply = "Unknown command. Try /help"
    for chunk in _chunk_text(str(reply)[:12000], 4000):
        _api(token, "sendMessage", chat_id=chat_id, text=chunk)


def _chunk_text(s: str, max_len: int) -> list[str]:
    if len(s) <= max_len:
        return [s]
    return [s[i : i + max_len] for i in range(0, len(s), max_len)]


def _chat_for_push() -> int | None:
    raw = (os.environ.get("TELEGRAM_NOTIFY_CHAT_ID") or "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    st = _resolve_state_path()
    if st.is_file():
        try:
            d = json.loads(st.read_text(encoding="utf-8"))
            c = d.get("notify_chat_id")
            if c is not None:
                return int(c)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return None


def _notify_thread_main(token: str) -> None:
    from rlm.notify.telegram_rlm import load_notify_state, notification_cycle

    st_path = _resolve_state_path()

    def send(msg: str) -> None:
        cid = _chat_for_push()
        if cid is None:
            return
        for chunk in _chunk_text(msg, 4000):
            try:
                _api(token, "sendMessage", chat_id=cid, text=chunk)
            except Exception as e:  # noqa: BLE001
                print(f"[rlm-telegram] push send error: {e}", flush=True)

    if (os.environ.get("TELEGRAM_NOTIFY") or "1").strip() not in ("1", "true", "yes", "on"):
        print("[rlm-telegram] TELEGRAM_NOTIFY=0 — background pushes disabled", flush=True)
        return
    # Block until a chat is known (/start or env), then run forever
    while _chat_for_push() is None:
        time.sleep(2.0)
    try:
        interval = float((os.environ.get("TELEGRAM_NOTIFY_INTERVAL_SEC") or "20").strip())
    except ValueError:
        interval = 20.0
    print(
        f"[rlm-telegram] background notify every {interval}s → chat {_chat_for_push()}",
        flush=True,
    )
    # Custom loop: reload chat id each cycle; merge state
    import time as _t

    while True:
        blob = load_notify_state(st_path)
        try:
            if _chat_for_push() is None:
                _t.sleep(5.0)
                continue
            messages, new_blob = notification_cycle(ROOT, blob)
            for m in messages:
                send(m)
            if new_blob != blob:
                st_path.parent.mkdir(parents=True, exist_ok=True)
                st_path.write_text(json.dumps(new_blob, indent=2, default=str), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            print(f"[rlm-telegram] notify cycle error: {e}", flush=True)
        _t.sleep(max(5.0, interval))


def main() -> int:
    _load_env()
    token = _token()
    allowed = _allowed()
    if allowed is not None:
        print(f"[rlm-telegram] allowed user IDs: {sorted(allowed)}", flush=True)
    else:
        print("[rlm-telegram] TELEGRAM_ALLOWED_USER_IDS not set — any user can talk to the bot", flush=True)
    lp = _long_poll_timeout_sec()
    print(f"[rlm-telegram] long-poll timeout={lp}s", flush=True)

    nt = threading.Thread(target=_notify_thread_main, args=(token,), name="rlm-telegram-notify", daemon=True)
    nt.start()

    last_offset: int | None = None
    while True:
        try:
            updates = _get_updates(token, last_offset)
        except urllib.error.HTTPError as e:
            if e.code == 409:
                print(
                    "[rlm-telegram] getUpdates HTTP 409 Conflict — only one client may long-poll this bot. "
                    "Stop any other rlm_telegram_bot, run_master.py --telegram-bot, IDE test, or second server "
                    "using the same TELEGRAM_BOT_TOKEN; then restart this service.",
                    flush=True,
                )
            else:
                print(f"[rlm-telegram] getUpdates HTTP {e.code}: {e.reason}; sleep 5s", flush=True)
            time.sleep(5)
            continue
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
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[rlm-telegram] stopped (Ctrl+C).", flush=True)
        raise SystemExit(0) from None
