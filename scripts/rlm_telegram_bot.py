#!/usr/bin/env python3
"""
Telegram bot for RLM: commands + optional file-driven **push** alerts (options + equity).

**Push alerts** (when ``TELEGRAM_NOTIFY=1`` and ``notify_chat_id`` is set from ``/start`` or
``TELEGRAM_NOTIFY_CHAT_ID``): new **active** ``plan_id`` in ``universe_trade_plans.json``;
opens / take-profit / exit signals in ``trade_log.csv`` (monitor); equity open/close in
``equity_positions_state.json``.

**Commands**: /start, /help, /status, /pnl, /universe, /portfolio, /balances, /brief (session timer JSON)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from rlm.data.paths import get_rlm_runtime_root  # noqa: E402

_NOTIFY_STATE_LOCK = threading.Lock()
_NOTIFY_CYCLE_MANAGED_STATE_KEYS = frozenset(
    {
        "notify_seeded",
        "announced_trade_open",
        "last_opt_signal",
        "announced_tp",
        "announced_exit",
        "last_equity_open",
        "announced_equity_close",
        "last_universe_active_ids",
        "challenge_trade_n",
        "challenge_open_ids",
    }
)


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    runtime = get_rlm_runtime_root()
    for cand in (runtime / ".env", REPO / ".env"):
        if cand.is_file():
            load_dotenv(cand, override=True)


def _env_first(*keys: str) -> str:
    for k in keys:
        v = (os.environ.get(k) or "").strip()
        if v:
            return v
    return ""


def _token() -> str:
    t = _env_first(
        "RLM_SYSTEMS_CONTROL_TELEGRAM_BOT_TOKEN",
        "TELEGRAM_BOT_TOKEN",
    )
    if not t:
        print(
            "Set RLM_SYSTEMS_CONTROL_TELEGRAM_BOT_TOKEN (preferred) or TELEGRAM_BOT_TOKEN in the environment or .env",
            file=sys.stderr,
        )
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
    allow_all = _env_first(
        "RLM_SYSTEMS_CONTROL_TELEGRAM_ALLOW_ALL_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
    ).lower()
    if allow_all in {"1", "true", "yes", "on"}:
        return None

    raw = _env_first(
        "RLM_SYSTEMS_CONTROL_TELEGRAM_ALLOWED_USER_IDS",
        "TELEGRAM_ALLOWED_USER_IDS",
    )
    out: set[int] = set()
    for part in raw.replace(";", ",").split(","):
        p = part.strip()
        if p.isdigit() or (p.startswith("-") and p[1:].isdigit()):
            out.add(int(p))
    if out:
        return out

    # In private chats Telegram uses the same integer for chat_id and user_id,
    # so a configured push chat can safely double as the default single-user
    # allow-list. Group chats must set explicit user IDs.
    chat_raw = _env_first(
        "RLM_SYSTEMS_CONTROL_TELEGRAM_CHAT_ID",
        "TELEGRAM_NOTIFY_CHAT_ID",
    )
    if chat_raw.isdigit():
        return {int(chat_raw)}
    return set()


def _resolve_state_path() -> Path:
    raw = (os.environ.get("TELEGRAM_STATE_PATH") or "").strip()
    rt = get_rlm_runtime_root()
    if raw:
        return Path(raw) if Path(raw).is_absolute() else rt / raw
    return rt / "data" / "processed" / "telegram_notify_state.json"


def _load_notify_state_blob(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_notify_state_blob(path: Path, blob: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=2, default=str), encoding="utf-8")


def _persist_notify_cycle_state(path: Path, cycle_blob: dict[str, Any]) -> None:
    """Persist notify de-dupe state without overwriting a concurrent /start chat binding."""
    with _NOTIFY_STATE_LOCK:
        latest = _load_notify_state_blob(path)
        merged = dict(latest)
        for key in _NOTIFY_CYCLE_MANAGED_STATE_KEYS:
            if key in cycle_blob:
                merged[key] = cycle_blob[key]
            else:
                merged.pop(key, None)
        if merged != latest:
            _write_notify_state_blob(path, merged)


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
        build_pnl_text,
        build_session_brief_text,
        build_status_brief,
        build_universe_and_positions,
        build_universe_report,
    )

    if allowed is not None and user_id not in allowed:
        _api(token, "sendMessage", chat_id=chat_id, text="Not authorized for this bot.")
        return
    root = get_rlm_runtime_root()
    t = (text or "").strip()
    t_low = t.lower()
    if t.startswith("/start"):
        st = _resolve_state_path()
        with _NOTIFY_STATE_LOCK:
            blob = _load_notify_state_blob(st)
            blob["notify_chat_id"] = chat_id
            _write_notify_state_blob(st, blob)
        reply = (
            "RLM bot online. Push alerts use this chat.\n"
            "Commands: /help /status /pnl /universe /portfolio /balances /brief"
        )
    elif t_low.startswith("/help"):
        reply = (
            "/status — plan file summary\n"
            "/pnl — daily / weekly / all-time P&L for equities, options, PDT challenge + IBKR balances\n"
            "/universe — ranked active trade ideas\n"
            "/portfolio — universe + open option rows (trade_log) + equity state\n"
            "/balances — IBKR net liq, cash, and STK/OPT position rows (needs Gateway + ibapi)\n"
            "/brief — last session_brief.json (pre/post-close timer run)\n"
            "Push alerts: new active universe plan_id; trade_log open / TP / exit; equity open/close."
        )
    elif t_low.startswith("/status"):
        reply = build_status_brief(root)
    elif t_low.startswith("/pnl"):
        reply = build_pnl_text(root)
    elif t_low.startswith("/portfolio") or t_low.startswith("/positions"):
        reply = build_universe_and_positions(root, max_active=12, max_positions=20)
    elif t_low.startswith("/universe") or t_low.startswith("/report"):
        reply = build_universe_report(root, max_active=12)
    elif t_low.startswith("/balances") or t_low.startswith("/balance"):
        reply = build_balances_text(root)
    elif t_low.startswith("/brief") or t_low.startswith("/session"):
        reply = build_session_brief_text(root)
    else:
        reply = "Unknown command. Try /help"
    for chunk in _chunk_text(str(reply)[:12000], 4000):
        _api(token, "sendMessage", chat_id=chat_id, text=chunk)


def _chunk_text(s: str, max_len: int) -> list[str]:
    if len(s) <= max_len:
        return [s]
    return [s[i : i + max_len] for i in range(0, len(s), max_len)]


def _chat_for_push(allowed: set[int] | None = None) -> int | None:
    raw = _env_first(
        "RLM_SYSTEMS_CONTROL_TELEGRAM_CHAT_ID",
        "TELEGRAM_NOTIFY_CHAT_ID",
    )
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    st = _resolve_state_path()
    if st.is_file():
        with _NOTIFY_STATE_LOCK:
            d = _load_notify_state_blob(st)
        try:
            c = d.get("notify_chat_id")
            if c is not None:
                cid = int(c)
                if allowed is None or cid in allowed:
                    return cid
        except (ValueError, TypeError):
            pass
    return None


def _notify_thread_main(token: str, allowed: set[int] | None) -> None:
    from rlm.notify.telegram_rlm import notification_cycle

    st_path = _resolve_state_path()
    root = get_rlm_runtime_root()

    def send(msg: str) -> None:
        cid = _chat_for_push(allowed)
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
    while _chat_for_push(allowed) is None:
        time.sleep(2.0)
    try:
        interval = float((os.environ.get("TELEGRAM_NOTIFY_INTERVAL_SEC") or "20").strip())
    except ValueError:
        interval = 20.0
    print(
        f"[rlm-telegram] background notify every {interval}s → chat {_chat_for_push(allowed)}",
        flush=True,
    )
    # Custom loop: reload chat id each cycle; merge state
    import time as _t

    while True:
        with _NOTIFY_STATE_LOCK:
            blob = _load_notify_state_blob(st_path)
        try:
            if _chat_for_push(allowed) is None:
                _t.sleep(5.0)
                continue
            messages, new_blob = notification_cycle(root, blob)
            for m in messages:
                send(m)
            if new_blob != blob:
                _persist_notify_cycle_state(st_path, new_blob)
        except Exception as e:  # noqa: BLE001
            print(f"[rlm-telegram] notify cycle error: {e}", flush=True)
        _t.sleep(max(5.0, interval))


def main() -> int:
    _load_env()
    os.environ.setdefault("RLM_ROOT", str(REPO.resolve()))
    token = _token()
    allowed = _allowed()
    if allowed is not None:
        if allowed:
            print(f"[rlm-telegram] allowed user IDs: {sorted(allowed)}", flush=True)
        else:
            print(
                "[rlm-telegram] no allowed user IDs configured — commands and state-based pushes are disabled. "
                "Set RLM_SYSTEMS_CONTROL_TELEGRAM_ALLOWED_USER_IDS (preferred) or "
                "RLM_SYSTEMS_CONTROL_TELEGRAM_ALLOW_ALL_USERS=1 for an intentionally public/testing bot.",
                flush=True,
            )
    else:
        print(
            "[rlm-telegram] allow-all Telegram mode enabled by explicit configuration",
            flush=True,
        )
    lp = _long_poll_timeout_sec()
    print(f"[rlm-telegram] long-poll timeout={lp}s", flush=True)

    nt = threading.Thread(target=_notify_thread_main, args=(token, allowed), name="rlm-telegram-notify", daemon=True)
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
                    "using the same RLM_SYSTEMS_CONTROL_TELEGRAM_BOT_TOKEN/TELEGRAM_BOT_TOKEN; then restart this service.",
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
