"""
StarfleetCrew — orchestrates Scotty, Spock, and Kirk.

Run cadence (configurable via env):
  CREW_HEALTH_INTERVAL   seconds between Scotty health checks   (default 120)
  CREW_ANALYSIS_INTERVAL seconds between Spock market analyses  (default 300)
  CREW_BRIEFING_INTERVAL seconds between Kirk full briefings     (default 600)

Telegram integration uses the same env vars as the rest of RLM:
  TELEGRAM_BOT_TOKEN, TELEGRAM_NOTIFY_CHAT_ID

If ``TELEGRAM_NOTIFY_CHAT_ID`` is unset, the crew reads ``notify_chat_id`` from
``data/processed/telegram_notify_state.json`` (written when you send ``/start`` to the RLM bot).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rlm.agents.base import LLMClient, LLMConfig
from rlm.agents.kirk import CommandDecision, KirkAgent
from rlm.agents.scotty import HealthReport, ScottyAgent
from rlm.agents.spock import SpockAgent, SpockBriefing

# -----------------------------------------------------------------------
# Telegram helper (standalone, no dependency on telegram_rlm.py)
# -----------------------------------------------------------------------


def _notify_state_path(root: Path) -> Path:
    raw = (os.environ.get("TELEGRAM_STATE_PATH") or "").strip()
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else root / p
    return root / "data" / "processed" / "telegram_notify_state.json"


def resolve_telegram_chat_id(root: Path) -> str:
    """``TELEGRAM_NOTIFY_CHAT_ID`` or ``notify_chat_id`` from bot ``/start`` state file."""
    raw = (os.environ.get("TELEGRAM_NOTIFY_CHAT_ID") or "").strip()
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


def _tg_send(text: str, token: str, chat_id: str, *, silent: bool = False) -> bool:
    if not token:
        print("[crew-telegram] TELEGRAM_BOT_TOKEN missing — not sending", flush=True)
        return False
    if not chat_id:
        print(
            "[crew-telegram] No chat id — set TELEGRAM_NOTIFY_CHAT_ID in .env or send /start to the bot",
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


# -----------------------------------------------------------------------


@dataclass
class CrewConfig:
    health_interval: int = int(os.environ.get("CREW_HEALTH_INTERVAL", "60"))
    analysis_interval: int = int(os.environ.get("CREW_ANALYSIS_INTERVAL", "60"))
    briefing_interval: int = int(os.environ.get("CREW_BRIEFING_INTERVAL", "3600"))
    telegram_token: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.environ.get("TELEGRAM_NOTIFY_CHAT_ID", "")
    silent_health_ok: bool = True  # don't spam Telegram when everything is fine
    services: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.services is None:
            raw = os.environ.get("CREW_SERVICES", "")
            self.services = [s.strip() for s in raw.split(",") if s.strip()] or None


class StarfleetCrew:
    """
    Runs all three agents on a schedule and routes their output to Telegram.

    Usage::

        crew = StarfleetCrew(root=Path("/opt/Regime-Locus-Matrix"))
        crew.run()          # blocking loop
    """

    def __init__(
        self,
        root: Path,
        config: Optional[CrewConfig] = None,
        llm: Optional[LLMClient] = None,
    ) -> None:
        self.root = root
        self.cfg = config or CrewConfig()
        cid = resolve_telegram_chat_id(root)
        if cid:
            self.cfg.telegram_chat_id = cid
        elif not (self.cfg.telegram_chat_id or "").strip():
            print(
                "[Crew] No Telegram chat id — set TELEGRAM_NOTIFY_CHAT_ID or send /start to the RLM bot once.",
                flush=True,
            )
        self.llm = llm or LLMClient(LLMConfig.from_env())

        scotty_services = self.cfg.services or None
        self.scotty = ScottyAgent(root, self.llm, services=scotty_services)
        self.spock = SpockAgent(root, self.llm)
        self.kirk = KirkAgent(root, self.llm)

        self._last_health: float = 0.0
        self._last_analysis: float = 0.0
        self._last_briefing: float = 0.0

        # Cache last report objects so Kirk can use them when triggered on interval
        self._last_health_report: Optional[HealthReport] = None
        self._last_scotty_diag: str = ""
        self._last_briefing_obj: Optional[SpockBriefing] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Blocking main loop."""
        # Initial status only
        self._send(
            "<b>[Starfleet Crew] Online</b>\n"
            "Autonomous Execution: ACTIVE | Silent Mode: ON\n"
            f"Cadence: Health {self.cfg.health_interval}s | Analysis {self.cfg.analysis_interval}s | Briefing {self.cfg.briefing_interval}s",
            force_notify=True,
        )
        while True:
            tick_start = time.monotonic()
            self._tick()
            elapsed = time.monotonic() - tick_start
            time.sleep(max(0.0, 10.0 - elapsed))

    def run_once(self) -> CommandDecision:
        """Single full cycle — useful for testing or cron invocation."""
        report, diagnosis = self.scotty.check()
        briefing = self.spock.analyse()
        decision = self.kirk.command(report, briefing, diagnosis)
        self._execute_orders(decision)
        return decision

    # ------------------------------------------------------------------
    # Tick logic
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        from rlm.agents.gate import SystemGate

        gate = SystemGate(self.root)

        now = time.monotonic()

        # Scotty health check
        if now - self._last_health >= self.cfg.health_interval:
            self._last_health = now
            try:
                report, diagnosis = self.scotty.check()
                self._last_health_report = report
                self._last_scotty_diag = diagnosis
                # INTERNAL ACTION: Scotty already tries to restart services in scotty.check()
            except Exception as exc:
                print(f"[Scotty ERROR] {exc}", flush=True)

        # Spock market analysis
        if now - self._last_analysis >= self.cfg.analysis_interval:
            self._last_analysis = now
            try:
                briefing = self.spock.analyse()
                self._last_briefing_obj = briefing
            except Exception as exc:
                print(f"[Spock ERROR] {exc}", flush=True)

        # Kirk full command briefing
        if now - self._last_briefing >= self.cfg.briefing_interval:
            self._last_briefing = now
            if self._last_health_report and self._last_briefing_obj:
                try:
                    decision = self.kirk.command(
                        self._last_health_report,
                        self._last_briefing_obj,
                        self._last_scotty_diag,
                    )
                    self._execute_orders(decision, gate)
                    # ONLY NOTIFY if it's a critical system failure that requires human manual bypass
                    if (
                        decision.system_status == "CRITICAL"
                        and decision.command == "ALERT OPERATOR"
                    ):
                        self._send(decision.to_telegram_message(), force_notify=True)
                except Exception as exc:
                    print(f"[Kirk ERROR] {exc}", flush=True)

        # EOD Report (triggered once daily around 16:15 ET / 20:15 UTC)
        # We use a simple hour/minute check.
        from datetime import datetime, timezone

        now_utc = datetime.now(timezone.utc)
        if now_utc.hour == 20 and now_utc.minute >= 15 and now_utc.minute < 30:
            if not getattr(self, "_eod_sent_today", False):
                try:
                    from rlm.notify.pnl_report import calculate_daily_pnl

                    report_text = calculate_daily_pnl(self.root)
                    self._send(report_text, force_notify=True)
                    self._eod_sent_today = True
                except Exception as exc:
                    print(f"[EOD ERROR] {exc}", flush=True)
        elif now_utc.hour != 20:
            self._eod_sent_today = False

    def _execute_orders(self, decision: CommandDecision, gate: Optional[SystemGate] = None) -> None:
        """Translate Kirk's decision into system-level state."""
        from rlm.agents.gate import SystemGate

        g = gate or SystemGate(self.root)
        g.update(
            posture=decision.market_posture,
            status=decision.system_status,
            timestamp=decision.timestamp,
        )
        print(
            f"[Crew] Executed Command: {decision.command} (Posture: {decision.market_posture})",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Formatting (kept for reference or manual runs)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_full_briefing(
        report: HealthReport,
        diagnosis: str,
        briefing: SpockBriefing,
        decision: CommandDecision,
    ) -> str:
        parts = [
            "<b>[Scotty]</b>",
            diagnosis[:600],
            "",
            "<b>[Spock]</b>",
            briefing.llm_text[:800],
            "",
            "<b>[Kirk]</b>",
            decision.to_telegram_message(),
        ]
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Telegram
    # ------------------------------------------------------------------

    def _send(self, text: str, *, force_notify: bool = False) -> None:
        """Send message only if force_notify is True (silences routine chatter)."""
        if not force_notify:
            return

        cid = (self.cfg.telegram_chat_id or "").strip() or resolve_telegram_chat_id(self.root)
        if cid:
            self.cfg.telegram_chat_id = cid
        _tg_send(
            text,
            self.cfg.telegram_token,
            cid,
            silent=False,
        )
        print(text, flush=True)
