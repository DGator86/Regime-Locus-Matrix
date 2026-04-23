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
    health_interval: int = int(os.environ.get("CREW_HEALTH_INTERVAL", "120"))
    analysis_interval: int = int(os.environ.get("CREW_ANALYSIS_INTERVAL", "300"))
    briefing_interval: int = int(os.environ.get("CREW_BRIEFING_INTERVAL", "600"))
    telegram_token: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.environ.get("TELEGRAM_NOTIFY_CHAT_ID", "")
    silent_health_ok: bool = True     # don't spam Telegram when everything is fine
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
        self._send(
            "<b>[Starfleet Crew] Online</b>\n"
            f"Scotty (health every {self.cfg.health_interval}s) | "
            f"Spock (analysis every {self.cfg.analysis_interval}s) | "
            f"Kirk (briefings every {self.cfg.briefing_interval}s)\n"
            f"LLM backend: {self.llm.cfg.backend} / {self.llm.cfg.model}",
            force_notify=True,
        )
        while True:
            now = time.monotonic()
            self._tick(now)
            time.sleep(10)

    def run_once(self) -> CommandDecision:
        """Single full cycle — useful for testing or cron invocation."""
        report, diagnosis = self.scotty.check()
        briefing = self.spock.analyse()
        decision = self.kirk.command(report, briefing, diagnosis)
        self._send(
            self._format_full_briefing(report, diagnosis, briefing, decision),
            force_notify=True,
        )
        return decision

    # ------------------------------------------------------------------
    # Tick logic
    # ------------------------------------------------------------------

    def _tick(self, now: float) -> None:
        # Scotty health check
        if now - self._last_health >= self.cfg.health_interval:
            self._last_health = now
            try:
                report, diagnosis = self.scotty.check()
                self._last_health_report = report
                self._last_scotty_diag = diagnosis
                if not report.overall_ok or not self.cfg.silent_health_ok:
                    self._send(
                        f"<b>[Scotty]</b>\n{diagnosis}",
                        force_notify=(not report.overall_ok) or (not self.cfg.silent_health_ok),
                    )
            except Exception as exc:
                self._send(f"<b>[Scotty ERROR]</b> {exc}", force_notify=True)

        # Spock market analysis
        if now - self._last_analysis >= self.cfg.analysis_interval:
            self._last_analysis = now
            try:
                briefing = self.spock.analyse()
                self._last_briefing_obj = briefing
                # Only send if Spock found something meaningful
                if briefing.overall_risk in ("HIGH", "CRITICAL"):
                    self._send(
                        f"<b>[Spock]</b> Risk: {briefing.overall_risk}\n{briefing.llm_text[:1200]}",
                        force_notify=True,
                    )
            except Exception as exc:
                self._send(f"<b>[Spock ERROR]</b> {exc}", force_notify=True)

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
                    # Only send automated briefing if scanner is open OR if system is in trouble
                    from rlm.utils.market_hours import is_scanner_window_open
                    if is_scanner_window_open() or decision.system_status != "NOMINAL":
                        self._send(decision.to_telegram_message(), force_notify=True)
                except Exception as exc:
                    self._send(f"<b>[Kirk ERROR]</b> {exc}", force_notify=True)

    # ------------------------------------------------------------------
    # Formatting
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
        """If ``force_notify``, Telegram message rings even when ``silent_health_ok``."""
        silent = bool(self.cfg.silent_health_ok and not force_notify)
        cid = (self.cfg.telegram_chat_id or "").strip() or resolve_telegram_chat_id(self.root)
        if cid:
            self.cfg.telegram_chat_id = cid
        _tg_send(
            text,
            self.cfg.telegram_token,
            cid,
            silent=silent,
        )
        print(text, flush=True)
