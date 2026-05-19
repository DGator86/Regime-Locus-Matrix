"""Alpha Vantage GLOBAL_QUOTE with on-disk cache (free tier is 25 calls/day).

Used as a fallback when IBKR ``reqMktData`` returns no tick (e.g. error 10089).
Set ``ALPHA_VANTAGE_API_KEY`` in ``.env``. Tune cache with ``RLM_ALPHAVANTAGE_CACHE_SEC``
(default 900s) to stay under rate limits.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

_AV_BASE = "https://www.alphavantage.co/query"


def load_alphavantage_api_key() -> str | None:
    raw = (os.environ.get("ALPHA_VANTAGE_API_KEY") or "").strip()
    return raw or None


def _cache_path(root: Path) -> Path:
    return root / "data" / "processed" / "alphavantage_quote_cache.json"


def _cache_ttl_sec() -> float:
    raw = (os.environ.get("RLM_ALPHAVANTAGE_CACHE_SEC") or "900").strip()
    try:
        return max(60.0, float(raw))
    except ValueError:
        return 900.0


def _read_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_cache(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _parse_global_quote(payload: dict[str, Any]) -> float | None:
    if payload.get("Note") or payload.get("Information"):
        return None
    block = payload.get("Global Quote")
    if not isinstance(block, dict):
        return None
    raw = block.get("05. price") or block.get("price")
    if raw is None:
        return None
    try:
        val = float(str(raw).strip())
    except (ValueError, TypeError):
        return None
    return val if val > 0 else None


def fetch_global_quote(symbol: str, *, api_key: str | None = None) -> float | None:
    """One live GLOBAL_QUOTE request (no cache)."""
    key = (api_key or load_alphavantage_api_key() or "").strip()
    if not key:
        return None
    sym = str(symbol or "").strip().upper()
    if not sym:
        return None
    qs = urllib.parse.urlencode(
        {"function": "GLOBAL_QUOTE", "symbol": sym, "apikey": key}
    )
    url = f"{_AV_BASE}?{qs}"
    try:
        with urllib.request.urlopen(url, timeout=20.0) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return _parse_global_quote(data)


def fetch_cached_global_quote(
    root: Path,
    symbol: str,
    *,
    api_key: str | None = None,
    now_mono: float | None = None,
) -> float | None:
    """Return last price, using a file cache under ``data/processed/``."""
    sym = str(symbol or "").strip().upper()
    if not sym:
        return None
    path = _cache_path(root)
    cache = _read_cache(path)
    entry = cache.get(sym)
    ttl = _cache_ttl_sec()
    mono = time.monotonic() if now_mono is None else now_mono
    if isinstance(entry, dict):
        try:
            price = float(entry.get("price"))
            cached_at = float(entry.get("mono", 0))
        except (TypeError, ValueError):
            price, cached_at = 0.0, 0.0
        if price > 0 and (mono - cached_at) < ttl:
            return price

    fresh = fetch_global_quote(sym, api_key=api_key)
    if fresh is None or fresh <= 0:
        if isinstance(entry, dict):
            try:
                stale = float(entry.get("price"))
                if stale > 0:
                    return stale
            except (TypeError, ValueError):
                pass
        return None

    cache[sym] = {"price": fresh, "mono": mono, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _write_cache(path, cache)
    return fresh
