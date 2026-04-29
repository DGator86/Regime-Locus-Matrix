"""HTTP client for the [Massive REST API](https://massive.com/docs/rest/quickstart).

Authentication: ``apiKey`` query parameter. Set ``MASSIVE_API_KEY`` in the environment
or in a ``.env`` file (see ``.env.example``).

Documentation index: https://massive.com/docs/llms.txt
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://api.massive.com"
DEFAULT_TIMEOUT_S = 60.0


def load_massive_api_key(*, env_var: str = "MASSIVE_API_KEY") -> str:
    load_dotenv()
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise ValueError(
            f"Missing {env_var}. Add it to your environment or .env file " "(never commit real keys to git)."
        )
    return key


def _ensure_api_key_param(url: str, api_key: str) -> str:
    if "apiKey=" in url or "apikey=" in url.lower():
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{urlencode({'apiKey': api_key})}"


class MassiveClient:
    """Thin JSON client over documented GET endpoints."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        api_key_env_var: str = "MASSIVE_API_KEY",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self._api_key = api_key if api_key is not None else load_massive_api_key(env_var=api_key_env_var)

    def get(
        self,
        path: str,
        params: Mapping[str, str | int | float | bool | None] | None = None,
    ) -> Any:
        """GET ``path`` (e.g. ``/v3/snapshot/options/SPY``). Adds ``apiKey`` automatically."""
        if not path.startswith("/"):
            path = "/" + path
        q: dict[str, str] = {"apiKey": self._api_key}
        if params:
            for k, v in params.items():
                if v is None:
                    continue
                q[str(k)] = str(v)
        url = f"{self.base_url}{path}"
        url += "?" + urlencode(q)
        return self._request(url)

    def get_by_url(self, url: str) -> Any:
        """Follow a Massive ``next_url`` (appends ``apiKey`` if absent)."""
        full = _ensure_api_key_param(url, self._api_key)
        return self._request(full)

    def _request(self, url: str) -> Any:
        req = Request(
            url,
            method="GET",
            headers={"Accept": "application/json"},
        )
        try:
            with urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:2000]
            path_hint = urlparse(url).path or url[:80]
            raise RuntimeError(f"Massive HTTP {e.code} for {path_hint}: {body}") from e
        except URLError as e:
            raise RuntimeError(f"Massive request failed for {url[:120]}: {e}") from e

        if not raw.strip():
            return None
        return json.loads(raw)

    def option_chain_snapshot(
        self,
        underlying: str,
        **params: str | int | float | bool | None,
    ) -> Any:
        """GET ``/v3/snapshot/options/{underlying}`` (paginated; use :func:`collect_option_snapshot_pages`)."""
        u = quote(str(underlying).upper(), safe="")
        return self.get(f"/v3/snapshot/options/{u}", params)

    def stock_aggs_range(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        from_: str,
        to: str,
        **params: str | int | float | bool | None,
    ) -> Any:
        """GET ``/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}``."""
        t = quote(str(ticker).upper(), safe="")
        return self.get(f"/v2/aggs/ticker/{t}/range/{multiplier}/{timespan}/{from_}/{to}", params)

    def stock_trades(
        self,
        ticker: str,
        **params: str | int | float | bool | None,
    ) -> Any:
        """GET ``/v3/trades/{ticker}`` (paginated; see :func:`rlm.data.massive_stocks.collect_stock_trades`)."""
        t = quote(str(ticker).upper(), safe="")
        return self.get(f"/v3/trades/{t}", params)

    def stock_quotes(
        self,
        ticker: str,
        **params: str | int | float | bool | None,
    ) -> Any:
        """GET ``/v3/quotes/{ticker}`` (NBBO history; paginated)."""
        t = quote(str(ticker).upper(), safe="")
        return self.get(f"/v3/quotes/{t}", params)

    def option_contracts_reference(
        self,
        **params: str | int | float | bool | None,
    ) -> Any:
        """GET ``/v3/reference/options/contracts`` (paginated via ``next_url``)."""
        return self.get("/v3/reference/options/contracts", params)

    def option_aggs_range(
        self,
        options_ticker: str,
        multiplier: int,
        timespan: str,
        from_: str,
        to: str,
        **params: str | int | float | bool | None,
    ) -> Any:
        """GET ``/v2/aggs/ticker/{optionsTicker}/range/...`` (use ``O:`` OCC-style tickers)."""
        t = quote(str(options_ticker).upper(), safe="")
        return self.get(
            f"/v2/aggs/ticker/{t}/range/{int(multiplier)}/{timespan}/{from_}/{to}",
            params,
        )

    def option_trades(
        self,
        options_ticker: str,
        **params: str | int | float | bool | None,
    ) -> Any:
        """GET ``/v3/trades/{optionsTicker}`` (paginated)."""
        t = quote(str(options_ticker).upper(), safe="")
        return self.get(f"/v3/trades/{t}", params)

    def option_quotes(
        self,
        options_ticker: str,
        **params: str | int | float | bool | None,
    ) -> Any:
        """GET ``/v3/quotes/{optionsTicker}`` (paginated)."""
        t = quote(str(options_ticker).upper(), safe="")
        return self.get(f"/v3/quotes/{t}", params)
