"""Join option-chain aggregates and macro series onto equity bars so factors and forecasts are well-defined."""

from __future__ import annotations

import warnings
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    import polars as pl
except Exception:  # pragma: no cover - polars is optional at runtime
    pl = None

from rlm.data.bs_greeks import bs_greeks_row
from rlm.data.option_chain import normalize_option_chain
from rlm.options.surface import build_surface_feature_frame

_EXCHANGE_TZ = ZoneInfo("America/New_York")
_INTRADAY_DOWNLOAD_CHUNK_DAYS = 7
_INTRADAY_DOWNLOAD_LOOKBACK_DAYS = 30


def _oi_weight(s: pd.Series) -> pd.Series:
    oi = pd.to_numeric(s, errors="coerce")
    if oi.isna().all():
        return pd.Series(1.0, index=s.index)
    return oi.fillna(0.0).clip(lower=1.0)


def _rolling_iv_rank(iv_series: pd.Series, window: int = 252, min_periods: int = 20) -> pd.Series:
    """Percentile of today's ATM IV within the trailing ``window`` (0–1)."""

    def _pct_last(a: np.ndarray) -> float:
        if a.size < min_periods:
            return float("nan")
        last = a[-1]
        if not np.isfinite(last):
            return float("nan")
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return float("nan")
        return float((finite <= last).sum() / finite.size)

    return iv_series.rolling(window, min_periods=min_periods).apply(_pct_last, raw=True)


def _bars_are_intraday(index: pd.Index) -> bool:
    ts = pd.DatetimeIndex(pd.to_datetime(index))
    if ts.empty:
        return False
    counts = pd.Series(1, index=ts.normalize()).groupby(level=0).sum()
    return bool((counts > 1).any())


def _to_exchange_time(index: pd.Index) -> pd.DatetimeIndex:
    ts = pd.DatetimeIndex(pd.to_datetime(index))
    if ts.tz is None:
        return ts.tz_localize(_EXCHANGE_TZ)
    return ts.tz_convert(_EXCHANGE_TZ)


def _extract_close_series(df: pd.DataFrame, sym: str) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        close_col = ("Close", sym) if ("Close", sym) in df.columns else ("Adj Close", sym)
        if close_col in df.columns:
            s = df[close_col]
        else:
            s = df["Close"].iloc[:, 0] if "Close" in df.columns else df.iloc[:, -1]
    else:
        s = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    return pd.to_numeric(s, errors="coerce").dropna()


def _load_yfinance_close_series(
    sym: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str | None = None,
) -> pd.Series:
    try:
        import yfinance as yf
    except ImportError:
        return pd.Series(dtype=float)

    kwargs = {
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "progress": False,
        "auto_adjust": False,
    }
    if interval is not None:
        kwargs["interval"] = interval

    try:
        df = yf.download(sym, **kwargs)
    except Exception:
        return pd.Series(dtype=float)
    if df.empty:
        return pd.Series(dtype=float)

    s = _extract_close_series(df, sym)
    if s.empty:
        return pd.Series(dtype=float)

    ts = pd.DatetimeIndex(pd.to_datetime(s.index))
    if interval is None:
        s.index = ts.normalize()
        return s.groupby(level=0).last()

    if ts.tz is None:
        ts = ts.tz_localize(_EXCHANGE_TZ)
    else:
        ts = ts.tz_convert(_EXCHANGE_TZ)
    s.index = ts
    s = s.sort_index()
    return s[~s.index.duplicated(keep="last")]


def _load_intraday_vix_series(sym: str, bars_index: pd.Index) -> pd.Series:
    bars_ts = _to_exchange_time(bars_index)
    if bars_ts.empty:
        return pd.Series(dtype=float)

    now = pd.Timestamp.now(tz=_EXCHANGE_TZ)
    oldest_available = (now - pd.Timedelta(days=_INTRADAY_DOWNLOAD_LOOKBACK_DAYS)).normalize()
    if bars_ts.min().normalize() < oldest_available:
        warnings.warn(
            f"{sym} 1-minute Yahoo history is only available for roughly the last "
            f"{_INTRADAY_DOWNLOAD_LOOKBACK_DAYS} days; older intraday bars will remain missing "
            "unless you prepopulate `vix`/`vvix`.",
            RuntimeWarning,
            stacklevel=2,
        )
    request_start = max(
        bars_ts.min().normalize(),
        oldest_available,
    )
    request_end = min(
        bars_ts.max().normalize() + pd.Timedelta(days=1),
        now.normalize() + pd.Timedelta(days=1),
    )
    if request_start >= request_end:
        return pd.Series(dtype=float)

    parts: list[pd.Series] = []
    chunk_start = request_start
    while chunk_start < request_end:
        chunk_end = min(
            chunk_start + pd.Timedelta(days=_INTRADAY_DOWNLOAD_CHUNK_DAYS),
            request_end,
        )
        chunk = _load_yfinance_close_series(
            sym,
            start=chunk_start,
            end=chunk_end,
            interval="1m",
        )
        if not chunk.empty:
            parts.append(chunk)
        chunk_start = chunk_end

    if not parts:
        return pd.Series(dtype=float)

    source = pd.concat(parts).sort_index()
    source = source[~source.index.duplicated(keep="last")]

    out = pd.Series(np.nan, index=bars_ts, dtype=float)
    bar_days = pd.Index(bars_ts.normalize().unique()).sort_values()
    for day in bar_days:
        day_end = day + pd.Timedelta(days=1)
        day_bars = bars_ts[(bars_ts >= day) & (bars_ts < day_end)]
        day_source = source[(source.index >= day) & (source.index < day_end)]
        if day_source.empty:
            continue
        out.loc[day_bars] = day_source.reindex(day_bars, method="ffill").to_numpy()
    return out


def _gamma_fill(
    sub: pd.DataFrame,
    spot: float,
    *,
    risk_free: float = 0.052,
) -> pd.Series:
    """Return gamma per row, using chain gamma when present else BS."""
    if "gamma" in sub.columns:
        g = pd.to_numeric(sub["gamma"], errors="coerce")
    else:
        g = pd.Series(np.nan, index=sub.index)
    if g.notna().any():
        return g
    out = pd.Series(index=sub.index, dtype=float)
    for i, row in sub.iterrows():
        k = float(row["strike"])
        t = max(float(row["dte"]), 1.0) / 365.0
        iv = float(row.get("iv") or 0.0)
        if not np.isfinite(iv) or iv <= 0:
            iv = 0.2
        is_call = str(row["option_type"]).lower() == "call"
        _, gm, _, _, _ = bs_greeks_row(
            spot=spot, strike=k, time_years=t, iv=iv, risk_free=risk_free, is_call=is_call
        )
        out.loc[i] = gm
    return out


def _to_polars_if_available(df: pd.DataFrame) -> pl.DataFrame | None:
    if pl is None:
        return None
    return pl.from_pandas(df)


def enrich_bars_from_option_chain(
    bars: pd.DataFrame,
    chain: pd.DataFrame,
    *,
    underlying: str,
    contract_multiplier: int = 100,
) -> pd.DataFrame:
    """
    Add dealer-flow raw columns (``gex``, ``vanna``, …) and ``bid_ask_spread`` (ATM $ width)
    by aggregating each day's option snapshot. Aligns on calendar date of ``bars`` index.
    """
    if chain is None or chain.empty:
        return bars.copy()

    und = str(underlying).upper().strip()
    out = bars.copy()
    ch = normalize_option_chain(chain)
    ch = ch.loc[ch["underlying"].str.upper() == und].copy()
    if ch.empty:
        return out

    ch["_d"] = pd.to_datetime(ch["timestamp"]).dt.normalize()
    bar_dates = pd.to_datetime(out.index).normalize()
    date_index = pd.Index(bar_dates.unique()).sort_values()

    close_by_date = (
        pd.Series(
            out["close"].values,
            index=pd.to_datetime(out.index).normalize(),
        )
        .groupby(level=0)
        .last()
    )

    rows: list[dict[str, float | pd.Timestamp]] = []
    ch_pl = _to_polars_if_available(ch)

    for d in date_index:
        spot = float(close_by_date.get(d, np.nan))
        if not np.isfinite(spot) or spot <= 0:
            continue

        if ch_pl is not None:
            g = ch_pl.filter(pl.col("_d") == pl.lit(d)).to_pandas()
        else:
            g = ch.loc[ch["_d"] == d].copy()
        if g.empty:
            continue

        liquid = (g["dte"] >= 7) & (g["dte"] <= 120)
        gg = g.loc[liquid].copy() if liquid.any() else g.copy()
        if gg.empty:
            continue

        m = (gg["strike"] / spot - 1.0).abs() <= 0.06
        sub = gg.loc[m].copy() if m.any() else gg.copy()

        oi = _oi_weight(
            sub["open_interest"]
            if "open_interest" in sub.columns
            else pd.Series(1.0, index=sub.index)
        )
        iv = (
            pd.to_numeric(sub["iv"], errors="coerce")
            if "iv" in sub.columns
            else pd.Series(np.nan, index=sub.index)
        )

        calls = sub["option_type"].str.lower() == "call"
        put_iv_m = iv[~calls].median()
        call_iv_m = iv[calls].median()
        put_call_skew = (
            float(call_iv_m - put_iv_m)
            if np.isfinite(call_iv_m) and np.isfinite(put_iv_m)
            else np.nan
        )

        short_mask = (gg["dte"] >= 14) & (gg["dte"] <= 40)
        long_mask = (gg["dte"] >= 45) & (gg["dte"] <= 120)
        iv_s = (
            pd.to_numeric(gg.loc[short_mask, "iv"], errors="coerce").median()
            if "iv" in gg.columns
            else np.nan
        )
        iv_l = (
            pd.to_numeric(gg.loc[long_mask, "iv"], errors="coerce").median()
            if "iv" in gg.columns
            else np.nan
        )
        term_structure_ratio = (
            float(iv_s / iv_l)
            if np.isfinite(iv_s) and np.isfinite(iv_l) and iv_l > 1e-8
            else np.nan
        )

        gamma = _gamma_fill(sub, spot)
        sign = np.where(calls.reindex(sub.index).fillna(False), 1.0, -1.0)
        gex_net = float((gamma * oi * float(contract_multiplier) * sign).sum())
        gex = float(gex_net / max(spot * 1e3, 1.0))

        vanna = (
            pd.to_numeric(sub["vanna"], errors="coerce")
            if "vanna" in sub.columns
            else pd.Series(np.nan, index=sub.index)
        )
        if vanna.isna().all():
            vanna = pd.Series(0.0, index=sub.index)
        charm = (
            pd.to_numeric(sub["charm"], errors="coerce")
            if "charm" in sub.columns
            else pd.Series(np.nan, index=sub.index)
        )
        if charm.isna().all():
            charm = pd.Series(0.0, index=sub.index)

        w = oi
        vanna_sig = float((vanna.fillna(0) * w).sum() / max(float(w.sum()), 1.0))
        charm_sig = float((charm.fillna(0) * w).sum() / max(float(w.sum()), 1.0))

        delta = (
            pd.to_numeric(sub["delta"], errors="coerce").fillna(0.0)
            if "delta" in sub.columns
            else pd.Series(0.0, index=sub.index)
        )
        dealer_position_proxy = float(
            -(delta * oi * float(contract_multiplier)).sum() / max(spot * float(oi.sum()), 1.0)
        )

        spr = pd.to_numeric(sub["ask"], errors="coerce") - pd.to_numeric(
            sub["bid"], errors="coerce"
        )
        bid_ask_spread = float(spr.median()) if spr.notna().any() else np.nan

        if "spread_pct_mid" in sub.columns:
            spread_pct_mid = pd.to_numeric(sub["spread_pct_mid"], errors="coerce")
            options_spread_pct_mid = (
                float(spread_pct_mid.median()) if spread_pct_mid.notna().any() else np.nan
            )
        else:
            options_spread_pct_mid = np.nan

        options_volume = (
            float(pd.to_numeric(gg["volume"], errors="coerce").sum(min_count=1))
            if "volume" in gg.columns
            else np.nan
        )
        total_open_interest = (
            float(pd.to_numeric(gg["open_interest"], errors="coerce").sum(min_count=1))
            if "open_interest" in gg.columns
            else np.nan
        )
        if (
            np.isfinite(options_volume)
            and np.isfinite(total_open_interest)
            and total_open_interest > 0
        ):
            options_volume_to_oi = float(options_volume / total_open_interest)
        else:
            options_volume_to_oi = np.nan

        atm_iv = float(iv.median()) if iv.notna().any() else np.nan

        rows.append(
            {
                "_d": d,
                "gex": gex,
                "vanna": vanna_sig,
                "charm": charm_sig,
                "put_call_skew": put_call_skew,
                "term_structure_ratio": term_structure_ratio,
                "dealer_position_proxy": dealer_position_proxy,
                "bid_ask_spread": bid_ask_spread,
                "options_spread_pct_mid": options_spread_pct_mid,
                "options_volume": options_volume,
                "options_volume_to_oi": options_volume_to_oi,
                "_atm_iv": atm_iv,
            }
        )

    if not rows:
        return out

    feat = pd.DataFrame(rows).set_index("_d")
    feat["iv_rank"] = _rolling_iv_rank(feat["_atm_iv"])
    feat = feat.drop(columns=["_atm_iv"])

    out = out.copy()
    out["_d"] = bar_dates
    for col in feat.columns:
        out[col] = out["_d"].map(feat[col])
    out = out.drop(columns=["_d"])

    neutral = {
        "gex": 0.0,
        "vanna": 0.0,
        "charm": 0.0,
        "put_call_skew": 0.0,
        "dealer_position_proxy": 0.0,
        "iv_rank": 0.5,
        "term_structure_ratio": 1.0,
    }
    for k, v in neutral.items():
        if k in out.columns:
            out[k] = out[k].fillna(v)

    if "bid_ask_spread" in out.columns:
        hl = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
        approx = (hl * out["close"] * 0.05).rolling(20, min_periods=5).median()
        out["bid_ask_spread"] = out["bid_ask_spread"].fillna(approx).fillna(0.05)

    return out


def enrich_bars_with_surface_features(
    bars: pd.DataFrame,
    chain: pd.DataFrame,
    *,
    underlying: str,
) -> pd.DataFrame:
    """
    Add SVI/local-vol surface-derived features by calendar date.
    """
    if chain is None or chain.empty:
        return bars.copy()

    und = str(underlying).upper().strip()
    out = bars.copy()
    ch = normalize_option_chain(chain)
    ch = ch.loc[ch["underlying"].str.upper() == und].copy()
    if (
        ch.empty
        or "iv" not in ch.columns
        or pd.to_numeric(ch["iv"], errors="coerce").notna().sum() == 0
    ):
        return out

    surface = build_surface_feature_frame(ch)
    if surface.empty:
        return out

    out = out.copy()
    out["_d"] = pd.to_datetime(out.index).normalize()
    for col in surface.columns:
        out[col] = out["_d"].map(surface[col])
    out = out.drop(columns=["_d"])

    neutral = {
        "surface_atm_forward_iv": np.nan,
        "surface_skew": 0.0,
        "surface_convexity": 0.0,
        "surface_term_slope": 0.0,
    }
    for k, v in neutral.items():
        if k in out.columns:
            out[k] = out[k].fillna(v)

    return out


def enrich_bars_with_vix(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Attach ^VIX and ^VVIX when available.

    Daily bars receive same-date closes. Intraday bars use the last known 1-minute
    reading at or before each bar timestamp, without carrying stale values across days.
    """
    out = bars.copy()
    need_vix = "vix" not in out.columns or out["vix"].isna().all()
    need_vvix = "vvix" not in out.columns or out["vvix"].isna().all()
    if not need_vix and not need_vvix:
        return out

    bars_index = pd.DatetimeIndex(pd.to_datetime(out.index))
    is_intraday = _bars_are_intraday(bars_index)

    if is_intraday:
        if need_vix:
            vx = _load_intraday_vix_series("^VIX", bars_index)
            if not vx.empty:
                out["vix"] = vx.to_numpy()
        if need_vvix:
            vv = _load_intraday_vix_series("^VVIX", bars_index)
            if not vv.empty:
                out["vvix"] = vv.to_numpy()
        return out

    start = pd.Timestamp(out.index.min()).normalize() - pd.Timedelta(days=7)
    end = pd.Timestamp(out.index.max()).normalize() + pd.Timedelta(days=2)
    idx = pd.DatetimeIndex(pd.to_datetime(out.index)).normalize()

    if need_vix:
        vx = _load_yfinance_close_series("^VIX", start=start, end=end)
        if not vx.empty:
            out["vix"] = pd.Series(idx).map(vx).values
    if need_vvix:
        vv = _load_yfinance_close_series("^VVIX", start=start, end=end)
        if not vv.empty:
            out["vvix"] = pd.Series(idx).map(vv).values
        elif "vix" in out.columns:
            out["vvix"] = pd.to_numeric(out.get("vvix"), errors="coerce").fillna(out["vix"] * 5.0)

    return out


def prepare_bars_for_factors(
    bars: pd.DataFrame,
    option_chain: pd.DataFrame | None,
    *,
    underlying: str,
    attach_vix: bool = True,
) -> pd.DataFrame:
    """
    Full pre-factor pass: chain-based dealer + liquidity fields, then optional VIX/VVIX.
    """
    b = bars.sort_index().copy()
    if option_chain is not None and not option_chain.empty:
        b = enrich_bars_from_option_chain(b, option_chain, underlying=underlying)
        b = enrich_bars_with_surface_features(b, option_chain, underlying=underlying)
    if attach_vix:
        b = enrich_bars_with_vix(b)
    return b
