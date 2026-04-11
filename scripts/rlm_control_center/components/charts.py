"""Plotly chart builders for the Control Center."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rlm.options.surface import fit_svi_for_expiry, svi_raw


def _ensure_option_chain_dte(df: pd.DataFrame) -> pd.DataFrame:
    """
    ``fit_svi_for_expiry`` requires a numeric ``dte`` column (calendar days).
    Many raw chain CSVs only have ``expiry``; derive ``dte`` when missing.
    """
    out = df.copy()
    if "dte" in out.columns:
        out["dte"] = pd.to_numeric(out["dte"], errors="coerce")
        return out
    if "expiry" not in out.columns:
        out["dte"] = 30.0
        return out
    exp = pd.to_datetime(out["expiry"], errors="coerce")
    ref = pd.NaT
    if "timestamp" in out.columns:
        ts = pd.to_datetime(out["timestamp"], errors="coerce")
        ref = ts.median()
    if pd.isna(ref):
        ref = pd.Timestamp.utcnow().normalize()
    else:
        ref = pd.Timestamp(ref).normalize()
    dte_days = (exp.dt.normalize() - ref).dt.days.astype(float)
    out["dte"] = dte_days.fillna(30.0).clip(lower=1.0)
    return out


def fig_price_locus(processed: pd.DataFrame, *, title: str) -> go.Figure:
    idx = processed.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.RangeIndex(len(processed))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28])
    fig.add_trace(
        go.Scatter(x=idx, y=processed["close"], name="Close", line=dict(color="#00ff9d", width=1.2)),
        row=1,
        col=1,
    )
    for name, col, dash, color in [
        ("1σ upper", "upper_1s", "dot", "#00f5ff"),
        ("1σ lower", "lower_1s", "dot", "#00f5ff"),
        ("2σ upper", "upper_2s", "dash", "#a855f7"),
        ("2σ lower", "lower_2s", "dash", "#a855f7"),
    ]:
        if col in processed.columns:
            fig.add_trace(
                go.Scatter(
                    x=idx,
                    y=processed[col],
                    name=name,
                    line=dict(dash=dash, color=color, width=1),
                ),
                row=1,
                col=1,
            )
    if "sigma" in processed.columns:
        fig.add_trace(
            go.Bar(x=idx, y=processed["sigma"], name="σ", marker_color="#5588ff"),
            row=2,
            col=1,
        )
    fig.update_layout(
        template="plotly_dark",
        height=640,
        margin=dict(l=40, r=20, t=48, b=40),
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def fig_factor_radar(last: pd.Series) -> go.Figure:
    cats = ["S_D", "S_V", "S_L", "S_G"]
    vals = [float(last.get(c, 0) or 0) for c in cats]
    fig = go.Figure(
        data=go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="toself",
            line_color="#00f5ff",
            fillcolor="rgba(0,245,255,0.25)",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        polar=dict(radialaxis=dict(visible=True, range=[-1.0, 1.0])),
        height=400,
        title="Factor radar (latest bar)",
    )
    return fig


def fig_quantile_fan(processed: pd.DataFrame) -> go.Figure | None:
    cols = ["forecast_return_lower", "forecast_return_median", "forecast_return_upper"]
    if not all(c in processed.columns for c in cols):
        return None
    idx = processed.index if isinstance(processed.index, pd.DatetimeIndex) else pd.RangeIndex(len(processed))
    fig = go.Figure()
    x = list(idx)
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],
            y=list(processed["forecast_return_upper"]) + list(processed["forecast_return_lower"][::-1]),
            fill="toself",
            fillcolor="rgba(168,85,247,0.25)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Quantile band",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=processed["forecast_return_median"],
            name="Median",
            line=dict(color="#00f5ff", width=2),
        )
    )
    fig.update_layout(template="plotly_dark", height=360, title="Probabilistic return fan")
    return fig


def fig_hmm_markov_stacked(processed: pd.DataFrame) -> go.Figure | None:
    idx = processed.index if isinstance(processed.index, pd.DatetimeIndex) else pd.RangeIndex(len(processed))
    x = list(idx)
    if "hmm_probs" in processed.columns:
        mat = np.stack([np.asarray(v, dtype=float) for v in processed["hmm_probs"]], axis=0)
        n_states = mat.shape[1]
        fig = go.Figure()
        for i in range(n_states):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mat[:, i],
                    stackgroup="hmm",
                    name=f"HMM {i}",
                    mode="lines",
                    line=dict(width=0.6),
                )
            )
        fig.update_layout(template="plotly_dark", height=360, title="HMM state probabilities")
        return fig
    if "markov_probs" in processed.columns:
        mat = np.stack([np.asarray(v, dtype=float) for v in processed["markov_probs"]], axis=0)
        n_states = mat.shape[1]
        fig = go.Figure()
        for i in range(n_states):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mat[:, i],
                    stackgroup="mk",
                    name=f"Markov {i}",
                    mode="lines",
                    line=dict(width=0.6),
                )
            )
        fig.update_layout(template="plotly_dark", height=360, title="Markov state probabilities")
        return fig
    return None


def fig_mtf_alignment_heatmap(processed: pd.DataFrame) -> go.Figure | None:
    cols = [
        c
        for c in processed.columns
        if ("mtf" in c.lower() or "confluence" in c.lower()) and pd.api.types.is_numeric_dtype(processed[c])
    ]
    if not cols:
        return None
    cols = cols[:32]
    tail = processed[cols].tail(min(200, len(processed))).T
    fig = go.Figure(
        data=go.Heatmap(
            z=tail.values,
            x=[str(i) for i in tail.columns],
            y=tail.index.astype(str),
            colorscale="Viridis",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=max(320, 12 * len(cols)),
        title="MTF / confluence features (recent bars)",
    )
    return fig


def fig_svi_surface_latest(
    chain: pd.DataFrame,
    *,
    spot: float | None = None,
) -> go.Figure | None:
    """IV by strike for a few expiries (latest timestamp slice)."""
    if chain.empty or "iv" not in chain.columns:
        return None
    work = chain.copy()
    if "timestamp" in work.columns:
        work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
        last_ts = work["timestamp"].max()
        work = work[work["timestamp"] == last_ts]
    if work.empty:
        return None
    if spot is None or not np.isfinite(spot) or spot <= 0:
        if "underlying_price" in work.columns:
            spot = float(pd.to_numeric(work["underlying_price"], errors="coerce").median())
        else:
            spot = float(pd.to_numeric(work["strike"], errors="coerce").median())
    if not np.isfinite(spot) or spot <= 0:
        return None

    work = _ensure_option_chain_dte(work)

    fig = go.Figure()
    expiries: Sequence[Any]
    if "expiry" in work.columns:
        expiries = sorted(work["expiry"].unique())[:6]
    else:
        expiries = [None]
    palette = ["#00f5ff", "#a855f7", "#00ff9d", "#ff3355", "#ffaa00", "#88ccff"]
    for i, exp in enumerate(expiries):
        sl = work if exp is None else work[work["expiry"] == exp]
        if sl.empty or "strike" not in sl.columns:
            continue
        label = str(exp) if exp is not None else "chain"
        fig.add_trace(
            go.Scatter(
                x=sl["strike"],
                y=sl["iv"],
                mode="markers",
                name=label[:24],
                marker=dict(size=5, color=palette[i % len(palette)]),
            )
        )
        if len(sl) >= 5:
            params = fit_svi_for_expiry(sl, spot=spot)
            if params is not None:
                ks = np.linspace(float(sl["strike"].min()), float(sl["strike"].max()), 80)
                log_m = np.log(ks / float(spot))
                w = svi_raw(log_m, params.a, params.b, params.rho, params.m, params.sigma)
                iv_curve = np.sqrt(np.maximum(w, 0.0) / max(params.tau_years, 1e-9))
                fig.add_trace(
                    go.Scatter(
                        x=ks,
                        y=iv_curve,
                        mode="lines",
                        name=f"SVI {label[:12]}",
                        line=dict(color=palette[i % len(palette)], width=1.5),
                    )
                )
    fig.update_layout(template="plotly_dark", height=420, title="IV vs strike + SVI fit (latest snapshot)")
    fig.update_xaxes(title="Strike")
    fig.update_yaxes(title="IV")
    return fig


def fig_svi_fitted_curves(chain: pd.DataFrame, *, spot: float | None = None) -> go.Figure | None:
    """Per-expiry SVI total-variance fit vs log-moneyness."""
    if chain.empty or "iv" not in chain.columns or "strike" not in chain.columns:
        return None
    work = chain.copy()
    if "timestamp" in work.columns:
        work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
        work = work[work["timestamp"] == work["timestamp"].max()]
    if spot is None or not np.isfinite(spot) or spot <= 0:
        if "underlying_price" in work.columns:
            spot = float(pd.to_numeric(work["underlying_price"], errors="coerce").median())
        else:
            spot = float(pd.to_numeric(work["strike"], errors="coerce").median())
    if not np.isfinite(spot) or spot <= 0:
        return None

    work = _ensure_option_chain_dte(work)

    fig = go.Figure()
    if "expiry" not in work.columns:
        return fig_iv_scatter_only(work, spot=spot)

    for exp, grp in list(work.groupby("expiry", sort=True))[:5]:
        p = fit_svi_for_expiry(grp, spot=spot)
        if p is None:
            continue
        km = np.linspace(-0.35, 0.35, 60)
        w = svi_raw(km, p.a, p.b, p.rho, p.m, p.sigma)
        iv_curve = np.sqrt(np.maximum(w, 0.0) / max(p.tau_years, 1e-9))
        fig.add_trace(
            go.Scatter(x=km, y=iv_curve, mode="lines", name=f"SVI {str(exp)[:10]} (dte={p.dte:.0f})")
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        title="SVI-implied ATM IV curve vs log-moneyness",
        xaxis_title="log(K/S)",
        yaxis_title="Implied vol",
    )
    return fig


def fig_iv_scatter_only(work: pd.DataFrame, *, spot: float) -> go.Figure:
    log_m = np.log(pd.to_numeric(work["strike"], errors="coerce") / spot)
    iv = pd.to_numeric(work["iv"], errors="coerce")
    fig = go.Figure(go.Scatter(x=log_m, y=iv, mode="markers", marker=dict(color="#00f5ff", size=6)))
    fig.update_layout(template="plotly_dark", height=400, title="IV scatter (log-moneyness)")
    return fig


def fig_equity_regime(equity: pd.DataFrame, regime_col: str | None) -> go.Figure:
    if "timestamp" not in equity.columns or "equity" not in equity.columns:
        return go.Figure()
    eq = equity.copy()
    eq["timestamp"] = pd.to_datetime(eq["timestamp"], errors="coerce")
    eq = eq.dropna(subset=["timestamp"])
    color = "#00f5ff"
    if regime_col and regime_col in eq.columns:
        codes = pd.Categorical(eq[regime_col]).codes
        fig = go.Figure(
            go.Scatter(
                x=eq["timestamp"],
                y=eq["equity"],
                mode="lines+markers",
                marker=dict(size=3, color=codes, colorscale="Viridis", showscale=True),
                line=dict(color=color, width=1),
            )
        )
    else:
        fig = go.Figure(
            go.Scatter(x=eq["timestamp"], y=eq["equity"], mode="lines", line=dict(color=color, width=1.2))
        )
    fig.update_layout(template="plotly_dark", height=420, title="Equity curve")
    return fig


def write_figure_png_bytes(fig: go.Figure) -> bytes | None:
    try:
        return fig.to_image(format="png", scale=2)
    except Exception:
        return None
