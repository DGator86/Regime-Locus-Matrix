"""Plotly chart builders for the Control Center."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rlm.options.surface import fit_svi_for_expiry, svi_raw


def _ensure_option_chain_dte(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame contains a numeric `dte` column representing calendar days to expiry.
    
    If `dte` already exists it will be coerced to numeric. If `dte` is missing and `expiry` is absent, `dte` is set to 30.0 for all rows. If `expiry` is present, `dte` is computed as the number of days between the normalized expiry date and a reference date (the median of parsed `timestamp` values when available, otherwise the current UTC date normalized to midnight). Missing or non-finite results are filled with 30.0 and final values are clipped to be at least 1.0.
    
    Parameters:
        df (pd.DataFrame): Input option chain which may contain `dte`, `expiry`, and/or `timestamp` columns.
    
    Returns:
        pd.DataFrame: A copy of `df` with a numeric `dte` column (float) guaranteed to be finite and >= 1.0.
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
    """
    Create a two-row Plotly figure showing the price locus: close price with optional uncertainty bands and an optional sigma bar subplot.
    
    Parameters:
        processed (pd.DataFrame): Processed market data indexed by timestamp or by integer position. Must include a "close" column; may optionally include "upper_1s", "lower_1s", "upper_2s", "lower_2s", and "sigma" to render additional traces.
        title (str): Chart title.
    
    Returns:
        go.Figure: A Plotly figure containing a larger top subplot with the close price line and any available 1σ/2σ boundary lines, and a bottom subplot showing sigma as a bar series when the "sigma" column is present.
    """
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
    """
    Create a filled polar radar chart of four factors (S_D, S_V, S_L, S_G) from the provided latest values.
    
    Parameters:
    	last (pd.Series): Series containing factor values; expected keys are "S_D", "S_V", "S_L", "S_G". Missing or falsy values are treated as 0.
    
    Returns:
    	fig (go.Figure): A Plotly Figure with a closed radar (polar) trace showing the four factor values.
    """
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
    """
    Builds a Plotly figure showing a filled quantile band and median forecast from processed forecast return columns.
    
    Expects `processed` to contain the columns `forecast_return_lower`, `forecast_return_median`, and `forecast_return_upper`. The figure contains a filled polygon between upper and lower quantiles and a median line; the x-axis uses the DataFrame's DatetimeIndex when present, otherwise a RangeIndex.
    
    Parameters:
        processed (pd.DataFrame): DataFrame containing forecast quantile columns.
    
    Returns:
        go.Figure | None: A Plotly Figure with the quantile band and median line, or `None` if any required columns are missing.
    """
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
    """
    Create a stacked-area plot of state probabilities from a sequence of HMM or Markov probability vectors.
    
    The x-axis uses the DataFrame index (datetime index if present, otherwise a range index). Prefers the `hmm_probs` column if present; falls back to `markov_probs`. Each column entry must be an iterable of numeric state probabilities of consistent length across rows.
    
    Parameters:
        processed (pd.DataFrame): DataFrame containing either `hmm_probs` or `markov_probs`, where each row value is an iterable of state probabilities.
    
    Returns:
        go.Figure | None: A Plotly figure with stacked area traces (one per state) when a supported column is present, or `None` if neither column exists.
    """
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
    """
    Create a heatmap of recent multi-timeframe ("mtf") and "confluence" numeric features.
    
    Selects up to 32 numeric columns whose names contain "mtf" or "confluence" (case-insensitive), uses up to the last 200 rows, and renders a Plotly heatmap with features on the y-axis and recent bars on the x-axis.
    
    Parameters:
        processed (pd.DataFrame): DataFrame with feature columns; numeric columns whose names include "mtf" or "confluence" are considered.
    
    Returns:
        go.Figure | None: A Plotly Figure containing the heatmap, or `None` if no matching numeric columns are found.
    """
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
    """
    Plot implied volatility versus strike for up to six expiries using the latest timestamp slice.
    
    Builds a scatter of observed IV by strike for each expiry present in the most-recent timestamp of `chain` and overlays SVI-implied IV curves when a fit is available and the expiry group has sufficient points.
    
    Parameters:
        chain (pd.DataFrame): Option chain rows containing at least `iv` and `strike`; may include `expiry`, `timestamp`, and `underlying_price`.
        spot (float | None): Spot price to use for log-moneyness and SVI conversion. If omitted or invalid, the function infers spot from `underlying_price` median or `strike` median.
    
    Returns:
        go.Figure | None: A Plotly figure containing IV scatter points and optional SVI fit lines, or `None` if `chain` is empty, lacks required `iv` data, contains no rows in the latest timestamp slice, or a valid spot cannot be determined.
    """
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
    """
    Fit SVI total-variance curves for up to five expiries and return SVI-implied implied-volatility curves plotted against log-moneyness.
    
    The function operates on the latest timestamp slice if a `timestamp` column exists. If `expiry` is absent the function falls back to an IV vs log-moneyness scatter plot. If the DataFrame is empty or missing required `iv` or `strike` columns, or if a valid positive `spot` cannot be determined, the function returns `None`.
    
    Parameters:
        chain (pd.DataFrame): Option-chain snapshot containing at minimum `iv` and `strike`. May also include `expiry`, `timestamp`, and `underlying_price`.
        spot (float | None): Underlying price to use for log-moneyness. If omitted or invalid, the median of `underlying_price` (if present) or `strike` is used.
    
    Returns:
        go.Figure | None: A Plotly Figure with SVI-implied IV curves versus `log(K/S)` (or an IV scatter figure if no `expiry` column), or `None` when prerequisites are not met.
    """
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
    """
    Create a scatter plot of implied volatility versus log-moneyness.
    
    Parameters:
        work (pd.DataFrame): DataFrame containing numeric `strike` and `iv` columns.
        spot (float): Reference underlying price used to compute log-moneyness as log(strike / spot).
    
    Returns:
        go.Figure: A Plotly Figure with points at x = log(K / S) and y = implied volatility.
    """
    log_m = np.log(pd.to_numeric(work["strike"], errors="coerce") / spot)
    iv = pd.to_numeric(work["iv"], errors="coerce")
    fig = go.Figure(go.Scatter(x=log_m, y=iv, mode="markers", marker=dict(color="#00f5ff", size=6)))
    fig.update_layout(template="plotly_dark", height=400, title="IV scatter (log-moneyness)")
    return fig


def fig_equity_regime(equity: pd.DataFrame, regime_col: str | None) -> go.Figure:
    """
    Render an equity time series, optionally coloring markers by a discrete regime column.
    
    The function requires the DataFrame to contain 'timestamp' and 'equity' columns; if either is missing an empty Plotly Figure is returned. Timestamps are parsed and rows with invalid timestamps are dropped. If a valid regime column name is provided and present in the DataFrame, markers are colored by the column's categorical codes using the Viridis colorscale and a colorbar is shown; otherwise a plain line trace is drawn.
    
    Parameters:
        equity (pd.DataFrame): DataFrame containing at least 'timestamp' and 'equity' columns.
        regime_col (str | None): Optional column name whose categorical values will determine marker colors.
    
    Returns:
        go.Figure: A Plotly Figure showing the equity curve (with colored markers when a valid regime column is supplied).
    """
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
    """
    Serialize a Plotly figure to PNG image bytes.

    Parameters:
        fig (go.Figure): Plotly figure to serialize.

    Returns:
        bytes | None: PNG image bytes if serialization succeeds, `None` if serialization fails.
    """
    try:
        return fig.to_image(format="png", scale=2)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Cluster Analysis charts
# ──────────────────────────────────────────────────────────────────────────────

_CLUSTER_COLORS: list[str] = [
    "#00f5ff",  # cyan
    "#a855f7",  # purple
    "#00ff9d",  # green
    "#ff3355",  # red
    "#ffaa00",  # orange
    "#88ccff",  # sky blue
    "#ff88cc",  # pink
    "#ffff66",  # yellow
]


def _ccolor(cid: int) -> str:
    return _CLUSTER_COLORS[cid % len(_CLUSTER_COLORS)]


def fig_cluster_sankey(
    transition_df: pd.DataFrame,
    *,
    cluster_stats: pd.DataFrame | None = None,
    title: str = "Regime cluster transition flow",
) -> go.Figure:
    """Sankey diagram of cluster-to-cluster transition probabilities.

    ``transition_df`` is a square (n × n) row-normalised probability matrix
    (output of ``cluster_analysis.compute_transition_matrix``).
    Links < 2% probability are hidden to keep the chart readable.
    """
    n = len(transition_df)
    if n == 0:
        return go.Figure().update_layout(template="plotly_dark", title=title)

    def _node_label(cid: int, side: str) -> str:
        arrow = "→" if side == "src" else ""
        if cluster_stats is not None and cid in cluster_stats.index:
            pct = cluster_stats.loc[cid].get("pct_of_data", "")
            return f"C{cid} ({pct}%) {arrow}".strip()
        return f"C{cid} {arrow}".strip()

    node_labels = [_node_label(i, "src") for i in range(n)] + [
        _node_label(i, "tgt") for i in range(n)
    ]
    node_colors = [_ccolor(i) for i in range(n)] * 2

    sources: list[int] = []
    targets: list[int] = []
    values: list[float] = []
    link_colors: list[str] = []
    link_labels: list[str] = []

    prob_matrix = transition_df.values
    for i in range(n):
        for j in range(n):
            p = float(prob_matrix[i, j])
            if p < 0.02:
                continue
            sources.append(i)
            targets.append(n + j)
            values.append(round(p * 100, 2))
            c = _ccolor(i)
            r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
            link_colors.append(f"rgba({r},{g},{b},0.35)")
            link_labels.append(f"{p * 100:.1f}%")

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=20,
                    thickness=22,
                    line=dict(color="rgba(255,255,255,0.15)", width=0.5),
                    label=node_labels,
                    color=node_colors,
                    hovertemplate="<b>%{label}</b><extra></extra>",
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                    label=link_labels,
                    hovertemplate="<b>%{label}</b><br>Probability: %{value:.1f}%<extra></extra>",
                ),
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        height=420,
        title=title,
        margin=dict(l=20, r=20, t=48, b=20),
        font=dict(size=11),
    )
    return fig


def fig_cluster_radar(
    cluster_stats: pd.DataFrame,
    feature_cols: list[str],
    *,
    title: str = "Cluster regime centroids (factor means)",
) -> go.Figure | None:
    """Multi-cluster polar chart of mean factor values per cluster."""
    mean_cols = [f"mean_{f}" for f in feature_cols if f"mean_{f}" in cluster_stats.columns]
    if not mean_cols:
        return None
    theta = [c.replace("mean_", "") for c in mean_cols]
    fig = go.Figure()
    for cid in cluster_stats.index:
        row = cluster_stats.loc[cid]
        r_vals = [float(row.get(mc, 0) or 0) for mc in mean_cols]
        c = _ccolor(int(cid))
        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        fig.add_trace(
            go.Scatterpolar(
                r=r_vals + [r_vals[0]],
                theta=theta + [theta[0]],
                fill="toself",
                name=f"C{cid}",
                line=dict(color=c, width=2),
                fillcolor=f"rgba({r},{g},{b},0.18)",
                hovertemplate=f"<b>C{cid} — %{{theta}}</b><br>%{{r:.3f}}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        polar=dict(
            radialaxis=dict(visible=True, range=[-1.0, 1.0], gridcolor="rgba(255,255,255,0.15)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=420,
        title=title,
        legend=dict(orientation="h", y=-0.12),
        margin=dict(l=40, r=40, t=48, b=40),
    )
    return fig


def fig_price_with_clusters(
    df: pd.DataFrame,
    *,
    n_clusters: int = 4,
    title: str = "Price with regime cluster coloring",
) -> go.Figure:
    """Candlestick (or line) price chart with cluster-colored marker overlay.

    Bottom sub-panel shows cluster assignment as a step chart.
    ``df`` must contain ``'cluster'`` (int) and ``'close'`` columns.
    If ``'open'``, ``'high'``, ``'low'`` are also present, a full candlestick
    is rendered; otherwise a line chart is used.
    """
    has_ohlc = all(c in df.columns for c in ("open", "high", "low", "close"))
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.RangeIndex(len(df))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
    )

    if has_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=idx,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Price",
                increasing_line_color="#00ff9d",
                decreasing_line_color="#ff3355",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=idx, y=df["close"], name="Close",
                line=dict(color="#aaaaaa", width=1), showlegend=False,
            ),
            row=1,
            col=1,
        )

    if "cluster" in df.columns:
        for cid in sorted(df["cluster"].unique()):
            mask = df["cluster"] == cid
            x_vals = idx[mask] if hasattr(idx, "__getitem__") else [
                idx[i] for i, m in enumerate(mask) if m
            ]
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=df.loc[mask, "close"],
                    mode="markers",
                    marker=dict(color=_ccolor(int(cid)), size=5, opacity=0.85),
                    name=f"C{cid}",
                    legendgroup=f"C{cid}",
                ),
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=df["cluster"],
                mode="lines",
                line=dict(color="#888888", width=1, shape="hv"),
                fill="tozeroy",
                fillcolor="rgba(0,245,255,0.12)",
                name="Cluster",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title_text="Cluster",
            tickvals=list(range(n_clusters)),
            ticktext=[f"C{i}" for i in range(n_clusters)],
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        height=560,
        title=title,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=48, b=40),
    )
    return fig


def fig_signal_distribution(
    cluster_stats: pd.DataFrame,
    feature_cols: list[str],
    *,
    title: str = "Mean signal strength per cluster",
) -> go.Figure | None:
    """Grouped bar chart: mean factor value per cluster, one bar group per feature."""
    mean_cols = [f"mean_{f}" for f in feature_cols if f"mean_{f}" in cluster_stats.columns]
    if not mean_cols or cluster_stats.empty:
        return None

    cluster_ids = cluster_stats.index.tolist()
    x_labels = [f"C{cid}" for cid in cluster_ids]
    palette = ["#00f5ff", "#a855f7", "#00ff9d", "#ff3355", "#ffaa00", "#88ccff"]
    fig = go.Figure()

    for pi, mc in enumerate(mean_cols):
        fname = mc.replace("mean_", "")
        y_vals = [float(cluster_stats.loc[cid, mc]) for cid in cluster_ids]
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=y_vals,
                name=fname,
                marker_color=palette[pi % len(palette)],
                text=[f"{v:+.3f}" for v in y_vals],
                textposition="outside",
                hovertemplate=f"<b>{fname}</b><br>C%{{x}}: %{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        height=380,
        title=title,
        xaxis_title="Cluster",
        yaxis_title="Mean value",
        legend=dict(orientation="h", y=-0.22),
        margin=dict(l=40, r=20, t=48, b=60),
    )
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    return fig
