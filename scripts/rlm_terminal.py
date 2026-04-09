#!/usr/bin/env python3
"""
RLM Streamlit terminal — live-ish dashboard over the **real** stack:
bars → factors → state matrix → forecast bands → ROEE (:func:`select_trade_for_row`).

Install::

    pip install -e ".[ibkr,ui]"

Run (use ``python -m`` on Windows if ``streamlit`` is not on ``PATH``)::

    python -m streamlit run scripts/rlm_terminal.py

IBKR mode needs TWS/Gateway and ``IBKR_*`` in ``.env``. CSV mode reads ``data/raw/bars_{SYMBOL}.csv``.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from rlm.data.ibkr_stocks import fetch_historical_stock_bars
from rlm.datasets.bars_enrichment import prepare_bars_for_factors
from rlm.datasets.paths import rel_bars_csv
from rlm.factors.pipeline import FactorPipeline
from rlm.forecasting.pipeline import ForecastPipeline, HybridForecastPipeline
from rlm.roee.decision import select_trade_for_row
from rlm.roee.strategy_map import get_strategy_for_regime
from rlm.scoring.state_matrix import classify_state_matrix

HMM_CONFIDENCE_DEFAULT = 0.6
DIRECTIONS = ("bull", "bear", "range", "transition")
VOLS = ("low_vol", "high_vol", "transition")


def _load_bars_ibkr(symbol: str, duration: str, bar_size: str) -> pd.DataFrame:
    bars = fetch_historical_stock_bars(
        symbol,
        duration=duration,
        bar_size=bar_size,
        timeout_sec=120.0,
    )
    if bars.empty:
        return bars
    return bars.sort_values("timestamp").set_index("timestamp")


def _load_bars_csv(symbol: str, csv_override: str | None) -> pd.DataFrame:
    path = Path(csv_override.strip()) if csv_override else ROOT / rel_bars_csv(symbol)
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df.sort_values("timestamp").set_index("timestamp")


def run_rlm_pipeline(
    bars: pd.DataFrame,
    *,
    symbol: str,
    use_hmm: bool,
    move_window: int,
    vol_window: int,
    attach_vix: bool,
) -> tuple[pd.DataFrame | None, str | None]:
    if bars.empty or "close" not in bars.columns:
        return None, "No bars or missing 'close' column."

    try:
        df = prepare_bars_for_factors(
            bars.copy(),
            option_chain=None,
            underlying=symbol.upper(),
            attach_vix=attach_vix,
        )
        feats = FactorPipeline().run(df)
        feats = classify_state_matrix(feats)

        if use_hmm:
            pipe = HybridForecastPipeline(move_window=move_window, vol_window=vol_window)
            out = pipe.run(feats)
        else:
            pipe = ForecastPipeline(move_window=move_window, vol_window=vol_window)
            out = pipe.run(feats)

        out = out.copy()
        out["has_major_event"] = False
        return out, None
    except Exception as e:
        return None, str(e)


def _regime_heatmap_matrix(
    liquidity: str,
    dealer_flow: str,
    *,
    short_dte: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    z = np.zeros((len(DIRECTIONS), len(VOLS)))
    labels = np.empty(z.shape, dtype=object)
    for i, d in enumerate(DIRECTIONS):
        for j, v in enumerate(VOLS):
            c = get_strategy_for_regime(
                d, v, liquidity, dealer_flow, short_dte=short_dte
            )
            z[i, j] = float(c.max_risk_pct) * 100.0
            labels[i, j] = c.strategy_name.replace("_", " ")
    return z, labels, list(DIRECTIONS), list(VOLS)


def _equity_series(symbol: str) -> pd.Series | None:
    from rlm.datasets.paths import backtest_equity_filename

    p = ROOT / "data" / "processed" / backtest_equity_filename(symbol)
    if not p.is_file():
        return None
    try:
        eq = pd.read_csv(p)
        if "timestamp" not in eq.columns or "equity" not in eq.columns:
            return None
        eq["timestamp"] = pd.to_datetime(eq["timestamp"], errors="coerce")
        return eq.dropna(subset=["timestamp"]).set_index("timestamp")["equity"]
    except (OSError, ValueError):
        return None


def main() -> None:
    st.set_page_config(
        page_title="RLM • Regime Locus Matrix",
        layout="wide",
        page_icon="📈",
    )

    st.title("RLM Regime Locus Matrix Terminal")
    st.caption("Factors • state matrix • locus bands • ROEE — wired to the repo pipeline")

    with st.sidebar:
        st.header("Controls")
        symbol = st.text_input("Symbol", value="SPY", max_chars=10).upper().strip() or "SPY"
        mode = st.radio("Data source", ["IBKR", "CSV"], horizontal=True)
        duration = st.text_input("IBKR duration", value="120 D")
        bar_size = st.selectbox("IBKR bar size", ["1 day", "5 mins", "15 mins", "1 hour"])
        csv_path = st.text_input("CSV path (CSV mode)", value=str(ROOT / rel_bars_csv("SPY")))
        move_window = st.number_input("Forecast move window", min_value=20, max_value=500, value=100)
        vol_window = st.number_input("Forecast vol window", min_value=20, max_value=500, value=100)
        use_hmm = st.checkbox("HMM overlay + gate (HybridForecastPipeline)", value=False)
        attach_vix = st.checkbox("Attach VIX (slower; needs data)", value=True)
        short_dte = st.checkbox("ROEE short-DTE map (0DTE/1DTE)", value=False)
        strike_inc = st.number_input("Strike increment", min_value=0.5, max_value=25.0, value=1.0)
        refresh_rate = st.slider("Auto-refresh (seconds, 0=off)", 0, 120, 0)
        st.divider()
        st.caption("Combo orders: use `scripts/ibkr_place_roee_combo.py` or `run_master.py`.")
        if st.button("Refresh now"):
            st.rerun()

    def render_body() -> None:
        err: str | None = None
        processed: pd.DataFrame | None = None

        if mode == "IBKR":
            try:
                bars = _load_bars_ibkr(symbol, duration=duration.strip(), bar_size=bar_size)
            except Exception as e:
                err = f"IBKR fetch failed: {e}"
                bars = pd.DataFrame()
        else:
            bars = _load_bars_csv(symbol, csv_path)

        if err is None:
            processed, perr = run_rlm_pipeline(
                bars,
                symbol=symbol,
                use_hmm=use_hmm,
                move_window=int(move_window),
                vol_window=int(vol_window),
                attach_vix=attach_vix,
            )
            if perr:
                err = perr

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Live dashboard", "Regime matrix", "ROEE", "Positions", "Performance"]
        )

        with tab1:
            if err or processed is None or processed.empty:
                st.error(err or "No processed data.")
                return

            col1, col2 = st.columns([3, 1])
            idx = processed.index
            if not isinstance(idx, pd.DatetimeIndex):
                idx = pd.RangeIndex(len(processed))

            with col1:
                st.subheader(f"{symbol} — price and locus bands")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28])
                fig.add_trace(
                    go.Scatter(
                        x=idx,
                        y=processed["close"],
                        name="Close",
                        line=dict(color="#00ff88", width=1.2),
                    ),
                    row=1,
                    col=1,
                )
                for name, col, dash, color in [
                    ("1σ upper", "upper_1s", "dot", "#ffaa00"),
                    ("1σ lower", "lower_1s", "dot", "#ffaa00"),
                    ("2σ upper", "upper_2s", "dash", "#ff0066"),
                    ("2σ lower", "lower_2s", "dash", "#ff0066"),
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
                    margin=dict(l=40, r=20, t=40, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Latest regime")
                last = processed.iloc[-1]
                st.metric("Close", f"{float(last['close']):,.2f}")
                st.metric("σ (return scale)", f"{float(last.get('sigma', 0) or 0):.4f}")
                st.metric("Direction", str(last.get("direction_regime", "")))
                st.metric("Volatility", str(last.get("volatility_regime", "")))
                st.metric("Liquidity", str(last.get("liquidity_regime", "")))
                st.metric("Dealer flow", str(last.get("dealer_flow_regime", "")))
                st.text_area("Regime key", str(last.get("regime_key", "")), height=68)
                if use_hmm and "hmm_state_label" in processed.columns:
                    st.metric("HMM state", str(last.get("hmm_state_label", "")))
                    probs = last.get("hmm_probs")
                    if isinstance(probs, (list, tuple)) and probs:
                        st.caption(f"HMM max prob: {max(float(p) for p in probs):.0%}")

        with tab2:
            if processed is None or processed.empty:
                st.warning("Run pipeline first (tab 1).")
            else:
                last = processed.iloc[-1]
                liq = str(last.get("liquidity_regime", "high_liquidity"))
                gf = str(last.get("dealer_flow_regime", "supportive"))
                st.subheader("Strategy map slice (direction × volatility)")
                st.caption(
                    f"Liquidity = `{liq}`, dealer flow = `{gf}` (from latest bar). "
                    "Change symbol/data to shift scores."
                )
                z, labels, yl, xl = _regime_heatmap_matrix(liq, gf, short_dte=short_dte)
                hm = go.Figure(
                    data=go.Heatmap(
                        z=z,
                        x=xl,
                        y=yl,
                        text=labels,
                        texttemplate="%{text}<br>%{z:.2f}% max risk",
                        hovertemplate=(
                            "<b>%{y} × %{x}</b><br>%{text}<br>max risk: %{z:.2f}%<extra></extra>"
                        ),
                        colorscale="Viridis",
                        hoverongaps=False,
                    )
                )
                hm.update_layout(
                    template="plotly_dark",
                    height=420,
                    title="Strategy map (max risk % of capital)",
                )
                st.plotly_chart(hm, use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    pick_d = st.selectbox("Direction", DIRECTIONS)
                with c2:
                    pick_v = st.selectbox("Volatility", VOLS)
                cand = get_strategy_for_regime(
                    pick_d, pick_v, liq, gf, short_dte=short_dte
                )
                st.markdown(f"**{cand.strategy_name}** — {cand.rationale}")
                st.json(
                    {
                        "target_dte": [cand.target_dte_min, cand.target_dte_max],
                        "target_profit_pct": cand.target_profit_pct,
                        "max_risk_pct": cand.max_risk_pct,
                        "defined_risk": cand.defined_risk,
                    }
                )

                show_cols = [
                    c
                    for c in (
                        "direction_regime",
                        "volatility_regime",
                        "liquidity_regime",
                        "dealer_flow_regime",
                        "S_D",
                        "S_V",
                        "S_L",
                        "S_G",
                    )
                    if c in processed.columns
                ]
                st.dataframe(processed[show_cols].tail(24), use_container_width=True)

        with tab3:
            if processed is None or processed.empty:
                st.warning("No data.")
            else:
                last = processed.iloc[-1]
                hmm_thr = HMM_CONFIDENCE_DEFAULT if use_hmm else None
                decision = select_trade_for_row(
                    last,
                    strike_increment=float(strike_inc),
                    hmm_confidence_threshold=hmm_thr,
                    short_dte=short_dte,
                )
                st.subheader(f"Action: **{decision.action.upper()}**")
                if decision.strategy_name:
                    st.write("Strategy:", decision.strategy_name)
                st.write("Rationale:", decision.rationale or "—")
                if decision.size_fraction is not None:
                    st.metric("Size fraction", f"{float(decision.size_fraction):.1%}")
                if decision.legs:
                    st.write("Legs:")
                    for leg in decision.legs:
                        st.write(
                            f"- {leg.side.upper()} {leg.quantity}× "
                            f"{leg.option_type.upper()} @ {leg.strike}"
                            + (f" exp {leg.expiry}" if leg.expiry else "")
                        )
                if decision.metadata:
                    with st.expander("Metadata"):
                        st.json(decision.metadata)

        with tab4:
            st.subheader("Positions")
            want_ibkr = st.checkbox("Fetch IBKR snapshot (read-only)", value=False)
            if want_ibkr:
                try:
                    from rlm.data.ibkr_snapshot import fetch_ibkr_account_snapshot

                    snap = fetch_ibkr_account_snapshot(timeout_sec=20.0)
                    rows = [
                        {
                            "account": r.account,
                            "contract": r.local_symbol or r.symbol,
                            "type": r.sec_type,
                            "qty": r.position,
                            "avg_cost": r.avg_cost,
                        }
                        for r in snap.positions
                    ]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    st.caption(f"{snap.host}:{snap.port} clientId={snap.client_id}")
                except ImportError as e:
                    st.warning(str(e))
                except Exception as e:
                    st.error(str(e))
            else:
                st.info("Enable the checkbox to call `fetch_ibkr_account_snapshot` (requires ibapi + TWS).")

        with tab5:
            st.subheader("Backtest equity (processed)")
            ser = _equity_series(symbol)
            if ser is None or ser.empty:
                st.info(f"No `data/processed/{symbol}` equity CSV found for this symbol.")
            else:
                fig = go.Figure(
                    go.Scatter(x=ser.index, y=ser.values, mode="lines", name="Equity", line=dict(color="#00ccff"))
                )
                fig.update_layout(template="plotly_dark", height=400, title=f"Equity — {symbol}")
                st.plotly_chart(fig, use_container_width=True)

    frag = getattr(st, "fragment", None)
    if refresh_rate >= 5 and frag is not None:
        @frag(run_every=timedelta(seconds=int(refresh_rate)))
        def _wrapped() -> None:
            render_body()

        _wrapped()
    else:
        if refresh_rate > 0 and refresh_rate < 5:
            st.sidebar.warning("Auto-refresh requires ≥ 5 s (Streamlit fragment).")
        render_body()


if __name__ == "__main__":
    main()
