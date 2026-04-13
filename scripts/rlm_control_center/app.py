#!/usr/bin/env python3
"""
RLM Control Center v1.0 — flagship Streamlit dashboard.

Run from repo root::

    python -m streamlit run scripts/rlm_control_center/app.py

Requires: ``pip install -e ".[ui]"`` (optional ``ibkr`` for live bars).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from components import charts, modals, tables  # noqa: E402
from components.utils import (  # noqa: E402
    append_pipeline_log,
    clear_file_caches,
    enrich_universe_with_feature_tail,
    env_connected_ibkr,
    env_connected_massive,
    flatten_universe_result,
    load_dotenv_if_present,
    run_repo_script,
    safe_load_json,
    safe_read_csv,
    universe_health_score,
    utc_now_str,
)
from custom_css import inject_custom_css  # noqa: E402
from demo_data import get_demo_bars, get_demo_option_chain_stub  # noqa: E402

load_dotenv_if_present(ROOT)

try:
    from streamlit_option_menu import option_menu  # noqa: E402
except ImportError:
    option_menu = None  # type: ignore[misc, assignment]

from rlm.data.ibkr_stocks import fetch_historical_stock_bars  # noqa: E402
from rlm.datasets.paths import (  # noqa: E402
    backtest_equity_filename,
    rel_bars_csv,
    rel_forecast_features_csv,
    rel_option_chain_csv,
    walkforward_summary_filename,
)
from rlm.forecasting.live_model import LiveRegimeModelConfig  # noqa: E402
from rlm.roee.decision import select_trade_for_row  # noqa: E402
from rlm.roee.policy import select_trade  # noqa: E402
from rlm.roee.strategy_map import get_strategy_for_regime  # noqa: E402
from rlm.ui.pipeline_runner import ForecastMode, run_feature_forecast_stack  # noqa: E402

DIRECTIONS = ("bull", "bear", "range", "transition")
VOLS = ("low_vol", "high_vol", "transition")
DEFAULT_PLANS = ROOT / "data" / "processed" / "universe_trade_plans.json"


def _load_bars_ibkr(symbol: str, duration: str, bar_size: str) -> pd.DataFrame:
    """
    Load historical price bars for a symbol from the IBKR source.
    
    Parameters:
        symbol (str): Ticker symbol to fetch.
        duration (str): IBKR-style duration string (e.g., "1 D", "1 M").
        bar_size (str): IBKR bar size string (e.g., "1 min", "5 mins").
    
    Returns:
        pd.DataFrame: Empty DataFrame if no bars were returned. If non-empty, rows are sorted by the 'timestamp' column and 'timestamp' is set as the DataFrame index.
    """
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
    """
    Load OHLCV bars for a symbol from a CSV file and return them indexed by timestamp.
    
    Reads the CSV at `csv_override` (if provided) or at the default path for `symbol`; requires a 'timestamp' column which is parsed to datetimes. Rows with invalid timestamps are dropped, and the resulting DataFrame is sorted by timestamp and returned with the timestamp set as the index. If the file is missing or lacks a usable 'timestamp' column, an empty DataFrame is returned.
    
    Parameters:
        symbol (str): Ticker symbol used to derive the default CSV path when `csv_override` is not provided.
        csv_override (str | None): Optional filesystem path to a CSV file to load; if empty or None the default path for `symbol` is used.
    
    Returns:
        pd.DataFrame: DataFrame of bars indexed by timezone-naive timestamps (sorted). Returns an empty DataFrame if the file does not exist or does not contain a valid 'timestamp' column.
    """
    path = Path(csv_override.strip()) if csv_override else ROOT / rel_bars_csv(symbol)
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df.sort_values("timestamp").set_index("timestamp")


def _regime_heatmap_matrix(
    liquidity: str,
    dealer_flow: str,
    *,
    short_dte: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Builds a strategy heatmap and corresponding labels for all regime direction/volatility combinations.
    
    Parameters:
        liquidity (str): Liquidity regime identifier used to select strategies (e.g., "low", "high").
        dealer_flow (str): Dealer flow regime identifier used to select strategies (e.g., "positive", "negative").
        short_dte (bool): If true, select strategies appropriate for short days-to-expiry.
    
    Returns:
        tuple: A 4-tuple containing:
            - z (np.ndarray): 2D array of strategy maximum risk percentages expressed as percent (rows correspond to DIRECTIONS, columns to VOLS).
            - labels (np.ndarray): 2D object array of strategy display names (underscores replaced with spaces) matching `z` shape.
            - directions (list[str]): List of direction labels in the same row order as `z`.
            - vols (list[str]): List of volatility labels in the same column order as `z`.
    """
    z = np.zeros((len(DIRECTIONS), len(VOLS)))
    labels = np.empty(z.shape, dtype=object)
    for i, d in enumerate(DIRECTIONS):
        for j, v in enumerate(VOLS):
            c = get_strategy_for_regime(d, v, liquidity, dealer_flow, short_dte=short_dte)
            z[i, j] = float(c.max_risk_pct) * 100.0
            labels[i, j] = c.strategy_name.replace("_", " ")
    return z, labels, list(DIRECTIONS), list(VOLS)


def _init_session() -> None:
    """
    Initialize default Streamlit session state keys used by the dashboard.
    
    Sets default values for keys if they are not already present in st.session_state. Keys and defaults:
    - "selected_symbol": "SPY"
    - "forecast_mode": "deterministic"
    - "bars_mode_radio": "Demo"
    - "proc_df": None
    - "proc_symbol": ""
    - "open_drilldown": False
    """
    st.session_state.setdefault("selected_symbol", "SPY")
    st.session_state.setdefault("forecast_mode", "deterministic")
    st.session_state.setdefault("bars_mode_radio", "Demo")
    st.session_state.setdefault("proc_df", None)
    st.session_state.setdefault("proc_symbol", "")
    st.session_state.setdefault("open_drilldown", False)


def main() -> None:
    # set_page_config must be the first Streamlit command (before session_state / widgets).
    """
    Render the Streamlit-based "RLM Control Center v1.0" dashboard for interactive regime analysis and pipeline control.
    
    Initializes page configuration and session defaults, injects custom styling, builds the sidebar controls (symbol, bars source, forecast stack, model parameters, and pipeline actions), loads price bars from Demo/CSV/IBKR, runs the forecast/feature processing stack (when available), stores processed results in session state, and renders seven main tabs: Universe, Forecasts, Positions, Matrix, Backtest, Pipeline, and Settings. The function updates st.session_state, may invoke external scripts (pipeline actions), and presents charts, tables, downloads, and interactive controls; it reports errors and toasts to the UI as appropriate.
    """
    st.set_page_config(page_title="RLM Control Center v1.0", layout="wide", page_icon="⚔️")
    _init_session()
    inject_custom_css()

    st.markdown(
        '<div class="header-aurora"><span class="neon-text">RLM CONTROL CENTER</span> v1.0 — '
        "<span style='color:#9aa0ae;font-weight:400'>Regime Locus Matrix</span></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Tip: keep **Bars source = Demo** until you have `data/raw/bars_{SYM}.csv` or IBKR. "
        "Use **Deterministic** forecast first; HMM/Markov need enough history and can be slow."
    )

    massive_ok = env_connected_massive()
    ibkr_ok = env_connected_ibkr()
    dot_class = "pulse-dot dot-live" if (massive_ok or ibkr_ok) else "pulse-dot dot-off"
    conn_label = "live" if (massive_ok or ibkr_ok) else "offline"

    with st.sidebar:
        st.markdown(
            f'<p><span class="{dot_class}"></span><b>RLM</b> <span style="color:#888">({conn_label})</span></p>',
            unsafe_allow_html=True,
        )
        st.caption("Massive API" + (" ✓" if massive_ok else " —"))
        st.caption("IBKR host" + (" ✓" if ibkr_ok else " —"))

        symbol = st.text_input("Symbol", value=st.session_state["selected_symbol"], max_chars=8).upper().strip() or "SPY"
        st.session_state["selected_symbol"] = symbol

        st.radio(
            "Bars source",
            ["Demo", "CSV", "IBKR"],
            horizontal=True,
            help="Demo = synthetic OHLCV (always works). CSV = `data/raw/bars_{symbol}.csv`.",
            key="bars_mode_radio",
        )
        data_mode = str(st.session_state.get("bars_mode_radio", "Demo"))
        duration = st.text_input("IBKR duration", value="120 D")
        bar_size = st.selectbox("IBKR bar size", ["1 day", "5 mins", "15 mins", "1 hour"])
        csv_path = st.text_input("Bars CSV override", value=str(ROOT / rel_bars_csv(symbol)))

        _fm_labels = ["Deterministic", "Hybrid HMM", "Hybrid Markov", "Probabilistic"]
        _fm_keys: tuple[ForecastMode, ...] = ("deterministic", "hmm", "markov", "probabilistic")
        _cur = str(st.session_state.get("forecast_mode", "deterministic"))
        try:
            _def_i = _fm_keys.index(_cur)  # type: ignore[arg-type]
        except ValueError:
            _def_i = 0
        if option_menu is not None:
            try:
                picked = option_menu(
                    "Forecast stack",
                    _fm_labels,
                    menu_icon=None,
                    default_index=_def_i,
                    orientation="vertical",
                    key="rlm_forecast_stack_menu",
                )
                forecast_mode = _fm_keys[_fm_labels.index(picked)]
                st.session_state["forecast_mode"] = forecast_mode
            except Exception:
                forecast_mode = st.selectbox(
                    "Forecast stack",
                    _fm_keys,
                    index=_def_i,
                    format_func=lambda x: {
                        "deterministic": "Forecast (deterministic)",
                        "hmm": "Hybrid HMM",
                        "markov": "Hybrid Markov",
                        "probabilistic": "Probabilistic (optional JSON model)",
                    }[x],
                    key="rlm_forecast_fallback_select",
                )
                st.session_state["forecast_mode"] = forecast_mode
        else:
            forecast_mode = st.selectbox(
                "Forecast stack",
                _fm_keys,
                index=_def_i,
                format_func=lambda x: {
                    "deterministic": "Forecast (deterministic)",
                    "hmm": "Hybrid HMM",
                    "markov": "Hybrid Markov",
                    "probabilistic": "Probabilistic (optional JSON model)",
                }[x],
                key="rlm_forecast_fallback_select_no_option_menu",
            )
            st.session_state["forecast_mode"] = forecast_mode
        prob_path = st.text_input(
            "Probabilistic model JSON (optional)",
            value="",
            help="Path to quantile linear model artifact; empty uses distribution fallback.",
        )
        move_w = st.number_input("Move window", 20, 500, 100)
        vol_w = st.number_input("Vol window", 20, 500, 100)
        attach_vix = st.checkbox("Attach VIX in enrichment", value=False)
        short_dte = st.checkbox("ROEE short-DTE map", value=False)
        strike_inc = st.number_input("Strike increment", 0.5, 25.0, 1.0)
        auto_refresh = st.checkbox("Auto-refresh 30s", value=False)
        if st.button("Refresh data", key="sidebar_refresh_data"):
            clear_file_caches()
            st.session_state["proc_df"] = None
            st.rerun()

    top1, top2, top3 = st.columns([1, 2, 1])
    with top1:
        st.metric("UTC", utc_now_str())
    with top2:
        plans_payload = safe_load_json(str(DEFAULT_PLANS))
        health, health_detail = universe_health_score(plans_payload)
        st.metric("Universe health", f"{health:.0f}%", help=health_detail)
    with top3:
        if st.button("RUN FULL PIPELINE", type="primary", use_container_width=True, key="top_run_full_pipeline"):
            py = sys.executable
            code, out, err = run_repo_script(
                ROOT,
                py,
                "scripts/run_everything.py",
                # Use --opt=value so values starting with "-" are not parsed as new flags (Windows argparse).
                ["--skip-monitor", "--pipeline-args=--no-vix"],
                timeout_s=None,
            )
            append_pipeline_log(out + ("\n" + err if err else ""))
            clear_file_caches()
            if code == 0:
                st.toast("Pipeline finished OK", icon="✅")
            else:
                st.toast(f"Exit {code}", icon="⚠️")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "🌍 UNIVERSE",
            "📈 FORECASTS",
            "⚔️ POSITIONS",
            "🔮 MATRIX",
            "📊 BACKTEST",
            "🚀 PIPELINE",
            "⚙️ SETTINGS",
        ]
    )

    # --- load bars + processed (shared) ---
    bars = pd.DataFrame()
    err: str | None = None
    if data_mode == "Demo":
        bars = get_demo_bars(symbol=symbol)
    elif data_mode == "IBKR":
        try:
            bars = _load_bars_ibkr(symbol, duration=duration.strip(), bar_size=bar_size)
        except Exception as e:
            err = str(e)
    else:
        bars = _load_bars_csv(symbol, csv_path if csv_path else None)

    if err is None and bars.empty and data_mode != "Demo":
        err = "No bars found. Switch **Bars source** to **Demo**, or fix the CSV path / IBKR connection."

    live_cfg = LiveRegimeModelConfig()
    prob_model = Path(prob_path.strip()) if prob_path.strip() else None

    processed: pd.DataFrame | None = None
    perr: str | None = None
    if err is None:
        processed, perr = run_feature_forecast_stack(
            bars,
            symbol=symbol,
            attach_vix=attach_vix,
            move_window=int(move_w),
            vol_window=int(vol_w),
            forecast_mode=forecast_mode,
            live=live_cfg,
            probabilistic_model_path=prob_model,
        )
        if perr:
            err = perr
            if forecast_mode in ("hmm", "markov") and data_mode == "Demo":
                err = (
                    f"{perr}\n\n"
                    "HMM/Markov often struggle on short synthetic series. "
                    "Switch forecast to **Deterministic** or load a longer **CSV** history."
                )

    if processed is not None and not processed.empty:
        st.session_state["proc_df"] = processed
        st.session_state["proc_symbol"] = symbol

    with tab1:
        st.subheader("Universe status")
        path_plans = st.text_input("Plans JSON", value=str(DEFAULT_PLANS), key="universe_plans_path")
        payload = safe_load_json(path_plans.strip())
        if not payload:
            st.warning("No plans file found. Run universe pipeline first.")
        else:
            rows = [flatten_universe_result(r) for r in payload.get("results", []) if isinstance(r, dict)]
            udf = pd.DataFrame(rows)
            if not udf.empty and "symbol" in udf.columns:
                udf = enrich_universe_with_feature_tail(udf, ROOT)
                n_extra = len([c for c in udf.columns if str(c).startswith("feat_")])
                st.caption(f"Merged latest numeric row from `features_{{SYM}}.csv` per symbol ({n_extra} feat_* columns).")
            sel = tables.render_universe_aggrid(udf, key="universe_grid")
            c1, c2 = st.columns([3, 1])
            with c1:
                if sel:
                    st.session_state["selected_symbol"] = sel
                    st.success(f"Selected **{sel}** — used in Forecasts / Matrix tabs.")
            with c2:
                if st.button("Drill-down row symbol", key="universe_drilldown_symbol"):
                    sym = st.session_state.get("selected_symbol", symbol)
                    modals.show_symbol_history_dialog(str(sym), st.session_state.get("proc_df"))
            if not udf.empty:
                st.download_button(
                    "Download universe CSV",
                    data=udf.to_csv(index=False).encode("utf-8"),
                    file_name="universe_flat.csv",
                    mime="text/csv",
                )

    with tab2:
        st.subheader("Live forecasts")
        sym = str(st.session_state.get("selected_symbol", symbol))
        st.caption(f"Symbol: **{sym}** — stack: `{forecast_mode}` — bars: **{data_mode}**")
        if err or processed is None or processed.empty:
            st.error(err or "No processed data.")
            if err and data_mode != "Demo":
                if st.button("Switch to Demo bars", key="btn_demo_bars"):
                    st.session_state["bars_mode_radio"] = "Demo"
                    st.rerun()
        else:
            fig_p = charts.fig_price_locus(processed, title=f"{sym} — price & locus")
            st.plotly_chart(fig_p, use_container_width=True)
            png = charts.write_figure_png_bytes(fig_p)
            if png:
                st.download_button("PNG chart", data=png, file_name=f"{sym}_price_locus.png", mime="image/png")
            c1, c2 = st.columns(2)
            with c1:
                last = processed.iloc[-1]
                st.plotly_chart(charts.fig_factor_radar(last), use_container_width=True)
            with c2:
                fan = charts.fig_quantile_fan(processed)
                if fan:
                    st.plotly_chart(fan, use_container_width=True)
                else:
                    st.info("Quantile fan appears when probabilistic columns exist.")
            hm = charts.fig_hmm_markov_stacked(processed)
            if hm:
                st.plotly_chart(hm, use_container_width=True)
            mtf = charts.fig_mtf_alignment_heatmap(processed)
            if mtf:
                st.plotly_chart(mtf, use_container_width=True)
            else:
                st.caption("No MTF/confluence numeric columns in this run.")

            chain_path = ROOT / rel_option_chain_csv(sym)
            chain_df = safe_read_csv(str(chain_path))
            if chain_df is None or chain_df.empty:
                chain_df = get_demo_option_chain_stub(symbol=sym)
                st.caption("Using **demo** option chain (no CSV on disk).")
            svi_fig = charts.fig_svi_surface_latest(chain_df)
            if svi_fig:
                st.plotly_chart(svi_fig, use_container_width=True)
            svi2 = charts.fig_svi_fitted_curves(chain_df)
            if svi2:
                st.plotly_chart(svi2, use_container_width=True)

            st.download_button(
                "Download forecast tail CSV",
                processed.tail(500).to_csv().encode("utf-8"),
                file_name=f"{sym}_forecast_tail.csv",
                mime="text/csv",
            )

    with tab3:
        st.subheader("ROEE positions & plans")
        _pp = str(st.session_state.get("universe_plans_path", str(DEFAULT_PLANS)))
        p = safe_load_json(str(Path(_pp).resolve() if _pp else DEFAULT_PLANS))
        pos = tables.positions_from_plans(p)
        if pos.empty:
            st.info("No active plans in JSON.")
        else:
            st.dataframe(pos, use_container_width=True)
            st.download_button("Positions CSV", pos.to_csv(index=False).encode("utf-8"), file_name="positions.csv")
            if massive_ok and st.button("Sample MTM from Massive (first plan)", key="mtm_massive"):
                try:
                    from rlm.roee.chain_match import estimate_mark_value_from_matched_legs

                    raw = (p or {}).get("active_ranked") or (p or {}).get("results") or []
                    first = next((x for x in raw if isinstance(x, dict) and x.get("matched_legs")), None)
                    if first and isinstance(first.get("matched_legs"), list):
                        v = estimate_mark_value_from_matched_legs(first["matched_legs"])
                        st.toast(f"Mid mark estimate: {v}", icon="✅")
                    else:
                        st.toast("No matched_legs on plans", icon="ℹ️")
                except Exception as e:
                    st.toast(f"MTM failed: {e}", icon="⚠️")
        tl = tables.trade_log_table(ROOT)
        if not tl.empty:
            st.subheader("trade_log.csv (recent)")
            st.dataframe(tl.tail(80), use_container_width=True)

        allow_live = st.checkbox(
            "Allow live force-exit (otherwise dry-run only)", value=False, key="pos_allow_live_force_exit"
        )
        if st.button("Force-exit dry run", key="pos_force_exit_dry_run"):
            cmd = f'{sys.executable} scripts/monitor_active_trade_plans.py --plans "{_pp}" --once'
            st.code(cmd, language="bash")
            if allow_live:
                st.warning("Paper-close live path not auto-run from UI; use terminal with `--paper-close`.")

    with tab4:
        st.subheader("Regime locus matrix explorer")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            d = st.selectbox("Direction", DIRECTIONS, key="matrix_direction")
        with c2:
            v = st.selectbox("Volatility", VOLS, key="matrix_volatility")
        with c3:
            liq = st.selectbox(
                "Liquidity", ["high_liquidity", "low_liquidity", "transition"], key="matrix_liquidity"
            )
        with c4:
            flow = st.selectbox(
                "Dealer flow", ["supportive", "adversarial", "transition"], key="matrix_dealer_flow"
            )
        price = st.number_input("Synthetic price", value=500.0, step=1.0, key="matrix_synthetic_price")
        sigma = st.number_input("Synthetic σ", value=0.02, step=0.001, format="%.4f", key="matrix_synthetic_sigma")
        s_d = st.slider("S_D", -1.0, 1.0, 0.0, key="matrix_s_d")
        s_v = st.slider("S_V", -1.0, 1.0, 0.0, key="matrix_s_v")
        s_l = st.slider("S_L", -1.0, 1.0, 0.0, key="matrix_s_l")
        s_g = st.slider("S_G", -1.0, 1.0, 0.0, key="matrix_s_g")
        dec = select_trade(
            current_price=float(price),
            sigma=float(sigma),
            s_d=float(s_d),
            s_v=float(s_v),
            s_l=float(s_l),
            s_g=float(s_g),
            direction_regime=d,
            volatility_regime=v,
            liquidity_regime=liq,
            dealer_flow_regime=flow,
            regime_key=f"{d}|{v}|{liq}|{flow}",
            strike_increment=float(strike_inc),
            short_dte=short_dte,
        )
        st.json(
            {
                "action": dec.action,
                "strategy_name": dec.strategy_name,
                "rationale": dec.rationale,
                "size_fraction": dec.size_fraction,
                "legs": [str(x) for x in (dec.legs or [])],
            }
        )
        z, labels, yl, xl = _regime_heatmap_matrix(liq, flow, short_dte=short_dte)
        hm = go.Figure(
            data=go.Heatmap(
                z=z,
                x=xl,
                y=yl,
                text=labels,
                texttemplate="%{text}<br>%{z:.1f}%",
                colorscale="Viridis",
            )
        )
        hm.update_layout(template="plotly_dark", height=400, title="Strategy map (max risk %)")
        st.plotly_chart(hm, use_container_width=True)

        if processed is not None and not processed.empty:
            st.subheader("Bar-level `select_trade_for_row` (latest)")
            last = processed.iloc[-1]
            hmm_thr = 0.6 if forecast_mode in ("hmm", "markov") else None
            d2 = select_trade_for_row(
                last,
                strike_increment=float(strike_inc),
                hmm_confidence_threshold=hmm_thr,
                short_dte=short_dte,
            )
            st.write(d2.action, d2.strategy_name, d2.rationale)

    with tab5:
        st.subheader("Backtest & walk-forward")
        symf = str(st.session_state.get("selected_symbol", symbol))
        eq_path = ROOT / "data" / "processed" / backtest_equity_filename(symf)
        wf_path = ROOT / "data" / "processed" / walkforward_summary_filename(symf)
        eq = safe_read_csv(str(eq_path))
        wf = safe_read_csv(str(wf_path))
        if eq is not None and not eq.empty:
            reg_col = "regime_key" if "regime_key" in eq.columns else None
            st.plotly_chart(charts.fig_equity_regime(eq, reg_col), use_container_width=True)
            st.download_button("Equity CSV", eq.to_csv(index=False).encode("utf-8"), file_name="equity.csv")
        else:
            st.info(f"No equity file at `{eq_path}`.")
        if wf is not None and not wf.empty:
            st.subheader("Walk-forward summary")
            st.dataframe(wf, use_container_width=True)
            st.download_button("WF summary CSV", wf.to_csv(index=False).encode("utf-8"), file_name="wf_summary.csv")
        else:
            st.caption(f"No walk-forward summary at `{wf_path}`. Run `scripts/run_walkforward.py`.")
        st.markdown(
            "**Tournament:** compare Deterministic / HMM / Markov / Probabilistic by running walk-forward "
            "with different flags and comparing Sharpe columns in each `walkforward_summary_*.csv`."
        )
        up = st.file_uploader("Optional 2nd summary CSV to preview", type=["csv"])
        if up is not None:
            df2 = pd.read_csv(up)
            st.dataframe(df2.head(50), use_container_width=True)

    with tab6:
        st.subheader("Pipeline control")
        py = sys.executable
        actions = [
            ("Forecast pipeline", "scripts/run_forecast_pipeline.py", ["--no-vix"]),
            ("ROEE pipeline", "scripts/run_roee_pipeline.py", []),
            ("Universe options", "scripts/run_universe_options_pipeline.py", ["--no-vix", "--top", "5"]),
            ("Calibrate regimes", "scripts/calibrate_regime_models.py", []),
            ("Data lake", "scripts/run_data_lake_pipeline.py", []),
            (
                "Run everything (no follow)",
                "scripts/run_everything.py",
                ["--skip-monitor", "--pipeline-args=--no-vix"],
            ),
        ]
        for i, (label, script, args) in enumerate(actions):
            btn_key = f"pipe_btn_{i}_{script.replace('/', '_').replace('.', '_')}"
            if st.button(label, key=btn_key):
                with st.spinner(f"Running {script}…"):
                    code, out, err = run_repo_script(ROOT, py, script, args, timeout_s=None)
                append_pipeline_log(f"# {label}\n{out}\n{err}\nexit={code}\n")
                clear_file_caches()
                st.toast(f"{label}: exit {code}", icon="✅" if code == 0 else "⚠️")
        log_lines = list(st.session_state.get("pipeline_log", []) or [])
        st.code("\n".join(log_lines[-400:]) if log_lines else "(no logs yet)", language="text")

    with tab7:
        st.subheader("Settings (session)")
        st.json(LiveRegimeModelConfig().model_dump())
        st.caption("Adjust forecast stack from sidebar; full YAML persistence is intentionally not written here.")
        fc = safe_read_csv(str(ROOT / rel_forecast_features_csv(symbol)))
        if fc is not None and not fc.empty:
            st.success(f"Found `{rel_forecast_features_csv(symbol)}` ({len(fc)} rows).")
        else:
            st.caption("No precomputed forecast_features CSV for this symbol.")

    if err and data_mode != "Demo":
        st.sidebar.error(err[:400])

    if auto_refresh:
        st.sidebar.caption("30s auto-refresh: click **Refresh data** or reload the page periodically.")

if __name__ == "__main__":
    main()
