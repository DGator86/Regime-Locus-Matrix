"""``st.dialog`` drill-downs."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st


def show_symbol_history_dialog(symbol: str, processed: pd.DataFrame | None) -> None:
    """Factor + regime tail for the selected symbol (must be called to open dialog)."""

    @st.dialog(f"Drill-down — {symbol}", width="large")
    def _inner() -> None:
        if processed is None or processed.empty:
            st.warning("No processed feature frame in session for this symbol.")
            return
        cols = [
            c
            for c in processed.columns
            if c
            in (
                "close",
                "sigma",
                "S_D",
                "S_V",
                "S_L",
                "S_G",
                "direction_regime",
                "volatility_regime",
                "liquidity_regime",
                "dealer_flow_regime",
                "regime_key",
                "upper_1s",
                "lower_1s",
            )
            or "hmm" in c.lower()
            or "markov" in c.lower()
        ]
        cols = [c for c in cols if c in processed.columns]
        st.dataframe(processed[cols].tail(240), use_container_width=True, height=420)

    _inner()


def show_json_dialog(title: str, obj: object) -> None:
    @st.dialog(title, width="large")
    def _inner() -> None:
        st.code(json.dumps(obj, indent=2, default=str), language="json")

    _inner()
