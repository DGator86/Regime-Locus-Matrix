"""``st.dialog`` drill-downs."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st


def show_symbol_history_dialog(symbol: str, processed: pd.DataFrame | None) -> None:
    """
    Open a Streamlit dialog showing recent processed feature rows for a symbol.
    
    If `processed` is None or empty, the dialog shows a warning and closes. Otherwise the dialog displays the last 240 rows of a curated set of columns from `processed`: specific feature/regime names (e.g., "close", "sigma", "S_*", various "_regime" columns, "regime_key", "upper_1s", "lower_1s") plus any columns whose names contain "hmm" or "markov" (case-insensitive).
    
    Parameters:
        symbol (str): Symbol used in the dialog title.
        processed (pd.DataFrame | None): Processed feature frame for the symbol; columns will be filtered to the curated set before rendering.
    """

    @st.dialog(f"Drill-down — {symbol}", width="large")
    def _inner() -> None:
        """
        Display a filtered view of the processed feature frame inside the dialog.
        
        If `processed` is None or empty, shows a warning and does nothing. Otherwise selects columns with specific feature/regime names or whose names contain "hmm" or "markov" (case-insensitive) and renders the last 240 rows as a Streamlit dataframe using the container width and a fixed height of 420.
        """
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
    """
    Open a Streamlit dialog with the given title and render the provided object as formatted JSON.
    
    The object is serialized with indentation for readability; values that are not JSON-serializable are converted using str().
    
    Parameters:
        title (str): Title for the dialog window.
        obj (object): Object to serialize and display as JSON.
    """
    @st.dialog(title, width="large")
    def _inner() -> None:
        st.code(json.dumps(obj, indent=2, default=str), language="json")

    _inner()
