"""AgGrid tables for universe and positions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder, GridUpdateMode

    _HAS_AGGRID = True
except ImportError:
    _HAS_AGGRID = False


def render_universe_aggrid(df: pd.DataFrame, *, key: str) -> str | None:
    """
    Render sortable/filterable universe table.
    Returns selected ``symbol`` column if a row is selected.
    """
    if df.empty:
        st.info("No universe rows to display.")
        return None
    if not _HAS_AGGRID:
        st.dataframe(df, use_container_width=True, height=400)
        sym_col = "symbol" if "symbol" in df.columns else None
        choices = (
            [""] + [str(x) for x in df[sym_col].dropna().unique()]
            if sym_col
            else [""] + [str(i) for i in range(min(10, len(df)))]
        )
        pick = st.selectbox("Select symbol / row", choices)
        return pick or None

    try:
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_selection(selection_mode="single", use_checkbox=False)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25)
        gb.configure_side_bar()
        go = gb.build()
        resp = AgGrid(
            df,
            gridOptions=go,
            height=420,
            width="100%",
            data_return_mode=DataReturnMode.FILTERED,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            fit_columns_on_grid_load=True,
            key=key,
            theme="streamlit",
            enable_enterprise_modules=False,
        )
    except Exception as e:
        st.warning(f"AgGrid failed ({e}); using dataframe.")
        st.dataframe(df, use_container_width=True, height=400)
        return None

    selected = resp.get("selected_rows") or []
    if isinstance(selected, pd.DataFrame) and not selected.empty:
        sym = selected.iloc[0].get("symbol")
        return str(sym) if sym is not None else None
    if isinstance(selected, list) and selected:
        row = selected[0]
        if isinstance(row, dict):
            sym = row.get("symbol")
            return str(sym) if sym is not None else None
    return None


def positions_from_plans(payload: dict[str, Any] | None) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    rows = payload.get("active_ranked") or [r for r in payload.get("results", []) if r.get("status") == "active"]
    if not rows:
        return pd.DataFrame()
    flat: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        dec = r.get("decision") if isinstance(r.get("decision"), dict) else {}
        flat.append(
            {
                "symbol": r.get("symbol"),
                "status": r.get("status"),
                "plan_id": r.get("plan_id"),
                "action": dec.get("action"),
                "strategy": dec.get("strategy_name"),
                "rank_score": r.get("rank_score"),
                "entry_debit": r.get("entry_debit_dollars"),
                "mid_mark": r.get("entry_mid_mark_dollars"),
                "skip_reason": r.get("skip_reason"),
                "matched_legs": str(r.get("matched_legs", ""))[:300],
            }
        )
    return pd.DataFrame(flat)


def trade_log_table(root: Path) -> pd.DataFrame:
    p = root / "data" / "processed" / "trade_log.csv"
    if not p.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except (OSError, ValueError):
        return pd.DataFrame()
