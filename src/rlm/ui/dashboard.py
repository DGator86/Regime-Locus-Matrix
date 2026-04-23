import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timezone
import os

# Set page config for a premium feel
st.set_page_config(
    page_title="RLM Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for a sleek look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #161b22;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb;
    }
    </style>
    """, unsafe_allow_html=True)

def load_data():
    data_root = Path(os.environ.get("RLM_DATA_ROOT", "data"))
    
    # 1. Challenge State
    challenge_state = None
    challenge_path = data_root / "challenge" / "state.json"
    if challenge_path.exists():
        try:
            challenge_state = json.loads(challenge_path.read_text(encoding="utf-8"))
        except: pass
        
    # 2. Trade Log (Swing Options)
    trade_log = pd.DataFrame()
    log_path = data_root / "processed" / "trade_log.csv"
    if log_path.exists():
        try:
            trade_log = pd.read_csv(log_path)
        except: pass
        
    # 3. Equities State
    equity_state = {}
    eq_path = data_root / "processed" / "equity_positions_state.json"
    if eq_path.exists():
        try:
            equity_state = json.loads(eq_path.read_text(encoding="utf-8"))
        except: pass
        
    return challenge_state, trade_log, equity_state

# --- App Layout ---

st.title("🛡️ Starfleet Crew Terminal")
st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")

challenge, swing_log, equities = load_data()

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)

total_balance = 0
if challenge:
    total_balance = challenge.get("balance", 0)

realized_pnl = 0
unrealized_pnl = 0
if not swing_log.empty:
    realized_pnl = swing_log[swing_log["closed"] == 1]["unrealized_pnl"].sum()
    unrealized_pnl = swing_log[swing_log["closed"] == 0]["unrealized_pnl"].sum()

with col1:
    st.metric("Challenge Balance", f"${total_balance:,.2f}", delta=f"{challenge.get('total_return_pct', 0):.1f}%" if challenge else None)

with col2:
    st.metric("Swing Realized PnL", f"${realized_pnl:,.2f}")

with col3:
    st.metric("Swing Unrealized PnL", f"${unrealized_pnl:,.2f}")

with col4:
    win_rate = 0
    if challenge:
        win_rate = challenge.get("win_rate", 0) * 100
    st.metric("Challenge Win Rate", f"{win_rate:.1f}%")

st.divider()

tab_challenge, tab_swing, tab_equities = st.tabs(["🚀 Daytrade Challenge", "📊 Swing Options", "📈 Equities"])

with tab_challenge:
    if challenge:
        st.subheader("Challenge Progress")
        progress = challenge.get("progress_pct", 0)
        st.progress(progress, text=f"Progress to $25k: {progress*100:.1f}%")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Stats**")
            st.write(f"- **Sessions**: {challenge.get('session_count', 0)}")
            st.write(f"- **Wins**: {challenge.get('wins', 0)}")
            st.write(f"- **Losses**: {challenge.get('losses', 0)}")
        with c2:
            st.write("**Current Milestone**")
            m = challenge.get("current_milestone", {})
            st.write(f"- **Label**: {m.get('label', 'N/A')}")
            st.write(f"- **Target**: ${m.get('target', 0):,.0f}")
            
        st.subheader("Open Positions")
        open_pos = challenge.get("open_positions", [])
        if open_pos:
            st.table(pd.DataFrame(open_pos))
        else:
            st.info("No open challenge positions.")
    else:
        st.warning("No challenge data found.")

with tab_swing:
    st.subheader("Active Swing Trades")
    if not swing_log.empty:
        active = swing_log[swing_log["closed"] == 0]
        if not active.empty:
            st.dataframe(active, width="stretch")
        else:
            st.info("No active swing trades.")
            
        st.subheader("Trade History")
        st.dataframe(swing_log.tail(20), width="stretch")
    else:
        st.warning("No trade log found.")

with tab_equities:
    st.subheader("Open Equity Positions")
    if equities:
        open_eq = [v for v in equities.values() if v.get("status") == "open"]
        if open_eq:
            st.dataframe(pd.DataFrame(open_eq), width="stretch")
        else:
            st.info("No open equity positions.")
    else:
        st.warning("No equity state data found.")

if st.button("Refresh Data"):
    st.rerun()
