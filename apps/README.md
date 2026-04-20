# apps/

Streamlit applications and interactive dashboards.

These are **not** part of the core `rlm` package.  Run them directly:

```bash
# RLM Terminal
streamlit run apps/rlm_terminal.py

# Control Center
streamlit run apps/rlm_control_center/app.py

# Live Trading Dashboard
streamlit run apps/live_trading_dashboard.py

# P&L Chart
streamlit run apps/pnl_chart_dashboard.py
```

Install UI dependencies first:

```bash
pip install -e ".[ui]"
```
