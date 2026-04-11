"""Cyber-dark Streamlit theme — complements ``.streamlit/config.toml``."""

from __future__ import annotations

import streamlit as st


def inject_custom_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Main shell — Streamlit 1.33+ */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
.block-container {
  background-color: #0a0a0a !important;
  color: #ececf1 !important;
  font-family: 'Inter', system-ui, sans-serif !important;
}

[data-testid="stHeader"] {
  background: rgba(10, 10, 10, 0.92) !important;
  border-bottom: 1px solid rgba(0, 245, 255, 0.12);
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #101014 0%, #0a0a0f 100%) !important;
  border-right: 1px solid rgba(0, 245, 255, 0.15) !important;
}

[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label {
  color: #dce0e8 !important;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
  background-color: #141418 !important;
  border-radius: 12px;
  padding: 6px 8px;
  border: 1px solid rgba(0, 245, 255, 0.12);
}

.stTabs [data-baseweb="tab"] {
  border-radius: 8px;
  color: #b8bcc8 !important;
}

.stTabs [aria-selected="true"] {
  background: rgba(0, 245, 255, 0.14) !important;
  color: #00f5ff !important;
}

div[data-testid="stMetric"] {
  background: #141418;
  border: 1px solid rgba(0, 245, 255, 0.12);
  border-radius: 12px;
  padding: 0.65rem 0.85rem;
}

div[data-testid="stMetricValue"] {
  color: #00f5ff !important;
}

div[data-testid="stMetricLabel"] {
  color: #9aa0ae !important;
}

.stAlert, [data-testid="stNotification"] {
  border-radius: 10px;
}

code, [data-testid="stCode"] pre {
  font-family: 'JetBrains Mono', monospace !important;
}

.neon-text {
  color: #00f5ff;
  text-shadow: 0 0 14px rgba(0, 245, 255, 0.35);
  font-weight: 700;
  letter-spacing: 0.06em;
}

.pulse-dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-right: 8px;
  animation: rlmPulse 1.8s ease-in-out infinite;
}

.dot-live { background: #00ff9d; box-shadow: 0 0 10px #00ff9d; }
.dot-off { background: #555; box-shadow: none; }

@keyframes rlmPulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.55; transform: scale(0.92); }
}

@keyframes rlmDrift {
  0% { background-position: 0% 50%; }
  100% { background-position: 100% 50%; }
}

.header-aurora {
  background: linear-gradient(
    120deg,
    rgba(0, 245, 255, 0.1) 0%,
    rgba(168, 85, 247, 0.08) 45%,
    rgba(0, 255, 157, 0.06) 100%
  );
  background-size: 200% 200%;
  animation: rlmDrift 20s ease infinite alternate;
  border-radius: 14px;
  padding: 1rem 1.25rem;
  border: 1px solid rgba(0, 245, 255, 0.18);
  margin-bottom: 1rem;
}

/* Hide Streamlit Community Cloud “Deploy” — local-only app. */
[data-testid="stDeployButton"] {
  display: none !important;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
        """,
        unsafe_allow_html=True,
    )
