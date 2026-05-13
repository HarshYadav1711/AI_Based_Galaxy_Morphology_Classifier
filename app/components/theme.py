"""Minimal astronomy-inspired Streamlit styling (CSS only)."""

from __future__ import annotations

import streamlit as st


def inject_theme() -> None:
    st.markdown(
        """
        <style>
            :root {
                --cosmos-bg-0: #070b14;
                --cosmos-bg-1: #0f1628;
                --cosmos-accent: #7eb8ff;
                --cosmos-muted: #9aa7bd;
                --cosmos-card: rgba(255,255,255,0.05);
                --cosmos-border: rgba(126,184,255,0.25);
            }
            [data-testid="stAppViewContainer"] {
                background: linear-gradient(
                    165deg, var(--cosmos-bg-0) 0%, #10182a 45%, var(--cosmos-bg-1) 100%
                );
            }
            [data-testid="stHeader"] { background-color: transparent; }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0b1020 0%, #121a2e 100%);
                border-right: 1px solid var(--cosmos-border);
            }
            h1, h2, h3 { color: #e8eefc !important; letter-spacing: 0.02em; }
            p, span, label { color: var(--cosmos-muted) !important; }
            [data-testid="stWidgetLabel"] p { color: #c5d0e6 !important; }
            div[data-testid="stExpander"] {
                background: var(--cosmos-card);
                border: 1px solid var(--cosmos-border);
                border-radius: 10px;
            }
            [data-testid="stMetricValue"] { color: var(--cosmos-accent) !important; }
            [data-testid="stMetricLabel"] { color: #aeb9cc !important; }
            .block-container { padding-top: 1.5rem; }
            hr { border-color: var(--cosmos-border) !important; opacity: 0.6; }
        </style>
        """,
        unsafe_allow_html=True,
    )
