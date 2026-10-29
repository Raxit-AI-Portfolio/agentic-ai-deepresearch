"""Shared header/navbar/footer — imported by every page so the chrome around
the app stays identical without duplicating markup per page."""

from __future__ import annotations

import streamlit as st

_LOGO_SVG = """
<svg width="32" height="32" viewBox="0 0 34 34" xmlns="http://www.w3.org/2000/svg">
  <circle cx="17" cy="17" r="16" fill="#4F46E5"/>
  <path d="M10 20 L15 12 L19 17 L24 9" stroke="white" stroke-width="2.2"
        fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="24" cy="9" r="2" fill="white"/>
</svg>
"""

# (label, target script path) — target paths are relative to the app root,
# matching what st.page_link / st.switch_page expect in a multipage app.
_PAGES = [
    ("Home", "Home.py"),
    ("Deep Research", "pages/1_Deep_Research.py"),
]


def apply_page_config(page_title: str) -> None:
    """Call once, first thing, in every page script."""
    st.set_page_config(
        page_title=f"{page_title} · Deep Research Assistant",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    # The custom header below replaces Streamlit's default sidebar page nav.
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stSidebarCollapsedControl"] {display: none;}
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1100px;}
        /* Multiselect (e.g. the source whitelist/blacklist controls) clips
           its selected chips to one line by default once there are more
           than fit the widget's width — wrap them onto multiple lines
           instead, with a scrollbar past a sane height so one huge list
           can't push the rest of the page down indefinitely. */
        [data-testid="stMultiSelect"] div[data-baseweb="select"] > div:first-child {
            flex-wrap: wrap !important;
            max-height: 160px;
            overflow-y: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(active: str) -> None:
    left, right = st.columns([2, 3])
    with left:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:.6rem;">'
            f'{_LOGO_SVG}'
            f'<span style="font-size:1.25rem;font-weight:700;">Deep Research Assistant</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with right:
        nav_cols = st.columns([1] * len(_PAGES) + [3])
        for i, (label, target) in enumerate(_PAGES):
            with nav_cols[i]:
                st.page_link(target, label=label, disabled=(label == active))
    st.divider()


def render_footer() -> None:
    st.divider()
    st.markdown(
        '<div style="text-align:center;color:gray;font-size:.85rem;padding-top:.5rem;">'
        'Deep Research Assistant — planning, research, and writing orchestrated with LangGraph.'
        '</div>',
        unsafe_allow_html=True,
    )
