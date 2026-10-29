import streamlit as st

from components.layout import apply_page_config, render_footer, render_header

apply_page_config("Home")
render_header("Home")

st.markdown("## Welcome 👋")
st.write(
    "The Deep Research Assistant plans, researches, writes, and exports a "
    "fully-sourced report on any topic you give it — with a human review "
    "step for the outline before any writing happens."
)
st.markdown(
    "**How it works:**\n"
    "1. Configure your report and generate an outline.\n"
    "2. Review and edit the outline, and optionally supply your own media.\n"
    "3. Submit to generate the full report, then download it."
)

if st.button("Start a new report →", type="primary"):
    st.switch_page("pages/1_Deep_Research.py")

render_footer()
