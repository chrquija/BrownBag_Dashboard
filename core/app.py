import streamlit as st
import pandas as pd

#page configuration
st.set_page_config(
    page_title="ADVANTEC WEB APP",
    page_icon="🛣️",
    layout="wide"
)

# 2. Main Title
st.title("Active Transportation & Operations Management Dashboard")

# 3. Centered, theme-adaptive dashboard objective/subheader
dashboard_objective = """
<div style="
    font-size: 1.15rem;
    font-weight: 400;
    color: var(--text-color);
    background: var(--background-color);
    padding: 1.2rem 1.5rem;
    border-radius: 14px;
    box-shadow: 0 2px 16px 0 var(--shadow-color, rgba(0,0,0,0.06));
    margin-bottom: 2rem;
    line-height: 1.7;
    ">
    <b>The ADVANTEC App</b> provides traffic engineering recommendations for the Coachella Valley through Machine Learning algorithms trained on <b>MILLIONS OF DATA POINTS to REDUCE Travel Time, Fuel Consumption, and Green House Gases</b> through identification of anomalies, provision of cycle length recommendations, and predictive analytics.
</div>
"""

st.markdown(dashboard_objective, unsafe_allow_html=True)
