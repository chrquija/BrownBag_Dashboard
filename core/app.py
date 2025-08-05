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
<div style='
    display: flex; 
    justify-content: center; 
    align-items: center;
    text-align: center;
    margin-bottom: 2rem;
'>
    <span style="
        font-size: 1.15rem;
        font-weight: 400;
        color: var(--text-color);
        background: var(--background-color);
        padding: 0.7rem 1.2rem;
        border-radius: 14px;
        box-shadow: 0 2px 16px 0 var(--shadow-color, rgba(0,0,0,0.06));
        max-width: 850px;
        line-height: 1.7;
        ">
        The ADVANTEC App provides advanced traffic engineering recommendations for the Coachella Valley through Machine Learning algorithms trained on HUNDREDS OF THOUSANDS OF DATA POINTS to reduce Travel Time, Fuel Consumption, and Green House Gases through identification of anomalies, provision of cycle length recommendations, and predictive analytics.
    </span>
</div>
"""

st.markdown(dashboard_objective, unsafe_allow_html=True)
