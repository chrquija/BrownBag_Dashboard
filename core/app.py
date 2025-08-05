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

# 3. Subheader / Dashboard Objective
st.markdown(
    """
    <div style="font-size:1.1em; color:#444; background-color:#F6F6F6; border-radius:8px; padding:16px 20px; margin-bottom:18px;">
    <b>The ADVANTEC App</b> provides advanced traffic engineering recommendations for the Coachella Valley through Machine Learning algorithms trained on hundreds of thousands of data points.<br><br>
    <b>Objective:</b> Reduce travel time, fuel consumption, and greenhouse gas emissions by detecting anomalies, delivering real-time cycle length recommendations, and offering predictive analytics for smarter corridor and intersection management.
    </div>
    """,
    unsafe_allow_html=True
)