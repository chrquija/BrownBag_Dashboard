import streamlit as st
import pandas as pd
from Interactive_Map import create_corridor_map


# Page configuration
st.set_page_config(
    page_title="ADVANTEC WEB APP",
    page_icon="🛣️",
    layout="wide"
)

st.title("Active Transportation & Operations Management Dashboard")

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
    <b>The ADVANTEC App</b> provides traffic engineering recommendations for the Coachella Valley using <b>MILLIONS OF DATA POINTS trained on Machine Learning Algorithms to REDUCE Travel Time, Fuel Consumption, and Green House Gases.</b> This is accomplished through the identification of anomalies, provision of cycle length recommendations, and predictive modeling.
</div>
"""

st.markdown(dashboard_objective, unsafe_allow_html=True)

# Load Data
df = pd.read_csv("https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/MOCK_DATA/mock_corridor_data.csv")

# Corridor selection
df["corridor"] = df["origin"] + " → " + df["destination"]
corridor_options = sorted(df["corridor"].unique())
corridor = st.selectbox("Select Corridor", corridor_options)

# --- FRIENDLY DATE FILTER (show as YYYY-MM-DD, not with time) ---
df["date"] = pd.to_datetime(df["date"]).dt.date
date_options = sorted(df["date"].unique())
date = st.selectbox("Select Date", date_options)

# Variable toggle
variable = st.radio(
    "Variable to visualize",
    ["travel_time", "speed"],
    format_func=lambda x: "Travel Time (min)" if x == "travel_time" else "Speed (mph)"
)

# Filtered Data
filtered = df[
    (df["corridor"] == corridor) &
    (df["date"] == date)
]

if filtered.empty:
    st.warning("No data for this selection.")
else:
    st.subheader(f"Map: {corridor} on {date} ({variable.replace('_',' ').title()})")
    st.pydeck_chart(create_corridor_map(filtered, variable=variable))

    st.subheader(f"{variable.replace('_',' ').title()} by Hour")
    st.line_chart(filtered.set_index("hour")[variable])

    with st.expander("Show Data Table"):
        st.dataframe(filtered)