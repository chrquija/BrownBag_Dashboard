import streamlit as st
import pandas as pd
from sidebar_functions import process_traffic_data  # Import your function
from datetime import datetime, timedelta

# page configuration
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
    <b>The ADVANTEC App</b> provides traffic engineering recommendations for the Coachella Valley using <b>MILLIONS OF DATA POINTS trained on Machine Learning Algorithms to REDUCE Travel Time, Fuel Consumption, and Green House Gases.</b> This is accomplished through the identification of anomalies, provision of cycle length recommendations, and predictive modeling.
</div>
"""

st.markdown(dashboard_objective, unsafe_allow_html=True)
st.header("🛣️ Corridor Analysis")

# Load Data
df = pd.read_csv("https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/MOCK_DATA/mock_corridor_data.csv")


# Date range and granularity selector
with st.sidebar:
    st.title("🛣️ Controls")

    # Filter section
    with st.expander("📊 Data Filters", expanded=True):
        corridor = st.selectbox("Corridor", ["Option 1", "Option 2"])

        # Date range selector
        st.subheader("📅 Date Range")
        date_range = st.date_input(
            "Select Date Range",
            value=(datetime.now() - timedelta(days=7), datetime.now()),
            help="Choose start and end dates for analysis"
        )

        # Time granularity selector
        st.subheader("⏰ Data Granularity")
        granularity = st.selectbox(
            "Aggregation Level",
            options=["Hourly", "Daily", "Weekly", "Monthly"],
            index=0,  # Default to hourly
            help="Choose how to aggregate your hourly data"
        )

        # Optional: Time of day filter for hourly data
        if granularity == "Hourly":
            time_filter = st.selectbox(
                "Time Period",
                options=["All Hours", "Peak Hours (7-9 AM, 4-6 PM)", "AM Peak (7-9 AM)",
                         "PM Peak (4-6 PM)", "Off-Peak", "Custom Range"],
                help="Filter specific hours of the day"
            )

            if time_filter == "Custom Range":
                col1, col2 = st.columns(2)
                with col1:
                    start_hour = st.selectbox("Start Hour", range(0, 24), index=7)
                with col2:
                    end_hour = st.selectbox("End Hour", range(1, 25), index=18)

    # Analysis tools
    with st.expander("🔧 Analysis Tools"):
        show_anomalies = st.checkbox("Show Anomalies")
        show_predictions = st.checkbox("Show Predictions")
        confidence_level = st.slider("Confidence Level", 80, 99, 95)

# Process the data using your sidebar selections (if dates are selected)
if len(date_range) == 2:
    filtered_data = process_traffic_data(
        df, date_range, granularity,
        time_filter if granularity == "Hourly" else None,
        start_hour if 'start_hour' in locals() else None,
        end_hour if 'end_hour' in locals() else None
    )
    st.write(f"Showing data from {date_range[0]} to {date_range[1]} at {granularity.lower()} granularity")
    st.write(f"Total records: {len(filtered_data)}")
else:
    st.warning("Please select both start and end dates")

