import streamlit as st
import pandas as pd
from sidebar_functions import process_traffic_data, load_traffic_data
from datetime import datetime, timedelta

# page configuration
st.set_page_config(
    page_title="ADVANTEC WEB APP",
    page_icon="🛣️",
    layout="wide"
)

# Main Title
st.title("Active Transportation & Operations Management Dashboard")

# Dashboard objective
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

# Load Data with progress indicator
with st.spinner('Loading traffic data from all corridor segments...'):
    df = load_traffic_data()

if df.empty:
    st.error("Failed to load traffic data. Please check your connection.")
    st.stop()

# Get available corridors from the data
corridor_options = ["All Segments"] + sorted(df['segment_name'].unique().tolist())

# Sidebar
with st.sidebar:
    st.title("🛣️ Controls")

    # Filter section
    with st.expander("📊 Data Filters", expanded=True):
        # Corridor selection with actual data
        corridor = st.selectbox("Corridor Segment", corridor_options)

        # Date range selector (use actual data range)
        min_date = df['local_datetime'].dt.date.min()
        max_date = df['local_datetime'].dt.date.max()
        
        st.subheader("📅 Date Range")
        st.info(f"Available data: {min_date} to {max_date}")
        
        date_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Choose start and end dates for analysis"
        )

        # Time granularity selector
        st.subheader("⏰ Data Granularity")
        granularity = st.selectbox(
            "Aggregation Level",
            options=["Hourly", "Daily", "Weekly", "Monthly"],
            index=0,
            help="Choose how to aggregate your hourly data"
        )

        # Time of day filter for hourly data
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

# Main content
st.header("🛣️ Washington Street Corridor Analysis")

# Filter data by selected corridor
if corridor != "All Segments":
    display_df = df[df['segment_name'] == corridor].copy()
else:
    display_df = df.copy()

# Process the data using sidebar selections
if len(date_range) == 2:
    filtered_data = process_traffic_data(
        display_df, date_range, granularity,
        time_filter if granularity == "Hourly" and 'time_filter' in locals() else None,
        start_hour if 'start_hour' in locals() else None,
        end_hour if 'end_hour' in locals() else None
    )
    
    # Display summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(filtered_data):,}")
    with col2:
        st.metric("Segments", filtered_data['segment_name'].nunique())
    with col3:
        st.metric("Date Range", f"{len(filtered_data['local_datetime'].dt.date.unique())} days")
    with col4:
        avg_speed = filtered_data['average_speed'].mean()
        st.metric("Avg Speed", f"{avg_speed:.1f} mph")
    
    # Display sample of filtered data
    with st.expander("📊 Data Preview", expanded=False):
        st.dataframe(filtered_data.head(100))
        
else:
    st.warning("Please select both start and end dates")
