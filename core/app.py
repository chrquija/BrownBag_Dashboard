import streamlit as st
import pandas as pd
from sidebar_functions import process_traffic_data, load_traffic_data, load_volume_data
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

# Create tabs for different analyses
tab1, tab2 = st.tabs(["🛣️ Corridor Performance", "🚦 Intersection Volumes"])

with tab1:
    st.header("Washington Street Corridor Analysis")
    
    # Load corridor data
    with st.spinner('Loading corridor data...'):
        corridor_df = load_traffic_data()

    if corridor_df.empty:
        st.error("Failed to load corridor data.")
    else:
        # Get available corridors
        corridor_options = ["All Segments"] + sorted(corridor_df['segment_name'].unique().tolist())

        # Sidebar for corridor analysis
        with st.sidebar:
            st.title("🛣️ Corridor Controls")
            
            with st.expander("📊 Corridor Filters", expanded=True):
                corridor = st.selectbox("Corridor Segment", corridor_options)

                # Date range selector
                min_date = corridor_df['local_datetime'].dt.date.min()
                max_date = corridor_df['local_datetime'].dt.date.max()
                
                st.subheader("📅 Date Range")
                st.info(f"Available data: {min_date} to {max_date}")
                
                date_range = st.date_input(
                    "Select Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )

                # Time granularity
                st.subheader("⏰ Data Granularity")
                granularity = st.selectbox(
                    "Aggregation Level",
                    options=["Hourly", "Daily", "Weekly", "Monthly"],
                    index=0
                )

                # Time filters for hourly data
                if granularity == "Hourly":
                    time_filter = st.selectbox(
                        "Time Period",
                        options=["All Hours", "Peak Hours (7-9 AM, 4-6 PM)", "AM Peak (7-9 AM)",
                                 "PM Peak (4-6 PM)", "Off-Peak", "Custom Range"]
                    )

                    if time_filter == "Custom Range":
                        col1, col2 = st.columns(2)
                        with col1:
                            start_hour = st.selectbox("Start Hour", range(0, 24), index=7)
                        with col2:
                            end_hour = st.selectbox("End Hour", range(1, 25), index=18)

            with st.expander("🔧 Analysis Tools"):
                show_anomalies = st.checkbox("Show Anomalies")
                show_predictions = st.checkbox("Show Predictions")
                confidence_level = st.slider("Confidence Level", 80, 99, 95)

        # Process corridor data
        if len(date_range) == 2:
            # Filter data by selected corridor
            if corridor != "All Segments":
                display_df = corridor_df[corridor_df['segment_name'] == corridor].copy()
            else:
                display_df = corridor_df.copy()

            # Process the data
            filtered_data = process_traffic_data(
                display_df, date_range, granularity,
                time_filter if granularity == "Hourly" and 'time_filter' in locals() else None,
                start_hour if 'start_hour' in locals() else None,
                end_hour if 'end_hour' in locals() else None
            )
            
            # Display metrics
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
            
            # Data preview
            with st.expander("📊 Corridor Data Preview"):
                st.dataframe(filtered_data.head(100))
        else:
            st.warning("Please select both start and end dates")

with tab2:
    st.header("Washington Street Intersection Volumes")
    
    # Load volume data
    with st.spinner('Loading volume data...'):
        volume_df = load_volume_data()

    if volume_df.empty:
        st.error("Failed to load volume data.")
    else:
        # Get available intersections
        intersection_options = ["All Intersections"] + sorted(volume_df['intersection_name'].unique().tolist())

        # Sidebar for volume analysis
        with st.sidebar:
            st.title("🚦 Volume Controls")
            
            with st.expander("📊 Volume Filters", expanded=True):
                intersection = st.selectbox("Intersection", intersection_options)

                # Date range selector
                min_date = volume_df['local_datetime'].dt.date.min()
                max_date = volume_df['local_datetime'].dt.date.max()
                
                st.subheader("📅 Date Range")
                st.info(f"Available data: {min_date} to {max_date}")
                
                date_range_vol = st.date_input(
                    "Select Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="volume_date_range"
                )

                # Time granularity
                st.subheader("⏰ Data Granularity")
                granularity_vol = st.selectbox(
                    "Aggregation Level",
                    options=["Hourly", "Daily", "Weekly", "Monthly"],
                    index=0,
                    key="volume_granularity"
                )

                # Time filters for hourly data
                if granularity_vol == "Hourly":
                    time_filter_vol = st.selectbox(
                        "Time Period",
                        options=["All Hours", "Peak Hours (7-9 AM, 4-6 PM)", "AM Peak (7-9 AM)",
                                 "PM Peak (4-6 PM)", "Off-Peak", "Custom Range"],
                        key="volume_time_filter"
                    )

        # Process volume data
        if len(date_range_vol) == 2:
            # Filter data by selected intersection
            if intersection != "All Intersections":
                display_df = volume_df[volume_df['intersection_name'] == intersection].copy()
            else:
                display_df = volume_df.copy()

            # Process the data
            filtered_volume_data = process_traffic_data(
                display_df, date_range_vol, granularity_vol,
                time_filter_vol if granularity_vol == "Hourly" and 'time_filter_vol' in locals() else None
            )
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(filtered_volume_data):,}")
            with col2:
                st.metric("Intersections", filtered_volume_data['intersection_name'].nunique() if 'intersection_name' in filtered_volume_data.columns else 0)
            with col3:
                st.metric("Date Range", f"{len(filtered_volume_data['local_datetime'].dt.date.unique())} days")
            with col4:
                if 'total_volume' in filtered_volume_data.columns:
                    avg_volume = filtered_volume_data['total_volume'].mean()
                    st.metric("Avg Hourly Volume", f"{avg_volume:.0f} vehicles")
                else:
                    st.metric("Avg Volume", "N/A")
            
            # Data preview
            with st.expander("📊 Volume Data Preview"):
                st.dataframe(filtered_volume_data.head(100))
        else:
            st.warning("Please select both start and end dates")
