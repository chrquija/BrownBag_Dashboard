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

#Title of center analysis content
st.header("🛣️ Washington Street Corridor Analysis")

# Load both types of data with progress indicator
with st.spinner('Loading traffic data from all corridor segments and intersections...'):
    corridor_df = load_traffic_data()
    volume_df = load_volume_data()

if corridor_df.empty:
    st.error("Failed to load corridor data. Please check your connection.")
    st.stop()

if volume_df.empty:
    st.warning("Failed to load volume data, but corridor analysis will continue.")

# Get available options from both datasets
corridor_options = ["All Segments"] + sorted(corridor_df['segment_name'].unique().tolist())
intersection_options = ["All Intersections"]
if not volume_df.empty:
    intersection_options += sorted(volume_df['intersection_name'].unique().tolist())

# Sidebar
with st.sidebar:
    st.title("🛣️ Controls")

    # Data Type Selection
    with st.expander("📊 Data Type", expanded=True):
        data_type = st.selectbox(
            "Analysis Type",
            options=["Corridor Performance", "Intersection Volumes", "Combined Analysis"],
            help="Choose the type of analysis to perform"
        )

    # Filter section
    with st.expander("🔍 Data Filters", expanded=True):
        if data_type == "Corridor Performance":
            # Corridor selection
            corridor = st.selectbox("Corridor Segment", corridor_options)
            selected_df = corridor_df
            min_date = corridor_df['local_datetime'].dt.date.min()
            max_date = corridor_df['local_datetime'].dt.date.max()
            
        elif data_type == "Intersection Volumes":
            # Intersection selection
            intersection = st.selectbox("Intersection", intersection_options)
            selected_df = volume_df
            if not volume_df.empty:
                min_date = volume_df['local_datetime'].dt.date.min()
                max_date = volume_df['local_datetime'].dt.date.max()
            else:
                min_date = max_date = None
                
        else:  # Combined Analysis
            corridor = st.selectbox("Corridor Segment", corridor_options)
            intersection = st.selectbox("Intersection", intersection_options)
            # Use corridor data as primary for date range
            selected_df = corridor_df
            min_date = corridor_df['local_datetime'].dt.date.min()
            max_date = corridor_df['local_datetime'].dt.date.max()

        # Date range selector
        if min_date and max_date:
            st.subheader("📅 Date Range")
            st.info(f"Available data: {min_date} to {max_date}")
            
            date_range = st.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                help="Choose start and end dates for analysis"
            )
        else:
            st.error("No date range available")
            date_range = ()

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

# Main content based on data type selection
if len(date_range) == 2:
    
    if data_type == "Corridor Performance":
        st.subheader("📈 Corridor Performance Analysis")
        
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
        
        # Display corridor metrics
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
    
    elif data_type == "Intersection Volumes":
        st.subheader("🚦 Intersection Volume Analysis")
        
        if volume_df.empty:
            st.error("Volume data not available")
        else:
            # Filter data by selected intersection
            if intersection != "All Intersections":
                display_df = volume_df[volume_df['intersection_name'] == intersection].copy()
            else:
                display_df = volume_df.copy()

            # Process the volume data
            filtered_data = process_traffic_data(
                display_df, date_range, granularity,
                time_filter if granularity == "Hourly" and 'time_filter' in locals() else None,
                start_hour if 'start_hour' in locals() else None,
                end_hour if 'end_hour' in locals() else None
            )
            
            # Display volume metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(filtered_data):,}")
            with col2:
                st.metric("Intersections", filtered_data['intersection_name'].nunique() if 'intersection_name' in filtered_data.columns else 0)
            with col3:
                st.metric("Date Range", f"{len(filtered_data['local_datetime'].dt.date.unique())} days")
            with col4:
                if 'total_volume' in filtered_data.columns:
                    avg_volume = filtered_data['total_volume'].mean()
                    st.metric("Avg Hourly Volume", f"{avg_volume:.0f} vehicles")
                else:
                    st.metric("Avg Volume", "N/A")
    
    else:  # Combined Analysis
        st.subheader("🔄 Combined Corridor & Volume Analysis")
        
        # Process corridor data
        if corridor != "All Segments":
            corridor_display_df = corridor_df[corridor_df['segment_name'] == corridor].copy()
        else:
            corridor_display_df = corridor_df.copy()

        filtered_corridor_data = process_traffic_data(
            corridor_display_df, date_range, granularity,
            time_filter if granularity == "Hourly" and 'time_filter' in locals() else None,
            start_hour if 'start_hour' in locals() else None,
            end_hour if 'end_hour' in locals() else None
        )
        
        # Process volume data
        if not volume_df.empty:
            if intersection != "All Intersections":
                volume_display_df = volume_df[volume_df['intersection_name'] == intersection].copy()
            else:
                volume_display_df = volume_df.copy()

            filtered_volume_data = process_traffic_data(
                volume_display_df, date_range, granularity,
                time_filter if granularity == "Hourly" and 'time_filter' in locals() else None,
                start_hour if 'start_hour' in locals() else None,
                end_hour if 'end_hour' in locals() else None
            )
        else:
            filtered_volume_data = pd.DataFrame()
        
        # Display combined metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            corridor_records = len(filtered_corridor_data)
            volume_records = len(filtered_volume_data) if not filtered_volume_data.empty else 0
            st.metric("Total Records", f"{corridor_records + volume_records:,}")
        with col2:
            st.metric("Corridor Segments", filtered_corridor_data['segment_name'].nunique())
        with col3:
            volume_intersections = filtered_volume_data['intersection_name'].nunique() if not filtered_volume_data.empty and 'intersection_name' in filtered_volume_data.columns else 0
            st.metric("Volume Intersections", volume_intersections)
        with col4:
            avg_speed = filtered_corridor_data['average_speed'].mean()
            st.metric("Avg Speed", f"{avg_speed:.1f} mph")

    # Data Preview Section
    with st.expander("📊 Data Preview", expanded=False):
        if data_type == "Corridor Performance":
            st.subheader("Corridor Data Sample")
            st.dataframe(filtered_data.head(100))
        elif data_type == "Intersection Volumes":
            st.subheader("Volume Data Sample")
            st.dataframe(filtered_data.head(100))
        else:  # Combined
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Corridor Data Sample")
                st.dataframe(filtered_corridor_data.head(50))
            with col2:
                st.subheader("Volume Data Sample")
                if not filtered_volume_data.empty:
                    st.dataframe(filtered_volume_data.head(50))
                else:
                    st.info("No volume data available")
        
else:
    st.warning("Please select both start and end dates")
