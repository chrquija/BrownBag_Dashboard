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

# Research Question
st.info("🔍 **Research Question**: What are the main bottlenecks (slowest intersections) on Washington St that are most prone to causing an increase in Travel Time?")

# Create tabs for different analyses
tab1, tab2 = st.tabs(["🛣️ Corridor Bottleneck Analysis", "🚦 Intersection Volume Impact"])

with tab1:
    st.header("Washington Street Corridor Bottleneck Analysis")
    
    # Load corridor data
    with st.spinner('Loading corridor performance data...'):
        corridor_df = load_traffic_data()

    if corridor_df.empty:
        st.error("Failed to load corridor data.")
    else:
        # Get available corridors
        corridor_options = ["All Segments"] + sorted(corridor_df['segment_name'].unique().tolist())

        # Sidebar for corridor analysis
        with st.sidebar:
            st.title("🛣️ Bottleneck Controls")
            
            with st.expander("📊 Analysis Filters", expanded=True):
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
            
            # BOTTLENECK-FOCUSED METRICS
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                max_delay = filtered_data['average_delay'].max()
                st.metric("Worst Delay", f"{max_delay:.1f} sec", 
                         help="Highest average delay recorded in selected timeframe")
            with col2:
                max_travel_time = filtered_data['average_traveltime'].max()
                st.metric("Longest Travel Time", f"{max_travel_time:.1f} min",
                         help="Highest average travel time recorded")
            with col3:
                min_speed = filtered_data['average_speed'].min()
                st.metric("Slowest Speed", f"{min_speed:.1f} mph",
                         help="Lowest average speed recorded (indicates congestion)")
            with col4:
                avg_delay = filtered_data['average_delay'].mean()
                st.metric("Average Delay", f"{avg_delay:.1f} sec",
                         help="Overall average delay across all segments")
            
            # BOTTLENECK IDENTIFICATION TABLE
            st.subheader("🚨 Top Bottleneck Segments (Ranked by Travel Time)")
            
            # Create bottleneck ranking
            bottleneck_summary = filtered_data.groupby(['segment_name', 'direction']).agg({
                'average_delay': ['mean', 'max'],
                'average_traveltime': ['mean', 'max'], 
                'average_speed': ['mean', 'min']
            }).round(2)
            
            # Flatten column names
            bottleneck_summary.columns = ['_'.join(col).strip() for col in bottleneck_summary.columns]
            bottleneck_summary = bottleneck_summary.reset_index()
            
            # Rename columns for clarity
            bottleneck_summary = bottleneck_summary.rename(columns={
                'average_delay_mean': 'Avg Delay (sec)',
                'average_delay_max': 'Max Delay (sec)',
                'average_traveltime_mean': 'Avg Travel Time (min)',
                'average_traveltime_max': 'Max Travel Time (min)',
                'average_speed_mean': 'Avg Speed (mph)',
                'average_speed_min': 'Min Speed (mph)'
            })
            
            # Sort by worst travel time
            bottleneck_summary = bottleneck_summary.sort_values('Max Travel Time (min)', ascending=False)
            
            # Color code the worst performers
            st.dataframe(
                bottleneck_summary.head(10),
                use_container_width=True
            )
            
            # DIRECTION COMPARISON
            st.subheader("🔄 Northbound vs Southbound Performance")
            
            direction_comparison = filtered_data.groupby('direction').agg({
                'average_delay': 'mean',
                'average_traveltime': 'mean',
                'average_speed': 'mean'
            }).round(2)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Northbound (NB)**")
                if 'NB' in direction_comparison.index:
                    nb_data = direction_comparison.loc['NB']
                    st.metric("NB Avg Delay", f"{nb_data['average_delay']:.1f} sec")
                    st.metric("NB Avg Travel Time", f"{nb_data['average_traveltime']:.1f} min")
                    st.metric("NB Avg Speed", f"{nb_data['average_speed']:.1f} mph")
            
            with col2:
                st.write("**Southbound (SB)**")
                if 'SB' in direction_comparison.index:
                    sb_data = direction_comparison.loc['SB']
                    st.metric("SB Avg Delay", f"{sb_data['average_delay']:.1f} sec")
                    st.metric("SB Avg Travel Time", f"{sb_data['average_traveltime']:.1f} min")
                    st.metric("SB Avg Speed", f"{sb_data['average_speed']:.1f} mph")
            
            # Data preview
            with st.expander("📊 Raw Performance Data"):
                st.dataframe(filtered_data.head(100))
        else:
            st.warning("Please select both start and end dates")

with tab2:
    st.header("Washington Street Intersection Volume Impact Analysis")
    
    # Load volume data
    with st.spinner('Loading intersection volume data...'):
        volume_df = load_volume_data()

    if volume_df.empty:
        st.error("Failed to load volume data.")
    else:
        # Get available intersections
        intersection_options = ["All Intersections"] + volume_df['intersection_name'].drop_duplicates().tolist()

        # Sidebar for volume analysis
        with st.sidebar:
            st.title("🚦 Volume Analysis Controls")
            
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

        # Process volume data
        if len(date_range_vol) == 2:
            # Filter data by selected intersection
            if intersection != "All Intersections":
                display_df = volume_df[volume_df['intersection_name'] == intersection].copy()
            else:
                display_df = volume_df.copy()

            # Process the data
            filtered_volume_data = process_traffic_data(
                display_df, date_range_vol, granularity_vol
            )
            
            # VOLUME-FOCUSED METRICS FOR BOTTLENECK ANALYSIS
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                max_volume = filtered_volume_data['total_volume'].max()
                st.metric("Peak Volume", f"{max_volume:,.0f} vehicles",
                         help="Highest hourly volume recorded (potential bottleneck indicator)")
            with col2:
                avg_volume = filtered_volume_data['total_volume'].mean()
                st.metric("Average Volume", f"{avg_volume:,.0f} vehicles",
                         help="Average hourly volume across timeframe")
            with col3:
                volume_std = filtered_volume_data['total_volume'].std()
                st.metric("Volume Variability", f"{volume_std:,.0f}",
                         help="Standard deviation - higher values indicate more congestion peaks")
            with col4:
                total_intersections = filtered_volume_data['intersection_name'].nunique() if 'intersection_name' in filtered_volume_data.columns else 0
                st.metric("Intersections Analyzed", f"{total_intersections}")
            
            # HIGHEST VOLUME INTERSECTIONS (BOTTLENECK CANDIDATES)
            st.subheader("🚨 Highest Volume Intersections (Potential Bottlenecks)")
            
            # Create volume ranking by intersection
            volume_summary = filtered_volume_data.groupby(['intersection_name', 'direction']).agg({
                'total_volume': ['mean', 'max', 'std']
            }).round(0)
            
            # Flatten column names
            volume_summary.columns = ['_'.join(col).strip() for col in volume_summary.columns]
            volume_summary = volume_summary.reset_index()
            
            # Rename columns
            volume_summary = volume_summary.rename(columns={
                'total_volume_mean': 'Avg Volume/Hour',
                'total_volume_max': 'Peak Volume/Hour',
                'total_volume_std': 'Volume Variability'
            })
            
            # Sort by peak volume (highest congestion potential)
            volume_summary = volume_summary.sort_values('Peak Volume/Hour', ascending=False)
            
            st.dataframe(
                volume_summary.head(10),
                use_container_width=True
            )
            
            # DIRECTION COMPARISON FOR VOLUMES
            st.subheader("🔄 Volume by Direction (Bottleneck Patterns)")
            
            volume_direction = filtered_volume_data.groupby('direction').agg({
                'total_volume': ['mean', 'max']
            }).round(0)
            
            volume_direction.columns = ['Average Volume', 'Peak Volume']
            volume_direction = volume_direction.reset_index()
            
            col1, col2 = st.columns(2)
            for idx, row in volume_direction.iterrows():
                direction = row['direction']
                avg_vol = row['Average Volume']
                peak_vol = row['Peak Volume']
                
                with col1 if idx == 0 else col2:
                    st.write(f"**{direction} Direction**")
                    st.metric(f"{direction} Avg Volume", f"{avg_vol:,.0f} veh/hr")
                    st.metric(f"{direction} Peak Volume", f"{peak_vol:,.0f} veh/hr")
            
            # Data preview
            with st.expander("📊 Raw Volume Data"):
                st.dataframe(filtered_volume_data.head(100))
        else:
            st.warning("Please select both start and end dates")
