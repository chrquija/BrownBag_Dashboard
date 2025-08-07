import streamlit as st
import pandas as pd
from sidebar_functions import process_traffic_data, load_traffic_data, load_volume_data
from datetime import datetime, timedelta
import numpy as np

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
tab1, tab2 = st.tabs(["🚧 Performance & Delay Analysis", "📊 Traffic Demand & Capacity Analysis"])

with tab1:
    st.header("🚧 Segment Performance & Travel Time Analysis")
    st.subheader("*Identifying segments with highest delays, slowest speeds, and longest travel times*")
    
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
            st.title("🚧 Performance Controls")
            
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
            
            # ADVANCED PERFORMANCE METRICS
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                max_delay = filtered_data['average_delay'].max()
                p95_delay = filtered_data['average_delay'].quantile(0.95)
                st.metric("Worst Delay", f"{max_delay:.1f} sec", 
                         delta=f"95th %tile: {p95_delay:.1f}s",
                         help="Highest delay vs 95th percentile")
            with col2:
                max_travel_time = filtered_data['average_traveltime'].max()
                median_travel_time = filtered_data['average_traveltime'].median()
                st.metric("Peak Travel Time", f"{max_travel_time:.1f} min",
                         delta=f"Median: {median_travel_time:.1f}m",
                         help="Peak vs typical travel time")
            with col3:
                min_speed = filtered_data['average_speed'].min()
                mean_speed = filtered_data['average_speed'].mean()
                st.metric("Congestion Speed", f"{min_speed:.1f} mph",
                         delta=f"Avg: {mean_speed:.1f} mph",
                         help="Slowest speed vs average")
            with col4:
                # Performance consistency (coefficient of variation for travel time)
                cv_traveltime = (filtered_data['average_traveltime'].std() / 
                               filtered_data['average_traveltime'].mean()) * 100
                st.metric("Travel Time Reliability", f"{cv_traveltime:.1f}%",
                         help="Lower % = more reliable (coefficient of variation)")
            with col5:
                # Speed drop severity
                speed_range = filtered_data['average_speed'].max() - filtered_data['average_speed'].min()
                st.metric("Speed Variability", f"{speed_range:.1f} mph",
                         help="Difference between fastest and slowest speeds")
            
            # ADVANCED BOTTLENECK IDENTIFICATION
            st.subheader("🚨 Critical Performance Analysis")
            
            # Create comprehensive performance scoring
            try:
                performance_analysis = filtered_data.groupby(['segment_name', 'direction']).agg({
                    'average_delay': ['mean', 'max', 'std'],
                    'average_traveltime': ['mean', 'max', 'std'], 
                    'average_speed': ['mean', 'min', 'std']
                }).round(2)
                
                # Flatten column names
                performance_analysis.columns = ['_'.join(col).strip() for col in performance_analysis.columns]
                performance_analysis = performance_analysis.reset_index()
                
                # Calculate performance scores (higher = worse)
                performance_analysis['Delay_Score'] = (
                    performance_analysis['average_delay_mean'] * 0.4 +
                    performance_analysis['average_delay_max'] * 0.4 +
                    performance_analysis['average_delay_std'] * 0.2
                )
                
                performance_analysis['TravelTime_Score'] = (
                    performance_analysis['average_traveltime_mean'] * 0.4 +
                    performance_analysis['average_traveltime_max'] * 0.4 +
                    performance_analysis['average_traveltime_std'] * 0.2
                )
                
                performance_analysis['Speed_Score'] = (
                    (60 - performance_analysis['average_speed_mean']) * 0.5 +  # Lower speed = higher score
                    (60 - performance_analysis['average_speed_min']) * 0.3 +
                    performance_analysis['average_speed_std'] * 0.2
                )
                
                # Combined bottleneck score
                performance_analysis['Bottleneck_Score'] = (
                    performance_analysis['Delay_Score'] * 0.35 +
                    performance_analysis['TravelTime_Score'] * 0.4 +
                    performance_analysis['Speed_Score'] * 0.25
                ).round(1)
                
                # Create final display table - only use columns that exist
                available_cols = performance_analysis.columns.tolist()
                display_cols = ['segment_name', 'direction', 'Bottleneck_Score']
                
                # Add columns if they exist
                perf_cols = ['average_delay_mean', 'average_delay_max', 
                           'average_traveltime_mean', 'average_traveltime_max',
                           'average_speed_mean', 'average_speed_min']
                
                for col in perf_cols:
                    if col in available_cols:
                        display_cols.append(col)
                
                final_performance = performance_analysis[display_cols].copy()
                
                # Rename columns for display
                rename_dict = {
                    'average_delay_mean': 'Avg Delay (sec)',
                    'average_delay_max': 'Peak Delay (sec)',
                    'average_traveltime_mean': 'Avg Travel Time (min)',
                    'average_traveltime_max': 'Peak Travel Time (min)',
                    'average_speed_mean': 'Avg Speed (mph)',
                    'average_speed_min': 'Slowest Speed (mph)'
                }
                
                final_performance = final_performance.rename(columns=rename_dict)
                
                # Sort by bottleneck score (worst first)
                final_performance = final_performance.sort_values('Bottleneck_Score', ascending=False)
                
                st.dataframe(
                    final_performance.head(10),
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Error in performance analysis: {str(e)}")
                # Fallback to simple analysis
                simple_performance = filtered_data.groupby(['segment_name', 'direction']).agg({
                    'average_delay': 'mean',
                    'average_traveltime': 'mean',
                    'average_speed': 'mean'
                }).round(2).reset_index()
                
                simple_performance = simple_performance.sort_values('average_traveltime', ascending=False)
                st.dataframe(simple_performance.head(10), use_container_width=True)
            
            # DIRECTIONAL COMPARISON
            st.subheader("🔄 Directional Performance Comparison")
            
            direction_analysis = filtered_data.groupby('direction').agg({
                'average_delay': 'mean',
                'average_traveltime': 'mean',
                'average_speed': 'mean'
            }).round(2)
            
            col1, col2 = st.columns(2)
            directions = direction_analysis.index.tolist()
            
            for idx, direction in enumerate(directions):
                with col1 if idx == 0 else col2:
                    st.write(f"**{direction} Direction Performance**")
                    
                    if direction in direction_analysis.index:
                        dir_data = direction_analysis.loc[direction]
                        st.metric(f"{direction} Avg Delay", f"{dir_data['average_delay']:.1f} sec")
                        st.metric(f"{direction} Avg Travel Time", f"{dir_data['average_traveltime']:.1f} min")
                        st.metric(f"{direction} Avg Speed", f"{dir_data['average_speed']:.1f} mph")
        else:
            st.warning("Please select both start and end dates")

with tab2:
    st.header("📊 Traffic Demand & Capacity Impact Analysis")
    st.subheader("*Analyzing volume patterns and intersection capacity utilization*")
    
    # Load volume data
    with st.spinner('Loading traffic demand data...'):
        volume_df = load_volume_data()

    if volume_df.empty:
        st.error("Failed to load volume data.")
    else:
        # Get available intersections (maintain order)
        intersection_options = ["All Intersections"] + volume_df['intersection_name'].drop_duplicates().tolist()

        # Sidebar for volume analysis
        with st.sidebar:
            st.title("📊 Demand Analysis Controls")
            
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
            
            # ADVANCED VOLUME METRICS
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                peak_volume = filtered_volume_data['total_volume'].max()
                p95_volume = filtered_volume_data['total_volume'].quantile(0.95)
                st.metric("Peak Demand", f"{peak_volume:,.0f} vph",
                         delta=f"95th: {p95_volume:,.0f}",
                         help="Peak vs 95th percentile volume")
            with col2:
                avg_volume = filtered_volume_data['total_volume'].mean()
                median_volume = filtered_volume_data['total_volume'].median()
                st.metric("Average Demand", f"{avg_volume:,.0f} vph",
                         delta=f"Median: {median_volume:,.0f}",
                         help="Mean vs median hourly volume")
            with col3:
                # Peak-to-average ratio
                peak_avg_ratio = peak_volume / avg_volume if avg_volume > 0 else 0
                st.metric("Peak/Avg Ratio", f"{peak_avg_ratio:.1f}x",
                         help="Higher ratio = more peaked demand")
            with col4:
                # Volume consistency (coefficient of variation)
                cv_volume = (filtered_volume_data['total_volume'].std() / 
                           filtered_volume_data['total_volume'].mean()) * 100 if avg_volume > 0 else 0
                st.metric("Demand Variability", f"{cv_volume:.1f}%",
                         help="Lower % = more consistent demand")
            with col5:
                # Estimate capacity utilization (assuming 1800 vph per lane capacity)
                estimated_capacity = 1800  # vehicles per hour per lane (typical)
                utilization = (peak_volume / estimated_capacity) * 100
                st.metric("Est. Capacity Use", f"{utilization:.0f}%",
                         help="Based on 1800 vph/lane capacity")
            
            # ADVANCED INTERSECTION RANKING
            st.subheader("🚨 High-Demand Intersection Analysis")
            
            try:
                # Create comprehensive volume analysis
                volume_analysis = filtered_volume_data.groupby(['intersection_name', 'direction']).agg({
                    'total_volume': ['mean', 'max', 'std']
                }).round(0)
                
                # Flatten column names
                volume_analysis.columns = ['_'.join(col).strip() for col in volume_analysis.columns]
                volume_analysis = volume_analysis.reset_index()
                
                # Calculate demand pressure scores
                volume_analysis['Peak_Pressure'] = volume_analysis['total_volume_max']
                volume_analysis['Avg_Pressure'] = volume_analysis['total_volume_mean']
                volume_analysis['Variability_Score'] = volume_analysis['total_volume_std']
                
                # Combined demand score (higher = more problematic)
                volume_analysis['Demand_Score'] = (
                    (volume_analysis['Peak_Pressure'] / 1800 * 100) * 0.4 +  # Peak capacity utilization
                    (volume_analysis['Avg_Pressure'] / 1000 * 100) * 0.3 +   # Average pressure
                    (volume_analysis['Variability_Score'] / 100) * 0.3       # Variability factor
                ).round(1)
                
                # Estimated congestion risk
                volume_analysis['Congestion_Risk'] = np.where(
                    volume_analysis['total_volume_max'] > 1500, 'HIGH',
                    np.where(volume_analysis['total_volume_max'] > 1200, 'MEDIUM', 'LOW')
                )
                
                # Create final display - only use available columns
                available_cols = volume_analysis.columns.tolist()
                base_cols = ['intersection_name', 'direction', 'Demand_Score', 'Congestion_Risk']
                
                vol_cols = ['total_volume_mean', 'total_volume_max', 'total_volume_std']
                display_cols = base_cols.copy()
                
                for col in vol_cols:
                    if col in available_cols:
                        display_cols.append(col)
                
                final_volume = volume_analysis[display_cols].copy()
                
                # Rename columns
                rename_dict = {
                    'total_volume_mean': 'Avg Volume (vph)',
                    'total_volume_max': 'Peak Volume (vph)',
                    'total_volume_std': 'Volume StdDev'
                }
                
                final_volume = final_volume.rename(columns=rename_dict)
                
                # Sort by demand score
                final_volume = final_volume.sort_values('Demand_Score', ascending=False)
                
                st.dataframe(
                    final_volume.head(10),
                    use_container_width=True
                )
                
                # CAPACITY RECOMMENDATIONS
                st.subheader("🔗 Capacity Management Recommendations")
                if utilization > 80:
                    st.error("🚨 Critical: Peak utilization >80% - Consider signal optimization or capacity improvements")
                elif utilization > 60:
                    st.warning("⚠️ Monitor: Peak utilization >60% - Plan for future improvements")
                else:
                    st.success("✅ Acceptable: Current capacity appears adequate for demand levels")
                
            except Exception as e:
                st.error(f"Error in volume analysis: {str(e)}")
                # Fallback to simple analysis
                simple_volume = filtered_volume_data.groupby(['intersection_name', 'direction']).agg({
                    'total_volume': ['mean', 'max']
                }).round(0).reset_index()
                
                simple_volume.columns = ['intersection_name', 'direction', 'Avg Volume', 'Peak Volume']
                simple_volume = simple_volume.sort_values('Peak Volume', ascending=False)
                st.dataframe(simple_volume.head(10), use_container_width=True)
            
            # Data preview
            with st.expander("📊 Raw Volume Data Sample"):
                st.dataframe(filtered_volume_data.head(100))
        else:
            st.warning("Please select both start and end dates")
