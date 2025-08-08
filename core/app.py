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

# Custom CSS for blue color scheme and dark mode compatibility
st.markdown("""
<style>
    .context-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin: 1rem 0 2rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(30,58,138,0.2);
    }

    .context-header h2 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }

    .context-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Dark mode compatibility */
    @media (prefers-color-scheme: dark) {
        .context-header {
            background: linear-gradient(90deg, #1e40af 0%, #2563eb 100%);
        }
    }

    .comparison-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: white;
        text-align: center;
    }

    .insight-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(37, 99, 235, 0.1) 100%);
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .performance-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0 0.3rem;
    }

    .badge-excellent { background: #dbeafe; color: #1e40af; }
    .badge-good { background: #bfdbfe; color: #1d4ed8; }
    .badge-fair { background: #fbbf24; color: #92400e; }
    .badge-poor { background: #fca5a5; color: #991b1b; }
    .badge-critical { background: #ef4444; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# Main Title
st.title("🛣️ Active Transportation & Operations Management Dashboard")

# Dashboard objective
dashboard_objective = """
<div style="
    font-size: 1.15rem;
    font-weight: 400;
    color: var(--text-color);
    background: var(--background-color);
    padding: 1.2rem 1.5rem;
    border-radius: 14px;
    box-shadow: 0 2px 16px 0 var(--shadow-color, rgba(59,130,246,0.1));
    margin-bottom: 2rem;
    line-height: 1.7;
    border-left: 4px solid #3b82f6;
    ">
    <b>The ADVANTEC App</b> provides traffic engineering recommendations for the Coachella Valley using <b>MILLIONS OF DATA POINTS trained on Machine Learning Algorithms to REDUCE Travel Time, Fuel Consumption, and Green House Gases.</b> This is accomplished through the identification of anomalies, provision of cycle length recommendations, and predictive modeling.
</div>
"""

st.markdown(dashboard_objective, unsafe_allow_html=True)

# Research Question
st.info(
    "🔍 **Research Question**: What are the main bottlenecks (slowest intersections) on Washington St that are most prone to causing an increase in Travel Time?")

# Create tabs for different analyses
tab1, tab2 = st.tabs(["🚧 Performance & Delay Analysis", "📊 Traffic Demand & Capacity Analysis"])

with tab1:
    st.header("🚧 Segment Performance & Travel Time Analysis")

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

            # Add comparison mode toggle
            comparison_mode = st.checkbox("📊 Enable Month-over-Month Comparison", value=False)

            with st.expander("📊 Analysis Filters", expanded=True):
                corridor = st.selectbox("Corridor Segment", corridor_options)

                # Date range selector
                min_date = corridor_df['local_datetime'].dt.date.min()
                max_date = corridor_df['local_datetime'].dt.date.max()

                st.subheader("📅 Date Range")
                st.info(f"Available data: {min_date} to {max_date}")

                if comparison_mode:
                    # Two separate date ranges for comparison
                    st.write("**Primary Period:**")
                    date_range_1 = st.date_input(
                        "Select Primary Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        key="primary_range"
                    )

                    st.write("**Comparison Period:**")
                    date_range_2 = st.date_input(
                        "Select Comparison Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        key="comparison_range"
                    )
                else:
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
        if comparison_mode and len(date_range_1) == 2 and len(date_range_2) == 2:
            # COMPARISON MODE
            # Filter data by selected corridor
            if corridor != "All Segments":
                display_df = corridor_df[corridor_df['segment_name'] == corridor].copy()
            else:
                display_df = corridor_df.copy()

            # Process data for both periods
            filtered_data_1 = process_traffic_data(
                display_df, date_range_1, granularity,
                time_filter if granularity == "Hourly" and 'time_filter' in locals() else None,
                start_hour if 'start_hour' in locals() else None,
                end_hour if 'end_hour' in locals() else None
            )

            filtered_data_2 = process_traffic_data(
                display_df, date_range_2, granularity,
                time_filter if granularity == "Hourly" and 'time_filter' in locals() else None,
                start_hour if 'start_hour' in locals() else None,
                end_hour if 'end_hour' in locals() else None
            )

            # Simplified context headers for comparison
            context_header_1 = f"""
            <div class="comparison-header">
                <h3>📊 Primary: {corridor} | {date_range_1[0].strftime('%b %d')} - {date_range_1[1].strftime('%b %d, %Y')} | {len(filtered_data_1):,} points</h3>
            </div>
            """

            context_header_2 = f"""
            <div class="comparison-header">
                <h3>📊 Comparison: {corridor} | {date_range_2[0].strftime('%b %d')} - {date_range_2[1].strftime('%b %d, %Y')} | {len(filtered_data_2):,} points</h3>
            </div>
            """

            # Side-by-side comparison
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(context_header_1, unsafe_allow_html=True)

                # Calculate metrics for period 1
                raw_data_1 = display_df[(display_df['local_datetime'].dt.date >= date_range_1[0]) &
                                        (display_df['local_datetime'].dt.date <= date_range_1[1])]

                worst_delay_1 = raw_data_1['average_delay'].max()
                worst_travel_time_1 = raw_data_1['average_traveltime'].max()
                slowest_speed_1 = raw_data_1['average_speed'].min()
                cv_traveltime_1 = (raw_data_1['average_traveltime'].std() / raw_data_1[
                    'average_traveltime'].mean()) * 100
                reliability_score_1 = max(0, 100 - cv_traveltime_1)

                st.metric("🚨 Peak Delay", f"{worst_delay_1:.1f}s")
                st.metric("⏱️ Peak Travel Time", f"{worst_travel_time_1:.1f}min")
                st.metric("🐌 Congestion Speed", f"{slowest_speed_1:.1f}mph")
                st.metric("🎯 Travel Reliability", f"{reliability_score_1:.0f}%")

            with col2:
                st.markdown(context_header_2, unsafe_allow_html=True)

                # Calculate metrics for period 2
                raw_data_2 = display_df[(display_df['local_datetime'].dt.date >= date_range_2[0]) &
                                        (display_df['local_datetime'].dt.date <= date_range_2[1])]

                worst_delay_2 = raw_data_2['average_delay'].max()
                worst_travel_time_2 = raw_data_2['average_traveltime'].max()
                slowest_speed_2 = raw_data_2['average_speed'].min()
                cv_traveltime_2 = (raw_data_2['average_traveltime'].std() / raw_data_2[
                    'average_traveltime'].mean()) * 100
                reliability_score_2 = max(0, 100 - cv_traveltime_2)

                # Calculate deltas
                delay_delta = worst_delay_2 - worst_delay_1
                travel_delta = worst_travel_time_2 - worst_travel_time_1
                speed_delta = slowest_speed_2 - slowest_speed_1
                reliability_delta = reliability_score_2 - reliability_score_1

                st.metric("🚨 Peak Delay", f"{worst_delay_2:.1f}s", delta=f"{delay_delta:+.1f}s")
                st.metric("⏱️ Peak Travel Time", f"{worst_travel_time_2:.1f}min", delta=f"{travel_delta:+.1f}min")
                st.metric("🐌 Congestion Speed", f"{slowest_speed_2:.1f}mph", delta=f"{speed_delta:+.1f}mph")
                st.metric("🎯 Travel Reliability", f"{reliability_score_2:.0f}%", delta=f"{reliability_delta:+.0f}%")

            # Comparison insights
            comparison_insight = f"""
            <div class="insight-box">
                <h4>📈 Period Comparison Analysis</h4>
                <p><strong>Performance Change:</strong> Peak delays {"increased" if delay_delta > 0 else "decreased"} by {abs(delay_delta):.1f} seconds, 
                while travel reliability {"improved" if reliability_delta > 0 else "declined"} by {abs(reliability_delta):.0f} percentage points.</p>
                <p><strong>Traffic Conditions:</strong> The comparison period shows {"worse" if travel_delta > 0 else "better"} overall performance with 
                {"higher" if travel_delta > 0 else "lower"} peak travel times.</p>
            </div>
            """
            st.markdown(comparison_insight, unsafe_allow_html=True)

        elif not comparison_mode and len(date_range) == 2:
            # SINGLE PERIOD MODE
            if corridor != "All Segments":
                display_df = corridor_df[corridor_df['segment_name'] == corridor].copy()
            else:
                display_df = corridor_df.copy()

            filtered_data = process_traffic_data(
                display_df, date_range, granularity,
                time_filter if granularity == "Hourly" and 'time_filter' in locals() else None,
                start_hour if 'start_hour' in locals() else None,
                end_hour if 'end_hour' in locals() else None
            )

            # Simplified context header
            context_header = f"""
            <div class="context-header">
                <h2>📊 {corridor}</h2>
                <p>{date_range[0].strftime('%B %d, %Y')} to {date_range[1].strftime('%B %d, %Y')} • {len(filtered_data):,} data points</p>
            </div>
            """
            st.markdown(context_header, unsafe_allow_html=True)

            # Use raw hourly data for meaningful insights
            raw_data = display_df[(display_df['local_datetime'].dt.date >= date_range[0]) &
                                  (display_df['local_datetime'].dt.date <= date_range[1])]

            # ENHANCED PERFORMANCE METRICS
            st.subheader("🎯 Key Performance Indicators")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                worst_delay = raw_data['average_delay'].max()
                p95_delay = raw_data['average_delay'].quantile(0.95)

                if worst_delay > 120:
                    badge_class = "badge-critical"
                elif worst_delay > 90:
                    badge_class = "badge-poor"
                elif worst_delay > 60:
                    badge_class = "badge-fair"
                else:
                    badge_class = "badge-good"

                st.metric("🚨 Peak Delay", f"{worst_delay:.1f}s",
                          delta=f"95th: {p95_delay:.1f}s")
                st.markdown(
                    f'<span class="performance-badge {badge_class}">{"Critical" if worst_delay > 120 else "Poor" if worst_delay > 90 else "Fair" if worst_delay > 60 else "Good"}</span>',
                    unsafe_allow_html=True)

            with col2:
                worst_travel_time = raw_data['average_traveltime'].max()
                avg_travel_time = raw_data['average_traveltime'].mean()
                travel_increase = (
                            (worst_travel_time - avg_travel_time) / avg_travel_time * 100) if avg_travel_time > 0 else 0

                st.metric("⏱️ Peak Travel Time", f"{worst_travel_time:.1f}min",
                          delta=f"+{travel_increase:.0f}% vs avg")

                if travel_increase > 100:
                    badge_class = "badge-critical"
                elif travel_increase > 50:
                    badge_class = "badge-poor"
                elif travel_increase > 25:
                    badge_class = "badge-fair"
                else:
                    badge_class = "badge-good"

                st.markdown(
                    f'<span class="performance-badge {badge_class}">{"Critical" if travel_increase > 100 else "High" if travel_increase > 50 else "Moderate" if travel_increase > 25 else "Low"} Impact</span>',
                    unsafe_allow_html=True)

            with col3:
                slowest_speed = raw_data['average_speed'].min()
                avg_speed = raw_data['average_speed'].mean()
                speed_drop = ((avg_speed - slowest_speed) / avg_speed * 100) if avg_speed > 0 else 0

                st.metric("🐌 Congestion Speed", f"{slowest_speed:.1f}mph",
                          delta=f"-{speed_drop:.0f}% vs avg")

                if slowest_speed < 15:
                    badge_class = "badge-critical"
                elif slowest_speed < 25:
                    badge_class = "badge-poor"
                elif slowest_speed < 35:
                    badge_class = "badge-fair"
                else:
                    badge_class = "badge-good"

                st.markdown(
                    f'<span class="performance-badge {badge_class}">{"Severe" if slowest_speed < 15 else "Heavy" if slowest_speed < 25 else "Moderate" if slowest_speed < 35 else "Light"} Congestion</span>',
                    unsafe_allow_html=True)

            with col4:
                cv_traveltime = (raw_data['average_traveltime'].std() /
                                 raw_data['average_traveltime'].mean()) * 100 if raw_data[
                                                                                     'average_traveltime'].mean() > 0 else 0
                reliability_score = max(0, 100 - cv_traveltime)

                st.metric("🎯 Travel Reliability", f"{reliability_score:.0f}%",
                          delta=f"CV: {cv_traveltime:.1f}%")

                if reliability_score > 80:
                    badge_class = "badge-excellent"
                elif reliability_score > 60:
                    badge_class = "badge-good"
                elif reliability_score > 40:
                    badge_class = "badge-fair"
                else:
                    badge_class = "badge-poor"

                st.markdown(
                    f'<span class="performance-badge {badge_class}">{"Excellent" if reliability_score > 80 else "Good" if reliability_score > 60 else "Fair" if reliability_score > 40 else "Poor"}</span>',
                    unsafe_allow_html=True)

            with col5:
                high_delay_pct = (raw_data['average_delay'] > 60).mean() * 100

                st.metric("⚠️ Congestion Frequency", f"{high_delay_pct:.1f}%",
                          delta=f"{(raw_data['average_delay'] > 60).sum()} hours")

                if high_delay_pct > 30:
                    badge_class = "badge-critical"
                elif high_delay_pct > 20:
                    badge_class = "badge-poor"
                elif high_delay_pct > 10:
                    badge_class = "badge-fair"
                else:
                    badge_class = "badge-good"

                st.markdown(
                    f'<span class="performance-badge {badge_class}">{"Very High" if high_delay_pct > 30 else "High" if high_delay_pct > 20 else "Moderate" if high_delay_pct > 10 else "Low"}</span>',
                    unsafe_allow_html=True)

            # Performance insight
            insight_text = f"""
            <div class="insight-box">
                <h4>💡 Analysis Insights</h4>
                <p><strong>Performance Summary:</strong> Peak delays reach {worst_delay:.0f} seconds with travel times varying up to {travel_increase:.0f}% above average during congestion periods.</p>
                <p><strong>Reliability:</strong> This corridor demonstrates {reliability_score:.0f}% travel time reliability, experiencing significant delays {high_delay_pct:.1f}% of the time.</p>
            </div>
            """
            st.markdown(insight_text, unsafe_allow_html=True)

        else:
            st.warning("Please select date ranges for analysis")

with tab2:
    st.header("📊 Traffic Demand & Capacity Impact Analysis")

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

            # Add comparison mode toggle for volume tab
            comparison_mode_vol = st.checkbox("📊 Enable Month-over-Month Comparison", value=False, key="vol_comparison")

            with st.expander("📊 Volume Filters", expanded=True):
                intersection = st.selectbox("Intersection", intersection_options)

                # Date range selector
                min_date = volume_df['local_datetime'].dt.date.min()
                max_date = volume_df['local_datetime'].dt.date.max()

                st.subheader("📅 Date Range")
                st.info(f"Available data: {min_date} to {max_date}")

                if comparison_mode_vol:
                    # Two separate date ranges for comparison
                    st.write("**Primary Period:**")
                    date_range_vol_1 = st.date_input(
                        "Select Primary Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        key="vol_primary_range"
                    )

                    st.write("**Comparison Period:**")
                    date_range_vol_2 = st.date_input(
                        "Select Comparison Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        key="vol_comparison_range"
                    )
                else:
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
        if comparison_mode_vol and len(date_range_vol_1) == 2 and len(date_range_vol_2) == 2:
            # VOLUME COMPARISON MODE
            if intersection != "All Intersections":
                display_df = volume_df[volume_df['intersection_name'] == intersection].copy()
            else:
                display_df = volume_df.copy()

            # Process data for both periods
            filtered_volume_data_1 = process_traffic_data(display_df, date_range_vol_1, granularity_vol)
            filtered_volume_data_2 = process_traffic_data(display_df, date_range_vol_2, granularity_vol)

            # Simplified context headers
            context_header_vol_1 = f"""
            <div class="comparison-header">
                <h3>📊 Primary: {intersection} | {date_range_vol_1[0].strftime('%b %d')} - {date_range_vol_1[1].strftime('%b %d, %Y')} | {len(filtered_volume_data_1):,} points</h3>
            </div>
            """

            context_header_vol_2 = f"""
            <div class="comparison-header">
                <h3>📊 Comparison: {intersection} | {date_range_vol_2[0].strftime('%b %d')} - {date_range_vol_2[1].strftime('%b %d, %Y')} | {len(filtered_volume_data_2):,} points</h3>
            </div>
            """

            # Side-by-side volume comparison
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(context_header_vol_1, unsafe_allow_html=True)

                raw_vol_1 = display_df[(display_df['local_datetime'].dt.date >= date_range_vol_1[0]) &
                                       (display_df['local_datetime'].dt.date <= date_range_vol_1[1])]

                peak_vol_1 = raw_vol_1['total_volume'].max()
                avg_vol_1 = raw_vol_1['total_volume'].mean()
                capacity_util_1 = (peak_vol_1 / 1800) * 100

                st.metric("🔥 Peak Demand", f"{peak_vol_1:,.0f} vph")
                st.metric("📊 Average Demand", f"{avg_vol_1:,.0f} vph")
                st.metric("📈 Peak Capacity %", f"{capacity_util_1:.0f}%")

            with col2:
                st.markdown(context_header_vol_2, unsafe_allow_html=True)

                raw_vol_2 = display_df[(display_df['local_datetime'].dt.date >= date_range_vol_2[0]) &
                                       (display_df['local_datetime'].dt.date <= date_range_vol_2[1])]

                peak_vol_2 = raw_vol_2['total_volume'].max()
                avg_vol_2 = raw_vol_2['total_volume'].mean()
                capacity_util_2 = (peak_vol_2 / 1800) * 100

                # Calculate deltas
                peak_delta = peak_vol_2 - peak_vol_1
                avg_delta = avg_vol_2 - avg_vol_1
                capacity_delta = capacity_util_2 - capacity_util_1

                st.metric("🔥 Peak Demand", f"{peak_vol_2:,.0f} vph", delta=f"{peak_delta:+,.0f}")
                st.metric("📊 Average Demand", f"{avg_vol_2:,.0f} vph", delta=f"{avg_delta:+,.0f}")
                st.metric("📈 Peak Capacity %", f"{capacity_util_2:.0f}%", delta=f"{capacity_delta:+.0f}%")

        elif not comparison_mode_vol and len(date_range_vol) == 2:
            # SINGLE PERIOD VOLUME MODE
            if intersection != "All Intersections":
                display_df = volume_df[volume_df['intersection_name'] == intersection].copy()
            else:
                display_df = volume_df.copy()

            filtered_volume_data = process_traffic_data(display_df, date_range_vol, granularity_vol)

            # Simplified context header for volume
            context_header_vol = f"""
            <div class="context-header">
                <h2>📊 {intersection}</h2>
                <p>{date_range_vol[0].strftime('%B %d, %Y')} to {date_range_vol[1].strftime('%B %d, %Y')} • {len(filtered_volume_data):,} data points</p>
            </div>
            """
            st.markdown(context_header_vol, unsafe_allow_html=True)

            # Use raw hourly data
            raw_volume_data = display_df[(display_df['local_datetime'].dt.date >= date_range_vol[0]) &
                                         (display_df['local_datetime'].dt.date <= date_range_vol[1])]

            # ENHANCED VOLUME METRICS
            st.subheader("🚦 Traffic Demand Indicators")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                peak_volume = raw_volume_data['total_volume'].max()
                p95_volume = raw_volume_data['total_volume'].quantile(0.95)
                capacity_util = (peak_volume / 1800) * 100

                if capacity_util > 90:
                    badge_class = "badge-critical"
                elif capacity_util > 75:
                    badge_class = "badge-poor"
                elif capacity_util > 60:
                    badge_class = "badge-fair"
                else:
                    badge_class = "badge-good"

                st.metric("🔥 Peak Demand", f"{peak_volume:,.0f} vph",
                          delta=f"95th: {p95_volume:,.0f}")
                st.markdown(f'<span class="performance-badge {badge_class}">{capacity_util:.0f}% Capacity</span>',
                            unsafe_allow_html=True)

            with col2:
                avg_volume = raw_volume_data['total_volume'].mean()
                median_volume = raw_volume_data['total_volume'].median()

                st.metric("📊 Average Demand", f"{avg_volume:,.0f} vph",
                          delta=f"Median: {median_volume:,.0f}")

                avg_util = (avg_volume / 1800) * 100
                if avg_util > 60:
                    badge_class = "badge-poor"
                elif avg_util > 40:
                    badge_class = "badge-fair"
                else:
                    badge_class = "badge-good"

                st.markdown(f'<span class="performance-badge {badge_class}">{avg_util:.0f}% Avg Util</span>',
                            unsafe_allow_html=True)

            with col3:
                peak_avg_ratio = peak_volume / avg_volume if avg_volume > 0 else 0

                st.metric("📈 Peak/Average Ratio", f"{peak_avg_ratio:.1f}x")

                if peak_avg_ratio > 3:
                    badge_class = "badge-poor"
                elif peak_avg_ratio > 2:
                    badge_class = "badge-fair"
                else:
                    badge_class = "badge-good"

                st.markdown(
                    f'<span class="performance-badge {badge_class}">{"High" if peak_avg_ratio > 3 else "Moderate" if peak_avg_ratio > 2 else "Low"} Peaking</span>',
                    unsafe_allow_html=True)

            with col4:
                cv_volume = (raw_volume_data['total_volume'].std() /
                             raw_volume_data['total_volume'].mean()) * 100 if avg_volume > 0 else 0

                st.metric("🎯 Demand Consistency", f"{100 - cv_volume:.0f}%",
                          delta=f"CV: {cv_volume:.1f}%")

                if cv_volume < 30:
                    badge_class = "badge-good"
                elif cv_volume < 50:
                    badge_class = "badge-fair"
                else:
                    badge_class = "badge-poor"

                st.markdown(
                    f'<span class="performance-badge {badge_class}">{"Consistent" if cv_volume < 30 else "Variable" if cv_volume < 50 else "Highly Variable"}</span>',
                    unsafe_allow_html=True)

            with col5:
                high_volume_hours = (raw_volume_data['total_volume'] > 1200).sum()
                total_hours = len(raw_volume_data)
                risk_pct = (high_volume_hours / total_hours * 100) if total_hours > 0 else 0

                st.metric("⚠️ High Volume Hours", f"{high_volume_hours}",
                          delta=f"{risk_pct:.1f}% of time")

                if risk_pct > 25:
                    badge_class = "badge-critical"
                elif risk_pct > 15:
                    badge_class = "badge-poor"
                elif risk_pct > 5:
                    badge_class = "badge-fair"
                else:
                    badge_class = "badge-good"

                st.markdown(
                    f'<span class="performance-badge {badge_class}">{"Very High" if risk_pct > 25 else "High" if risk_pct > 15 else "Moderate" if risk_pct > 5 else "Low"} Risk</span>',
                    unsafe_allow_html=True)

        else:
            st.warning("Please select date ranges for volume analysis")