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

# Custom CSS for dark mode compatibility and enhanced styling
st.markdown("""
<style>
    .context-header {
        background: linear-gradient(90deg, var(--primary-color, #ff6b6b) 0%, var(--secondary-color, #4ecdc4) 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin: 1rem 0 2rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
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
            background: linear-gradient(90deg, #c73650 0%, #37a69b 100%);
        }
    }
    
    .metric-container {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(78, 205, 196, 0.1) 0%, rgba(255, 107, 107, 0.1) 100%);
        border-left: 4px solid var(--primary-color, #4ecdc4);
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
    
    .badge-excellent { background: #d4edda; color: #155724; }
    .badge-good { background: #d1ecf1; color: #0c5460; }
    .badge-fair { background: #fff3cd; color: #856404; }
    .badge-poor { background: #f8d7da; color: #721c24; }
    .badge-critical { background: #f5c6cb; color: #491217; }
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
    box-shadow: 0 2px 16px 0 var(--shadow-color, rgba(0,0,0,0.06));
    margin-bottom: 2rem;
    line-height: 1.7;
    border-left: 4px solid #4ecdc4;
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
                    index=0,
                    help="Note: Higher aggregation levels will show averaged values and may reduce metric variability"
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
            
            # Create context header
            total_records = len(filtered_data)
            data_span = (date_range[1] - date_range[0]).days + 1
            
            time_context = ""
            if granularity == "Hourly" and 'time_filter' in locals():
                time_context = f" • {time_filter}"
            
            context_header = f"""
            <div class="context-header">
                <h2>📊 Performance Analysis: {corridor}</h2>
                <p>📅 {date_range[0].strftime('%B %d, %Y')} to {date_range[1].strftime('%B %d, %Y')} ({data_span} days) • 
                {granularity} Aggregation{time_context} • {total_records:,} data points</p>
            </div>
            """
            st.markdown(context_header, unsafe_allow_html=True)
            
            # ENHANCED PERFORMANCE METRICS with better context
            st.subheader("🎯 Key Performance Indicators")
            
            # Calculate metrics based on raw hourly data for more meaningful insights
            raw_data = display_df[(display_df['local_datetime'].dt.date >= date_range[0]) & 
                                 (display_df['local_datetime'].dt.date <= date_range[1])]
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                worst_delay = raw_data['average_delay'].max()
                p95_delay = raw_data['average_delay'].quantile(0.95)
                
                # Performance classification
                if worst_delay > 120: badge_class = "badge-critical"
                elif worst_delay > 90: badge_class = "badge-poor"
                elif worst_delay > 60: badge_class = "badge-fair"
                else: badge_class = "badge-good"
                
                st.metric("🚨 Peak Delay", f"{worst_delay:.1f}s", 
                         delta=f"95th: {p95_delay:.1f}s")
                st.markdown(f'<span class="performance-badge {badge_class}">{"Critical" if worst_delay > 120 else "Poor" if worst_delay > 90 else "Fair" if worst_delay > 60 else "Good"}</span>', 
                           unsafe_allow_html=True)
            
            with col2:
                worst_travel_time = raw_data['average_traveltime'].max()
                avg_travel_time = raw_data['average_traveltime'].mean()
                travel_increase = ((worst_travel_time - avg_travel_time) / avg_travel_time * 100) if avg_travel_time > 0 else 0
                
                st.metric("⏱️ Peak Travel Time", f"{worst_travel_time:.1f}min", 
                         delta=f"+{travel_increase:.0f}% vs avg")
                
                if travel_increase > 100: badge_class = "badge-critical"
                elif travel_increase > 50: badge_class = "badge-poor"
                elif travel_increase > 25: badge_class = "badge-fair"
                else: badge_class = "badge-good"
                
                st.markdown(f'<span class="performance-badge {badge_class}">{"Critical" if travel_increase > 100 else "High" if travel_increase > 50 else "Moderate" if travel_increase > 25 else "Low"} Impact</span>', 
                           unsafe_allow_html=True)
            
            with col3:
                slowest_speed = raw_data['average_speed'].min()
                avg_speed = raw_data['average_speed'].mean()
                speed_drop = ((avg_speed - slowest_speed) / avg_speed * 100) if avg_speed > 0 else 0
                
                st.metric("🐌 Congestion Speed", f"{slowest_speed:.1f}mph", 
                         delta=f"-{speed_drop:.0f}% vs avg")
                
                if slowest_speed < 15: badge_class = "badge-critical"
                elif slowest_speed < 25: badge_class = "badge-poor"
                elif slowest_speed < 35: badge_class = "badge-fair"
                else: badge_class = "badge-good"
                
                st.markdown(f'<span class="performance-badge {badge_class}">{"Severe" if slowest_speed < 15 else "Heavy" if slowest_speed < 25 else "Moderate" if slowest_speed < 35 else "Light"} Congestion</span>', 
                           unsafe_allow_html=True)
            
            with col4:
                # Reliability score (inverse of coefficient of variation)
                cv_traveltime = (raw_data['average_traveltime'].std() / 
                               raw_data['average_traveltime'].mean()) * 100 if raw_data['average_traveltime'].mean() > 0 else 0
                reliability_score = max(0, 100 - cv_traveltime)
                
                st.metric("🎯 Travel Reliability", f"{reliability_score:.0f}%", 
                         delta=f"CV: {cv_traveltime:.1f}%")
                
                if reliability_score > 80: badge_class = "badge-excellent"
                elif reliability_score > 60: badge_class = "badge-good"
                elif reliability_score > 40: badge_class = "badge-fair"
                else: badge_class = "badge-poor"
                
                st.markdown(f'<span class="performance-badge {badge_class}">{"Excellent" if reliability_score > 80 else "Good" if reliability_score > 60 else "Fair" if reliability_score > 40 else "Poor"}</span>', 
                           unsafe_allow_html=True)
            
            with col5:
                # Congestion frequency (% of time with delays > 60s)
                high_delay_pct = (raw_data['average_delay'] > 60).mean() * 100
                
                st.metric("⚠️ Congestion Frequency", f"{high_delay_pct:.1f}%", 
                         delta=f"{(raw_data['average_delay'] > 60).sum()} hours")
                
                if high_delay_pct > 30: badge_class = "badge-critical"
                elif high_delay_pct > 20: badge_class = "badge-poor"
                elif high_delay_pct > 10: badge_class = "badge-fair"
                else: badge_class = "badge-good"
                
                st.markdown(f'<span class="performance-badge {badge_class}">{"Very High" if high_delay_pct > 30 else "High" if high_delay_pct > 20 else "Moderate" if high_delay_pct > 10 else "Low"}</span>', 
                           unsafe_allow_html=True)
            
            # ENHANCED BOTTLENECK IDENTIFICATION
            st.subheader("🚨 Bottleneck Performance Ranking")
            
            # Advanced insight box
            insight_text = f"""
            <div class="insight-box">
                <h4>💡 Analysis Insights</h4>
                <p><strong>Data Context:</strong> Analysis based on {total_records:,} {granularity.lower()} data points over {data_span} days.</p>
                <p><strong>Performance Summary:</strong> Peak delays reach {worst_delay:.0f} seconds with travel times varying up to {travel_increase:.0f}% above average during congestion periods.</p>
                <p><strong>Reliability:</strong> This corridor demonstrates {reliability_score:.0f}% travel time reliability, experiencing significant delays {high_delay_pct:.1f}% of the time.</p>
            </div>
            """
            st.markdown(insight_text, unsafe_allow_html=True)
            
            try:
                # Use raw data for bottleneck analysis to avoid aggregation effects
                performance_analysis = raw_data.groupby(['segment_name', 'direction']).agg({
                    'average_delay': ['mean', 'max', 'std'],
                    'average_traveltime': ['mean', 'max', 'std'], 
                    'average_speed': ['mean', 'min', 'std']
                }).round(2)
                
                # Flatten column names
                performance_analysis.columns = ['_'.join(col).strip() for col in performance_analysis.columns]
                performance_analysis = performance_analysis.reset_index()
                
                # Enhanced scoring system
                performance_analysis['Delay_Impact'] = (
                    performance_analysis['average_delay_max'] * 0.5 +
                    performance_analysis['average_delay_mean'] * 0.3 +
                    performance_analysis['average_delay_std'] * 0.2
                )
                
                performance_analysis['Travel_Impact'] = (
                    performance_analysis['average_traveltime_max'] * 0.4 +
                    performance_analysis['average_traveltime_mean'] * 0.4 +
                    performance_analysis['average_traveltime_std'] * 0.2
                )
                
                performance_analysis['Speed_Impact'] = (
                    (45 - performance_analysis['average_speed_min']) * 0.5 +
                    (45 - performance_analysis['average_speed_mean']) * 0.3 +
                    performance_analysis['average_speed_std'] * 0.2
                )
                
                # Overall bottleneck score
                performance_analysis['Bottleneck_Score'] = (
                    performance_analysis['Delay_Impact'] * 0.4 +
                    performance_analysis['Travel_Impact'] * 0.35 +
                    performance_analysis['Speed_Impact'] * 0.25
                ).round(1)
                
                # Add performance rating
                performance_analysis['Performance_Rating'] = pd.cut(
                    performance_analysis['Bottleneck_Score'],
                    bins=[-np.inf, 20, 40, 60, 80, np.inf],
                    labels=['Excellent', 'Good', 'Fair', 'Poor', 'Critical']
                )
                
                # Create display table
                display_cols = ['segment_name', 'direction', 'Performance_Rating', 'Bottleneck_Score',
                               'average_delay_mean', 'average_delay_max',
                               'average_traveltime_mean', 'average_traveltime_max',
                               'average_speed_mean', 'average_speed_min']
                
                final_performance = performance_analysis[display_cols].rename(columns={
                    'Performance_Rating': '🎯 Rating',
                    'average_delay_mean': 'Avg Delay (s)',
                    'average_delay_max': 'Peak Delay (s)',
                    'average_traveltime_mean': 'Avg Time (min)',
                    'average_traveltime_max': 'Peak Time (min)',
                    'average_speed_mean': 'Avg Speed (mph)',
                    'average_speed_min': 'Min Speed (mph)'
                })
                
                # Sort by bottleneck score (worst first)
                final_performance = final_performance.sort_values('Bottleneck_Score', ascending=False)
                
                st.dataframe(
                    final_performance.head(10),
                    use_container_width=True,
                    column_config={
                        "Bottleneck_Score": st.column_config.NumberColumn(
                            "🚨 Impact Score",
                            help="Composite score: higher values indicate greater bottleneck impact",
                            format="%.1f"
                        ),
                        "🎯 Rating": st.column_config.TextColumn(
                            "🎯 Rating",
                            help="Overall performance classification"
                        )
                    }
                )
                
            except Exception as e:
                st.error(f"Error in performance analysis: {str(e)}")
                # Simple fallback
                simple_perf = raw_data.groupby(['segment_name', 'direction']).agg({
                    'average_delay': 'mean',
                    'average_traveltime': 'mean',
                    'average_speed': 'mean'
                }).round(2).reset_index()
                
                st.dataframe(simple_perf.sort_values('average_traveltime', ascending=False))
            
        else:
            st.warning("Please select both start and end dates")

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
                    key="volume_granularity",
                    help="Higher aggregation levels will sum volumes and may increase total values"
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
            
            # Create context header for volume analysis
            total_volume_records = len(filtered_volume_data)
            volume_data_span = (date_range_vol[1] - date_range_vol[0]).days + 1
            
            context_header_vol = f"""
            <div class="context-header">
                <h2>📊 Volume Analysis: {intersection}</h2>
                <p>📅 {date_range_vol[0].strftime('%B %d, %Y')} to {date_range_vol[1].strftime('%B %d, %Y')} ({volume_data_span} days) • 
                {granularity_vol} Aggregation • {total_volume_records:,} data points</p>
            </div>
            """
            st.markdown(context_header_vol, unsafe_allow_html=True)
            
            # Use raw hourly data for capacity analysis
            raw_volume_data = display_df[(display_df['local_datetime'].dt.date >= date_range_vol[0]) & 
                                        (display_df['local_datetime'].dt.date <= date_range_vol[1])]
            
            # ENHANCED VOLUME METRICS
            st.subheader("🚦 Traffic Demand Indicators")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                peak_volume = raw_volume_data['total_volume'].max()
                p95_volume = raw_volume_data['total_volume'].quantile(0.95)
                
                # Capacity assessment
                capacity_util = (peak_volume / 1800) * 100  # Assuming 1800 vph capacity
                
                if capacity_util > 90: badge_class = "badge-critical"
                elif capacity_util > 75: badge_class = "badge-poor"
                elif capacity_util > 60: badge_class = "badge-fair"
                else: badge_class = "badge-good"
                
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
                if avg_util > 60: badge_class = "badge-poor"
                elif avg_util > 40: badge_class = "badge-fair"
                else: badge_class = "badge-good"
                
                st.markdown(f'<span class="performance-badge {badge_class}">{avg_util:.0f}% Avg Util</span>', 
                           unsafe_allow_html=True)
            
            with col3:
                # Peak-to-average ratio
                peak_avg_ratio = peak_volume / avg_volume if avg_volume > 0 else 0
                
                st.metric("📈 Peak/Average Ratio", f"{peak_avg_ratio:.1f}x", 
                         help="Higher ratios indicate more peaked demand patterns")
                
                if peak_avg_ratio > 3: badge_class = "badge-poor"
                elif peak_avg_ratio > 2: badge_class = "badge-fair"
                else: badge_class = "badge-good"
                
                st.markdown(f'<span class="performance-badge {badge_class}">{"High" if peak_avg_ratio > 3 else "Moderate" if peak_avg_ratio > 2 else "Low"} Peaking</span>', 
                           unsafe_allow_html=True)
            
            with col4:
                # Demand consistency
                cv_volume = (raw_volume_data['total_volume'].std() / 
                           raw_volume_data['total_volume'].mean()) * 100 if avg_volume > 0 else 0
                
                st.metric("🎯 Demand Consistency", f"{100-cv_volume:.0f}%", 
                         delta=f"CV: {cv_volume:.1f}%")
                
                if cv_volume < 30: badge_class = "badge-good"
                elif cv_volume < 50: badge_class = "badge-fair"
                else: badge_class = "badge-poor"
                
                st.markdown(f'<span class="performance-badge {badge_class}">{"Consistent" if cv_volume < 30 else "Variable" if cv_volume < 50 else "Highly Variable"}</span>', 
                           unsafe_allow_html=True)
            
            with col5:
                # Congestion risk hours
                high_volume_hours = (raw_volume_data['total_volume'] > 1200).sum()
                total_hours = len(raw_volume_data)
                risk_pct = (high_volume_hours / total_hours * 100) if total_hours > 0 else 0
                
                st.metric("⚠️ High Volume Hours", f"{high_volume_hours}", 
                         delta=f"{risk_pct:.1f}% of time")
                
                if risk_pct > 25: badge_class = "badge-critical"
                elif risk_pct > 15: badge_class = "badge-poor"
                elif risk_pct > 5: badge_class = "badge-fair"
                else: badge_class = "badge-good"
                
                st.markdown(f'<span class="performance-badge {badge_class}">{"Very High" if risk_pct > 25 else "High" if risk_pct > 15 else "Moderate" if risk_pct > 5 else "Low"} Risk</span>', 
                           unsafe_allow_html=True)
            
            # Volume insight box
            volume_insight = f"""
            <div class="insight-box">
                <h4>💡 Volume Analysis Insights</h4>
                <p><strong>Capacity Assessment:</strong> Peak volumes reach {peak_volume:,} vph ({capacity_util:.0f}% of estimated 1800 vph capacity).</p>
                <p><strong>Demand Pattern:</strong> {peak_avg_ratio:.1f}x peak-to-average ratio indicates {"highly peaked" if peak_avg_ratio > 3 else "moderately peaked" if peak_avg_ratio > 2 else "relatively flat"} demand patterns.</p>
                <p><strong>Congestion Risk:</strong> Volumes exceed 1200 vph during {high_volume_hours} hours ({risk_pct:.1f}% of analysis period), indicating {"high" if risk_pct > 15 else "moderate" if risk_pct > 5 else "low"} congestion risk.</p>
            </div>
            """
            st.markdown(volume_insight, unsafe_allow_html=True)
            
            # Enhanced intersection ranking with raw data
            st.subheader("🚨 Intersection Demand & Capacity Analysis")
            
            try:
                volume_analysis = raw_volume_data.groupby(['intersection_name', 'direction']).agg({
                    'total_volume': ['mean', 'max', 'std', 'count']
                }).round(0)
                
                volume_analysis.columns = ['_'.join(col).strip() for col in volume_analysis.columns]
                volume_analysis = volume_analysis.reset_index()
                
                # Enhanced scoring
                volume_analysis['Capacity_Pressure'] = volume_analysis['total_volume_max'] / 1800 * 100
                volume_analysis['Average_Load'] = volume_analysis['total_volume_mean'] / 1800 * 100
                volume_analysis['Variability'] = volume_analysis['total_volume_std'] / volume_analysis['total_volume_mean'] * 100
                
                # Risk classification
                volume_analysis['Risk_Level'] = pd.cut(
                    volume_analysis['total_volume_max'],
                    bins=[0, 800, 1200, 1500, 1800, np.inf],
                    labels=['Low', 'Moderate', 'High', 'Very High', 'Critical']
                )
                
                # Display table
                display_cols = ['intersection_name', 'direction', 'Risk_Level', 'Capacity_Pressure',
                               'total_volume_mean', 'total_volume_max', 'total_volume_std']
                
                final_volume = volume_analysis[display_cols].rename(columns={
                    'Risk_Level': '⚠️ Risk Level',
                    'Capacity_Pressure': '📊 Peak Capacity %',
                    'total_volume_mean': 'Avg Volume (vph)',
                    'total_volume_max': 'Peak Volume (vph)',
                    'total_volume_std': 'Volume StdDev'
                })
                
                final_volume = final_volume.sort_values('Peak Volume (vph)', ascending=False)
                
                st.dataframe(
                    final_volume.head(10),
                    use_container_width=True,
                    column_config={
                        "📊 Peak Capacity %": st.column_config.NumberColumn(
                            "📊 Peak Capacity %",
                            help="Peak volume as percentage of estimated 1800 vph capacity",
                            format="%.0f%%"
                        )
                    }
                )
                
            except Exception as e:
                st.error(f"Error in volume analysis: {str(e)}")
                
        else:
            st.warning("Please select both start and end dates")
