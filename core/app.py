import streamlit as st
import pandas as pd
from sidebar_functions import process_traffic_data, load_traffic_data, load_volume_data
from datetime import datetime, timedelta
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Active Transportation & Operations Management Dashboard",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with blue gradient color scheme
st.markdown("""
<style>
    /* Main container improvements with blue gradients */
    .main-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(30, 60, 114, 0.3);
    }

    .context-header {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0 2rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(79, 172, 254, 0.3);
        backdrop-filter: blur(10px);
    }

    .context-header h2 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        letter-spacing: -0.5px;
    }

    .context-header p {
        margin: 1rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }

    /* Dark mode compatibility */
    @media (prefers-color-scheme: dark) {
        .context-header {
            background: linear-gradient(135deg, #2980b9 0%, #3498db 100%);
        }
    }

    /* Enhanced metric containers */
    .metric-container {
        background: rgba(79, 172, 254, 0.1);
        border: 1px solid rgba(79, 172, 254, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.2);
    }

    /* Enhanced insight box */
    .insight-box {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.15) 0%, rgba(0, 242, 254, 0.15) 100%);
        border-left: 5px solid #4facfe;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(5px);
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.1);
    }

    .insight-box h4 {
        color: #1e3c72;
        margin-top: 0;
        font-weight: 600;
    }

    /* Performance badges with blue theme */
    .performance-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }

    .performance-badge:hover {
        transform: scale(1.05);
        border-color: rgba(255,255,255,0.3);
    }

    .badge-excellent { 
        background: linear-gradient(45deg, #2ecc71, #27ae60); 
        color: white; 
    }
    .badge-good { 
        background: linear-gradient(45deg, #3498db, #2980b9); 
        color: white; 
    }
    .badge-fair { 
        background: linear-gradient(45deg, #f39c12, #e67e22); 
        color: white; 
    }
    .badge-poor { 
        background: linear-gradient(45deg, #e74c3c, #c0392b); 
        color: white; 
    }
    .badge-critical { 
        background: linear-gradient(45deg, #e74c3c, #8e44ad); 
        color: white; 
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }

    /* Enhanced tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0 24px;
        border-radius: 12px;
        background: rgba(79, 172, 254, 0.1);
        border: 1px solid rgba(79, 172, 254, 0.2);
    }

    /* Loading spinner customization */
    .stSpinner > div {
        border-color: #4facfe transparent transparent transparent !important;
    }

    /* Chart container styling */
    .chart-container {
        background: rgba(79, 172, 254, 0.05);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(79, 172, 254, 0.1);
    }

    /* Volume specific styling */
    .volume-metric {
        background: linear-gradient(135deg, rgba(52, 152, 219, 0.1), rgba(41, 128, 185, 0.1));
        border: 1px solid rgba(52, 152, 219, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }

    .volume-metric:hover {
        background: linear-gradient(135deg, rgba(52, 152, 219, 0.2), rgba(41, 128, 185, 0.2));
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Enhanced main title with corrected name
st.markdown("""
<div class="main-container">
    <h1 style="text-align: center; margin: 0; font-size: 2.5rem; font-weight: 800;">
        🛣️ Active Transportation & Operations Management Dashboard
    </h1>
    <p style="text-align: center; margin-top: 1rem; font-size: 1.2rem; opacity: 0.9;">
        Advanced Traffic Engineering & Operations Management Platform
    </p>
</div>
""", unsafe_allow_html=True)

# Enhanced dashboard objective
dashboard_objective = """
<div style="
    font-size: 1.2rem;
    font-weight: 400;
    color: var(--text-color);
    background: linear-gradient(135deg, rgba(79, 172, 254, 0.1), rgba(0, 242, 254, 0.05));
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(79, 172, 254, 0.1);
    margin: 2rem 0;
    line-height: 1.8;
    border: 1px solid rgba(79, 172, 254, 0.2);
    backdrop-filter: blur(10px);
    ">
    <div style="text-align: center; margin-bottom: 1rem;">
        <strong style="font-size: 1.4rem; color: #2980b9;">🚀 The ADVANTEC Platform</strong>
    </div>
    <p>Leverages <strong>millions of data points</strong> trained on advanced Machine Learning algorithms to optimize traffic flow, reduce travel time, minimize fuel consumption, and decrease greenhouse gas emissions across the Coachella Valley transportation network.</p>
    <p><strong>Key Capabilities:</strong> Real-time anomaly detection • Intelligent cycle length optimization • Predictive traffic modeling • Performance analytics</p>
</div>
"""
st.markdown(dashboard_objective, unsafe_allow_html=True)

# Enhanced research question with blue styling
st.markdown("""
<div style="
    background: linear-gradient(135deg, #3498db, #2980b9);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1.5rem 0;
    text-align: center;
    box-shadow: 0 6px 20px rgba(52, 152, 219, 0.3);
">
    <h3 style="margin: 0; font-weight: 600;">
        🔍 Research Question
    </h3>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
        What are the main bottlenecks (slowest intersections) on Washington St that are most prone to causing increased travel times?
    </p>
</div>
""", unsafe_allow_html=True)

# Create two tabs instead of three
tab1, tab2 = st.tabs([
    "🚧 Performance & Delay Analysis",
    "📊 Traffic Demand & Capacity Analysis"
])


def create_performance_chart(data, metric_type="delay"):
    """Create enhanced performance visualization"""
    if data.empty:
        return None

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Time Series Analysis", "Distribution Analysis"),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    if metric_type == "delay":
        y_col = 'average_delay'
        title = "Traffic Delay Analysis"
        color = '#e74c3c'
    else:
        y_col = 'average_traveltime'
        title = "Travel Time Analysis"
        color = '#3498db'

    # Time series
    fig.add_trace(
        go.Scatter(
            x=data['local_datetime'],
            y=data[y_col],
            mode='lines+markers',
            name=f'{metric_type.title()} Trend',
            line=dict(color=color, width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )

    # Distribution
    fig.add_trace(
        go.Histogram(
            x=data[y_col],
            nbinsx=30,
            name=f'{metric_type.title()} Distribution',
            marker_color=color,
            opacity=0.7
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        title=title,
        showlegend=True,
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def create_volume_charts(data):
    """Create comprehensive volume analysis charts"""
    if data.empty:
        return None, None, None

    # 1. Volume trends over time by intersection
    fig1 = px.line(
        data,
        x='local_datetime',
        y='total_volume',
        color='intersection_name',
        title='📈 Traffic Volume Trends by Intersection',
        labels={'total_volume': 'Volume (vehicles/hour)', 'local_datetime': 'Date/Time'},
        template='plotly_white'
    )
    fig1.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 2. Volume distribution and capacity analysis
    fig2 = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Volume Distribution by Intersection", "Capacity Utilization Analysis"),
        vertical_spacing=0.12
    )

    # Box plot for volume distribution
    for intersection in data['intersection_name'].unique():
        intersection_data = data[data['intersection_name'] == intersection]
        fig2.add_trace(
            go.Box(
                y=intersection_data['total_volume'],
                name=intersection,
                boxpoints='outliers'
            ),
            row=1, col=1
        )

    # Capacity utilization heatmap data
    hourly_avg = data.groupby([data['local_datetime'].dt.hour, 'intersection_name'])[
        'total_volume'].mean().reset_index()
    hourly_pivot = hourly_avg.pivot(index='intersection_name', columns='local_datetime', values='total_volume')

    fig2.add_trace(
        go.Heatmap(
            z=hourly_pivot.values,
            x=hourly_pivot.columns,
            y=hourly_pivot.index,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Avg Volume (vph)")
        ),
        row=2, col=1
    )

    fig2.update_layout(
        height=800,
        title='📊 Volume Distribution & Capacity Analysis',
        template='plotly_white',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    # 3. Peak hour analysis
    data['hour'] = data['local_datetime'].dt.hour
    hourly_volume = data.groupby(['hour', 'intersection_name'])['total_volume'].mean().reset_index()

    fig3 = px.line(
        hourly_volume,
        x='hour',
        y='total_volume',
        color='intersection_name',
        title='🕐 Average Hourly Volume Patterns',
        labels={'total_volume': 'Average Volume (vph)', 'hour': 'Hour of Day'},
        template='plotly_white'
    )

    # Add capacity reference lines
    fig3.add_hline(y=1800, line_dash="dash", line_color="red",
                   annotation_text="Theoretical Capacity (1800 vph)")
    fig3.add_hline(y=1200, line_dash="dot", line_color="orange",
                   annotation_text="High Volume Threshold (1200 vph)")

    fig3.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig1, fig2, fig3


def get_performance_rating(score):
    """Enhanced performance rating with icons"""
    if score > 80:
        return "🟢 Excellent", "badge-excellent"
    elif score > 60:
        return "🔵 Good", "badge-good"
    elif score > 40:
        return "🟡 Fair", "badge-fair"
    elif score > 20:
        return "🟠 Poor", "badge-poor"
    else:
        return "🔴 Critical", "badge-critical"


with tab1:
    st.header("🚧 Comprehensive Performance & Travel Time Analysis")

    # Enhanced loading with progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text('Loading corridor performance data...')
    progress_bar.progress(25)

    corridor_df = load_traffic_data()
    progress_bar.progress(75)

    if corridor_df.empty:
        st.error("❌ Failed to load corridor data. Please check your data sources.")
        progress_bar.progress(100)
    else:
        progress_bar.progress(100)
        status_text.text('✅ Data loaded successfully!')

        # Clear progress indicators after a short delay
        import time

        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

        # Enhanced sidebar with better organization
        with st.sidebar:
            st.markdown("### 🚧 Performance Analysis Controls")

            # Corridor selection with search capability
            corridor_options = ["All Segments"] + sorted(corridor_df['segment_name'].unique().tolist())
            corridor = st.selectbox(
                "🛣️ Select Corridor Segment",
                corridor_options,
                help="Choose a specific segment or analyze all segments together"
            )

            # Enhanced date range with presets
            min_date = corridor_df['local_datetime'].dt.date.min()
            max_date = corridor_df['local_datetime'].dt.date.max()

            st.markdown("#### 📅 Analysis Period")

            # Quick date presets
            preset_col1, preset_col2 = st.columns(2)
            with preset_col1:
                if st.button("📅 Last 7 Days"):
                    date_range = (max_date - timedelta(days=7), max_date)
                else:
                    date_range = None
            with preset_col2:
                if st.button("📅 Last 30 Days"):
                    date_range = (max_date - timedelta(days=30), max_date)
                elif date_range is None:
                    date_range = None

            if date_range is None:
                date_range = st.date_input(
                    "Custom Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    help=f"Available data: {min_date} to {max_date}"
                )

            # Enhanced granularity options
            st.markdown("#### ⏰ Analysis Settings")
            granularity = st.selectbox(
                "Data Aggregation",
                options=["Hourly", "Daily", "Weekly", "Monthly"],
                index=0,
                help="Higher aggregation levels provide smoother trends but may mask peak conditions"
            )

            # Advanced filters for hourly data
            if granularity == "Hourly":
                time_filter = st.selectbox(
                    "Time Period Focus",
                    options=["All Hours", "Peak Hours (7-9 AM, 4-6 PM)", "AM Peak (7-9 AM)",
                             "PM Peak (4-6 PM)", "Off-Peak", "Custom Range"],
                    help="Focus analysis on specific time periods for targeted insights"
                )

                if time_filter == "Custom Range":
                    col1, col2 = st.columns(2)
                    with col1:
                        start_hour = st.selectbox("Start Hour", range(0, 24), index=7)
                    with col2:
                        end_hour = st.selectbox("End Hour", range(1, 25), index=18)

        # Process data with enhanced error handling
        if len(date_range) == 2:
            try:
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

                if filtered_data.empty:
                    st.warning("⚠️ No data available for the selected filters. Please adjust your selection.")
                else:
                    # Enhanced context header with more information
                    total_records = len(filtered_data)
                    data_span = (date_range[1] - date_range[0]).days + 1

                    time_context = ""
                    if granularity == "Hourly" and 'time_filter' in locals():
                        time_context = f" • {time_filter}"

                    context_header = f"""
                    <div class="context-header">
                        <h2>📊 Performance Dashboard: {corridor}</h2>
                        <p>📅 {date_range[0].strftime('%B %d, %Y')} to {date_range[1].strftime('%B %d, %Y')} 
                        ({data_span} days) • {granularity} Aggregation{time_context}</p>
                        <p>📈 Analyzing {total_records:,} data points across the selected period</p>
                    </div>
                    """
                    st.markdown(context_header, unsafe_allow_html=True)

                    # Enhanced KPI section with better calculations
                    st.subheader("🎯 Key Performance Indicators")

                    # Use raw data for more accurate metrics
                    raw_data = display_df[(display_df['local_datetime'].dt.date >= date_range[0]) &
                                          (display_df['local_datetime'].dt.date <= date_range[1])]

                    if not raw_data.empty:
                        col1, col2, col3, col4, col5 = st.columns(5)

                        with col1:
                            worst_delay = raw_data['average_delay'].max()
                            p95_delay = raw_data['average_delay'].quantile(0.95)

                            rating, badge_class = get_performance_rating(100 - min(worst_delay / 2, 100))

                            st.metric(
                                "🚨 Peak Delay",
                                f"{worst_delay:.1f}s",
                                delta=f"95th: {p95_delay:.1f}s"
                            )
                            st.markdown(f'<span class="performance-badge {badge_class}">{rating}</span>',
                                        unsafe_allow_html=True)

                        with col2:
                            worst_travel_time = raw_data['average_traveltime'].max()
                            avg_travel_time = raw_data['average_traveltime'].mean()
                            travel_increase = ((
                                                           worst_travel_time - avg_travel_time) / avg_travel_time * 100) if avg_travel_time > 0 else 0

                            st.metric(
                                "⏱️ Peak Travel Time",
                                f"{worst_travel_time:.1f}min",
                                delta=f"+{travel_increase:.0f}% vs avg"
                            )

                            impact_rating, badge_class = get_performance_rating(100 - min(travel_increase, 100))
                            st.markdown(f'<span class="performance-badge {badge_class}">{impact_rating}</span>',
                                        unsafe_allow_html=True)

                        with col3:
                            slowest_speed = raw_data['average_speed'].min()
                            avg_speed = raw_data['average_speed'].mean()
                            speed_drop = ((avg_speed - slowest_speed) / avg_speed * 100) if avg_speed > 0 else 0

                            st.metric(
                                "🐌 Minimum Speed",
                                f"{slowest_speed:.1f}mph",
                                delta=f"-{speed_drop:.0f}% vs avg"
                            )

                            speed_rating, badge_class = get_performance_rating(slowest_speed * 2)
                            st.markdown(f'<span class="performance-badge {badge_class}">{speed_rating}</span>',
                                        unsafe_allow_html=True)

                        with col4:
                            # Enhanced reliability calculation
                            cv_traveltime = (raw_data['average_traveltime'].std() /
                                             raw_data['average_traveltime'].mean()) * 100 if raw_data[
                                                                                                 'average_traveltime'].mean() > 0 else 0
                            reliability_score = max(0, 100 - cv_traveltime)

                            st.metric(
                                "🎯 Reliability Index",
                                f"{reliability_score:.0f}%",
                                delta=f"CV: {cv_traveltime:.1f}%"
                            )

                            rel_rating, badge_class = get_performance_rating(reliability_score)
                            st.markdown(f'<span class="performance-badge {badge_class}">{rel_rating}</span>',
                                        unsafe_allow_html=True)

                        with col5:
                            # Congestion frequency with threshold analysis
                            high_delay_pct = (raw_data['average_delay'] > 60).mean() * 100

                            st.metric(
                                "⚠️ Congestion Frequency",
                                f"{high_delay_pct:.1f}%",
                                delta=f"{(raw_data['average_delay'] > 60).sum()} hours"
                            )

                            freq_rating, badge_class = get_performance_rating(100 - high_delay_pct)
                            st.markdown(f'<span class="performance-badge {badge_class}">{freq_rating}</span>',
                                        unsafe_allow_html=True)

                    # Enhanced visualization section
                    if len(filtered_data) > 1:
                        st.subheader("📈 Performance Trends")

                        viz_col1, viz_col2 = st.columns(2)

                        with viz_col1:
                            delay_chart = create_performance_chart(filtered_data, "delay")
                            if delay_chart:
                                st.plotly_chart(delay_chart, use_container_width=True)

                        with viz_col2:
                            travel_chart = create_performance_chart(filtered_data, "travel")
                            if travel_chart:
                                st.plotly_chart(travel_chart, use_container_width=True)

                    # Enhanced insight generation
                    if not raw_data.empty:
                        insight_text = f"""
                        <div class="insight-box">
                            <h4>💡 Advanced Performance Insights</h4>
                            <p><strong>📊 Data Overview:</strong> Analysis covers {total_records:,} {granularity.lower()} observations spanning {data_span} days with {len(raw_data):,} raw hourly measurements.</p>
                            <p><strong>🚨 Critical Findings:</strong> Peak delays reach {worst_delay:.0f} seconds ({worst_delay / 60:.1f} minutes) with travel times varying up to {travel_increase:.0f}% above baseline during peak congestion periods.</p>
                            <p><strong>📈 Reliability Assessment:</strong> This corridor demonstrates {reliability_score:.0f}% travel time reliability, experiencing significant delays (>60s) during {high_delay_pct:.1f}% of operational hours.</p>
                            <p><strong>🎯 Recommendation:</strong> {"Critical intervention needed" if worst_delay > 120 else "Optimization recommended" if worst_delay > 60 else "Monitor performance trends"} based on observed performance metrics.</p>
                        </div>
                        """
                        st.markdown(insight_text, unsafe_allow_html=True)

                    # Enhanced bottleneck analysis
                    st.subheader("🚨 Comprehensive Bottleneck Analysis")

                    if not raw_data.empty:
                        try:
                            performance_analysis = raw_data.groupby(['segment_name', 'direction']).agg({
                                'average_delay': ['mean', 'max', 'std', 'count'],
                                'average_traveltime': ['mean', 'max', 'std'],
                                'average_speed': ['mean', 'min', 'std']
                            }).round(2)

                            performance_analysis.columns = ['_'.join(col).strip() for col in
                                                            performance_analysis.columns]
                            performance_analysis = performance_analysis.reset_index()

                            # Advanced scoring system
                            performance_analysis['Bottleneck_Score'] = (
                                    performance_analysis['average_delay_max'] * 0.4 +
                                    performance_analysis['average_delay_mean'] * 0.3 +
                                    performance_analysis['average_traveltime_max'] * 0.3
                            ).round(1)

                            performance_analysis['Performance_Rating'] = pd.cut(
                                performance_analysis['Bottleneck_Score'],
                                bins=[-np.inf, 20, 40, 60, 80, np.inf],
                                labels=['🟢 Excellent', '🔵 Good', '🟡 Fair', '🟠 Poor', '🔴 Critical']
                            )

                            # Enhanced display
                            display_cols = [
                                'segment_name', 'direction', 'Performance_Rating',
                                'Bottleneck_Score', 'average_delay_mean', 'average_delay_max',
                                'average_traveltime_mean', 'average_traveltime_max',
                                'average_speed_mean', 'average_speed_min'
                            ]

                            final_performance = performance_analysis[display_cols].rename(columns={
                                'Performance_Rating': '🎯 Performance Rating',
                                'average_delay_mean': 'Avg Delay (s)',
                                'average_delay_max': 'Peak Delay (s)',
                                'average_traveltime_mean': 'Avg Time (min)',
                                'average_traveltime_max': 'Peak Time (min)',
                                'average_speed_mean': 'Avg Speed (mph)',
                                'average_speed_min': 'Min Speed (mph)'
                            })

                            final_performance = final_performance.sort_values('Bottleneck_Score', ascending=False)

                            st.dataframe(
                                final_performance.head(15),
                                use_container_width=True,
                                column_config={
                                    "Bottleneck_Score": st.column_config.NumberColumn(
                                        "🚨 Impact Score",
                                        help="Composite bottleneck impact score: higher values indicate greater need for intervention",
                                        format="%.1f"
                                    )
                                }
                            )

                        except Exception as e:
                            st.error(f"❌ Error in performance analysis: {str(e)}")

            except Exception as e:
                st.error(f"❌ Error processing traffic data: {str(e)}")

        else:
            st.warning("⚠️ Please select both start and end dates to proceed with the analysis.")

with tab2:
    st.header("📊 Advanced Traffic Demand & Capacity Analysis")

    # Load volume data with progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text('Loading traffic demand data...')
    progress_bar.progress(25)

    volume_df = load_volume_data()
    progress_bar.progress(100)

    if volume_df.empty:
        st.error("❌ Failed to load volume data. Please check your data sources.")
    else:
        status_text.text('✅ Volume data loaded successfully!')
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

        # Enhanced sidebar for volume analysis
        with st.sidebar:
            st.markdown("### 📊 Volume Analysis Controls")

            # Intersection selection
            intersection_options = ["All Intersections"] + volume_df['intersection_name'].drop_duplicates().tolist()
            intersection = st.selectbox(
                "🚦 Select Intersection",
                intersection_options,
                help="Choose a specific intersection or analyze all intersections together"
            )

            # Enhanced date range with presets
            min_date = volume_df['local_datetime'].dt.date.min()
            max_date = volume_df['local_datetime'].dt.date.max()

            st.markdown("#### 📅 Analysis Period")

            # Quick presets for volume data
            preset_col1, preset_col2 = st.columns(2)
            with preset_col1:
                if st.button("📅 Last 7 Days", key="vol_7d"):
                    date_range_vol = (max_date - timedelta(days=7), max_date)
                else:
                    date_range_vol = None
            with preset_col2:
                if st.button("📅 Last 30 Days", key="vol_30d"):
                    date_range_vol = (max_date - timedelta(days=30), max_date)
                elif date_range_vol is None:
                    date_range_vol = None

            if date_range_vol is None:
                date_range_vol = st.date_input(
                    "Custom Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="volume_date_range",
                    help=f"Available data: {min_date} to {max_date}"
                )

            # Granularity selection
            st.markdown("#### ⏰ Analysis Settings")
            granularity_vol = st.selectbox(
                "Data Aggregation",
                options=["Hourly", "Daily", "Weekly", "Monthly"],
                index=0,
                key="volume_granularity",
                help="Higher aggregation levels will sum volumes for broader trend analysis"
            )

            # Direction filter
            direction_options = ["All Directions"] + sorted(volume_df['direction'].unique().tolist())
            direction_filter = st.selectbox(
                "🔄 Direction Filter",
                direction_options,
                help="Filter analysis by traffic direction"
            )

        # Process volume data
        if len(date_range_vol) == 2:
            try:
                # Filter data by selected intersection
                if intersection != "All Intersections":
                    display_df = volume_df[volume_df['intersection_name'] == intersection].copy()
                else:
                    display_df = volume_df.copy()

                # Filter by direction if specified
                if direction_filter != "All Directions":
                    display_df = display_df[display_df['direction'] == direction_filter]

                # Process the data
                filtered_volume_data = process_traffic_data(
                    display_df, date_range_vol, granularity_vol
                )

                if filtered_volume_data.empty:
                    st.warning("⚠️ No volume data available for the selected filters. Please adjust your selection.")
                else:
                    # Enhanced context header for volume analysis
                    total_volume_records = len(filtered_volume_data)
                    volume_data_span = (date_range_vol[1] - date_range_vol[0]).days + 1

                    context_header_vol = f"""
                    <div class="context-header">
                        <h2>📊 Volume Analysis: {intersection}</h2>
                        <p>📅 {date_range_vol[0].strftime('%B %d, %Y')} to {date_range_vol[1].strftime('%B %d, %Y')} 
                        ({volume_data_span} days) • {granularity_vol} Aggregation</p>
                        <p>📈 Analyzing {total_volume_records:,} volume observations • Direction: {direction_filter}</p>
                    </div>
                    """
                    st.markdown(context_header_vol, unsafe_allow_html=True)

                    # Use raw hourly data for more accurate capacity analysis
                    raw_volume_data = display_df[(display_df['local_datetime'].dt.date >= date_range_vol[0]) &
                                                 (display_df['local_datetime'].dt.date <= date_range_vol[1])]

                    if direction_filter != "All Directions":
                        raw_volume_data = raw_volume_data[raw_volume_data['direction'] == direction_filter]

                    # ENHANCED VOLUME METRICS
                    st.subheader("🚦 Traffic Demand Performance Indicators")

                    if not raw_volume_data.empty:
                        col1, col2, col3, col4, col5 = st.columns(5)

                        with col1:
                            peak_volume = raw_volume_data['total_volume'].max()
                            p95_volume = raw_volume_data['total_volume'].quantile(0.95)

                            # Capacity assessment (assuming 1800 vph capacity)
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
                            st.markdown(
                                f'<span class="performance-badge {badge_class}">{capacity_util:.0f}% Capacity</span>',
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

                            st.markdown(
                                f'<span class="performance-badge {badge_class}">{avg_util:.0f}% Avg Util</span>',
                                unsafe_allow_html=True)

                        with col3:
                            # Peak-to-average ratio
                            peak_avg_ratio = peak_volume / avg_volume if avg_volume > 0 else 0

                            st.metric("📈 Peak/Average Ratio", f"{peak_avg_ratio:.1f}x",
                                      help="Higher ratios indicate more peaked demand patterns")

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
                            # Demand consistency
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
                            # Congestion risk hours
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

                    # Enhanced visualizations
                    st.subheader("📈 Volume Analysis Visualizations")

                    if len(filtered_volume_data) > 1:
                        # Create comprehensive volume charts
                        chart1, chart2, chart3 = create_volume_charts(filtered_volume_data)

                        # Display charts
                        if chart1:
                            st.plotly_chart(chart1, use_container_width=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            if chart3:
                                st.plotly_chart(chart3, use_container_width=True)
                        with col2:
                            if chart2:
                                st.plotly_chart(chart2, use_container_width=True)

                    # Volume insight box
                    if not raw_volume_data.empty:
                        volume_insight = f"""
                        <div class="insight-box">
                            <h4>💡 Advanced Volume Analysis Insights</h4>
                            <p><strong>📊 Capacity Assessment:</strong> Peak volumes reach {peak_volume:,} vph ({capacity_util:.0f}% of estimated 1800 vph capacity) with an average utilization of {avg_util:.0f}%.</p>
                            <p><strong>📈 Demand Characteristics:</strong> {peak_avg_ratio:.1f}x peak-to-average ratio indicates {"highly peaked" if peak_avg_ratio > 3 else "moderately peaked" if peak_avg_ratio > 2 else "relatively flat"} demand patterns with {100 - cv_volume:.0f}% consistency.</p>
                            <p><strong>⚠️ Congestion Risk:</strong> High-volume conditions (>1200 vph) occur during {high_volume_hours} hours ({risk_pct:.1f}% of analysis period), indicating {"critical" if risk_pct > 25 else "high" if risk_pct > 15 else "moderate" if risk_pct > 5 else "low"} capacity stress.</p>
                            <p><strong>🎯 Recommendations:</strong> {"Immediate capacity expansion needed" if capacity_util > 90 else "Consider signal optimization" if capacity_util > 75 else "Monitor trends and optimize timing" if capacity_util > 60 else "Current capacity appears adequate"}.</p>
                        </div>
                        """
                        st.markdown(volume_insight, unsafe_allow_html=True)

                    # Enhanced intersection ranking
                    st.subheader("🚨 Intersection Volume & Capacity Risk Analysis")

                    try:
                        volume_analysis = raw_volume_data.groupby(['intersection_name', 'direction']).agg({
                            'total_volume': ['mean', 'max', 'std', 'count']
                        }).round(0)

                        volume_analysis.columns = ['_'.join(col).strip() for col in volume_analysis.columns]
                        volume_analysis = volume_analysis.reset_index()

                        # Enhanced capacity and risk scoring
                        volume_analysis['Peak_Capacity_Util'] = (
                                    volume_analysis['total_volume_max'] / 1800 * 100).round(1)
                        volume_analysis['Avg_Capacity_Util'] = (
                                    volume_analysis['total_volume_mean'] / 1800 * 100).round(1)
                        volume_analysis['Volume_Variability'] = (volume_analysis['total_volume_std'] / volume_analysis[
                            'total_volume_mean'] * 100).round(1)
                        volume_analysis['Peak_Avg_Ratio'] = (
                                    volume_analysis['total_volume_max'] / volume_analysis['total_volume_mean']).round(1)

                        # Comprehensive risk scoring
                        volume_analysis['Capacity_Risk_Score'] = (
                                volume_analysis['Peak_Capacity_Util'] * 0.5 +
                                volume_analysis['Avg_Capacity_Util'] * 0.3 +
                                (volume_analysis['Peak_Avg_Ratio'] * 10) * 0.2
                        ).round(1)

                        # Risk classification
                        volume_analysis['Risk_Classification'] = pd.cut(
                            volume_analysis['Capacity_Risk_Score'],
                            bins=[0, 40, 60, 80, 90, 120],
                            labels=['🟢 Low Risk', '🟡 Moderate Risk', '🟠 High Risk', '🔴 Critical Risk', '🚨 Severe Risk'],
                            include_lowest=True
                        )

                        # Priority recommendations
                        volume_analysis['Action_Priority'] = pd.cut(
                            volume_analysis['Peak_Capacity_Util'],
                            bins=[0, 60, 75, 90, 100],
                            labels=['🟢 Monitor', '🟡 Optimize', '🟠 Upgrade', '🔴 Urgent'],
                            include_lowest=True
                        )

                        # Display enhanced table
                        display_cols = [
                            'intersection_name', 'direction', 'Risk_Classification', 'Action_Priority',
                            'Capacity_Risk_Score', 'Peak_Capacity_Util', 'Avg_Capacity_Util',
                            'total_volume_mean', 'total_volume_max', 'Peak_Avg_Ratio', 'total_volume_count'
                        ]

                        final_volume = volume_analysis[display_cols].rename(columns={
                            'Risk_Classification': '⚠️ Risk Level',
                            'Action_Priority': '🎯 Action Priority',
                            'Capacity_Risk_Score': '🚨 Risk Score',
                            'Peak_Capacity_Util': '📊 Peak Capacity %',
                            'Avg_Capacity_Util': '📊 Avg Capacity %',
                            'total_volume_mean': 'Avg Volume (vph)',
                            'total_volume_max': 'Peak Volume (vph)',
                            'Peak_Avg_Ratio': 'Peak/Avg Ratio',
                            'total_volume_count': 'Data Points'
                        })

                        final_volume = final_volume.sort_values('🚨 Risk Score', ascending=False)

                        st.dataframe(
                            final_volume.head(15),
                            use_container_width=True,
                            column_config={
                                "🚨 Risk Score": st.column_config.NumberColumn(
                                    "🚨 Capacity Risk Score",
                                    help="Composite risk score considering peak utilization, average load, and demand variability",
                                    format="%.1f",
                                    min_value=0,
                                    max_value=120
                                ),
                                "📊 Peak Capacity %": st.column_config.NumberColumn(
                                    "📊 Peak Capacity %",
                                    help="Peak volume as percentage of estimated 1800 vph capacity",
                                    format="%.1f%%",
                                    min_value=0,
                                    max_value=150
                                ),
                                "📊 Avg Capacity %": st.column_config.NumberColumn(
                                    "📊 Avg Capacity %",
                                    help="Average volume as percentage of estimated capacity",
                                    format="%.1f%%"
                                )
                            }
                        )

                    except Exception as e:
                        st.error(f"❌ Error in volume analysis: {str(e)}")
                        # Simple fallback
                        simple_volume = raw_volume_data.groupby(['intersection_name', 'direction']).agg({
                            'total_volume': ['mean', 'max']
                        }).round(0).reset_index()
                        simple_volume.columns = ['_'.join(col).strip() if col[1] else col[0] for col in
                                                 simple_volume.columns]
                        st.dataframe(simple_volume.sort_values('total_volume_max', ascending=False))

            except Exception as e:
                st.error(f"❌ Error processing volume data: {str(e)}")
                st.info("Please check your data sources and try again.")

        else:
            st.warning("⚠️ Please select both start and end dates to proceed with the volume analysis.")

# Enhanced footer with blue theme
st.markdown("""
---
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(79, 172, 254, 0.1), rgba(0, 242, 254, 0.05)); border-radius: 15px; margin-top: 2rem; border: 1px solid rgba(79, 172, 254, 0.2);">
    <h4 style="color: #2980b9; margin-bottom: 1rem;">🛣️ Active Transportation & Operations Management Dashboard</h4>
    <p style="opacity: 0.8; margin: 0;">Powered by Advanced Machine Learning • Real-time Traffic Intelligence • Sustainable Transportation Solutions</p>
    <p style="opacity: 0.6; margin-top: 0.5rem; font-size: 0.9rem;">© 2024 ADVANTEC Platform - Optimizing Transportation Networks</p>
</div>
""", unsafe_allow_html=True)