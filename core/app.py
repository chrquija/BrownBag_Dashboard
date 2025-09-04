# app.py
# ============================================
# User-Friendly Active Transportation Dashboard
# Redesigned for all age groups with simplified UX
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === External project functions ===
from sidebar_functions import (
    load_traffic_data,
    load_volume_data,
    process_traffic_data,
    get_corridor_df,
    get_volume_df,
    get_performance_rating,
    performance_chart,
    date_range_preset_controls,
    compute_perf_kpis_interpretable,
    render_badge,
)

# Cycle length section
from cycle_length_recommendations import render_cycle_length_section

# Map builders
from Map import build_corridor_map, build_intersection_map, build_intersections_overview

# =========================
# Page configuration - Mobile-first
# =========================
st.set_page_config(
    page_title="Traffic Dashboard - Washington Street",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",  # Start collapsed for cleaner look
)

# Enhanced config for better mobile experience
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "toggleSpikelines", "zoom2d", "pan2d"],
    "displayModeBar": False  # Hide toolbar completely for simplicity
}
MAP_HEIGHT = 500  # Smaller default height

# =========================
# Constants - Simplified
# =========================
CAPACITY_LIMIT = 1800  # vehicles per hour
HIGH_TRAFFIC_THRESHOLD = 1200  # vehicles per hour

# Corridor locations (south to north)
CORRIDOR_LOCATIONS = [
    "Avenue 52", "Calle Tampico", "Village Shopping Ctr", "Avenue 50",
    "Sagebrush Ave", "Eisenhower Dr", "Avenue 48", "Avenue 47",
    "Point Happy Simon", "Hwy 111"
]


# =========================
# Helper Functions - Simplified
# =========================
def get_simple_direction(direction_series):
    """Convert direction to simple North/South"""
    if direction_series is None:
        return "Unknown"
    dir_str = str(direction_series).lower()
    if 'north' in dir_str or 'nb' in dir_str:
        return "North"
    elif 'south' in dir_str or 'sb' in dir_str:
        return "South"
    return "Unknown"


def format_time_friendly(minutes):
    """Format time in user-friendly way"""
    if minutes < 1:
        return "Less than 1 min"
    elif minutes < 60:
        return f"{minutes:.0f} min"
    else:
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:.0f}h {mins:.0f}m"


def get_traffic_status(value, good_max, warning_max):
    """Get simple traffic light status"""
    if value <= good_max:
        return "🟢 Good", "good"
    elif value <= warning_max:
        return "🟡 Fair", "warning"
    else:
        return "🔴 Needs Attention", "alert"


# =========================
# Enhanced CSS - Mobile-friendly
# =========================
st.markdown("""
<style>
    /* Base mobile-first styles */
    .main-header {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(79, 172, 254, 0.3);
    }

    .main-header h1 {
        font-size: 1.8rem;
        margin: 0 0 0.5rem;
        font-weight: 700;
    }

    .main-header p {
        font-size: 1.1rem;
        margin: 0;
        opacity: 0.9;
    }

    /* Quick summary cards */
    .summary-card {
        background: linear-gradient(135deg, rgba(79,172,254,0.1), rgba(0,242,254,0.05));
        border: 1px solid rgba(79,172,254,0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }

    .summary-card h3 {
        color: #2980b9;
        margin: 0 0 0.5rem;
        font-size: 1.4rem;
    }

    .metric-large {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
        line-height: 1.1;
    }

    .metric-subtitle {
        font-size: 1.1rem;
        opacity: 0.8;
        margin: 0;
    }

    /* Status badges - larger and more accessible */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 600;
        margin: 0.5rem 0.25rem;
        min-height: 44px;
        line-height: 1.4;
    }

    .status-good {
        background: linear-gradient(45deg, #27ae60, #2ecc71);
        color: white;
    }

    .status-warning {
        background: linear-gradient(45deg, #e67e22, #f39c12);
        color: white;
    }

    .status-alert {
        background: linear-gradient(45deg, #c0392b, #e74c3c);
        color: white;
    }

    /* Simplified insight boxes */
    .insight-simple {
        background: #f8f9fa;
        border-left: 4px solid #4facfe;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        font-size: 1.1rem;
        line-height: 1.6;
    }

    .insight-simple h4 {
        color: #2c3e50;
        margin: 0 0 1rem;
        font-size: 1.3rem;
    }

    /* Mobile-responsive columns */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.5rem;
        }

        .metric-large {
            font-size: 2rem;
        }

        .summary-card {
            padding: 1rem;
        }
    }

    /* Help tooltips */
    .help-tooltip {
        background: #e8f4f9;
        border: 1px solid #bee5eb;
        border-radius: 6px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }

    /* Quick start guide */
    .quick-start {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }

    .quick-start h3 {
        margin: 0 0 1rem;
        font-size: 1.4rem;
    }

    .quick-start ol {
        margin: 0;
        padding-left: 1.2rem;
    }

    .quick-start li {
        margin: 0.5rem 0;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Main Header - Simplified
# =========================
st.markdown("""
<div class="main-header">
    <h1>🚗 Washington Street Traffic Dashboard</h1>
    <p>Simple traffic insights for your daily commute</p>
</div>
""", unsafe_allow_html=True)

# Quick Start Guide (collapsible)
with st.expander("👋 New here? Quick Start Guide", expanded=False):
    st.markdown("""
    <div class="quick-start">
        <h3>How to use this dashboard:</h3>
        <ol>
            <li><strong>Choose your route:</strong> Pick where you start and end your trip</li>
            <li><strong>Check traffic summary:</strong> See if your route is running smoothly</li>
            <li><strong>Plan your trip:</strong> Use the recommended travel time</li>
            <li><strong>Explore details:</strong> Click "Show More Details" for deeper insights</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Smart Defaults & Simplified Navigation
# =========================
# Use session state for user preferences
if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = 'simple'  # Default to simple mode

if 'route_selected' not in st.session_state:
    st.session_state.route_selected = False

# Main mode toggle
analysis_mode = st.radio(
    "Choose your experience:",
    ["🚀 Quick Analysis", "📊 Detailed Analysis"],
    key="mode_selector",
    horizontal=True,
    help="Quick Analysis shows the essentials. Detailed Analysis shows all data."
)

is_detailed_mode = analysis_mode == "📊 Detailed Analysis"


# =========================
# Load Data with Better Error Handling
# =========================
@st.cache_data
def load_data_safely():
    """Load data with user-friendly error handling"""
    try:
        corridor_data = get_corridor_df()
        volume_data = get_volume_df()
        return corridor_data, volume_data, None
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), str(e)


corridor_df, volume_df, load_error = load_data_safely()

if load_error:
    st.error(f"⚠️ Unable to load traffic data: {load_error}")
    st.info("Please check your internet connection and try again.")
    st.stop()

if corridor_df.empty and volume_df.empty:
    st.warning("⚠️ No traffic data available at the moment. Please try again later.")
    st.stop()

# =========================
# Simple Route Selection
# =========================
st.markdown("## 📍 Choose Your Route")

# Get available locations from data
if not corridor_df.empty:
    available_locations = []
    for loc in CORRIDOR_LOCATIONS:
        if any(loc in str(segment) for segment in corridor_df.get('segment_name', [])):
            available_locations.append(loc)
else:
    available_locations = CORRIDOR_LOCATIONS[:5]  # Fallback

if len(available_locations) >= 2:
    col1, col2 = st.columns(2)

    with col1:
        start_location = st.selectbox(
            "🟢 Starting from:",
            available_locations,
            key="start_loc",
            help="Where does your trip begin?"
        )

    with col2:
        end_options = [loc for loc in available_locations if loc != start_location]
        end_location = st.selectbox(
            "🔴 Going to:",
            end_options,
            index=len(end_options) - 1 if end_options else 0,
            key="end_loc",
            help="Where does your trip end?"
        )

    st.session_state.route_selected = True
else:
    st.error("⚠️ Not enough location data available for route selection.")
    st.stop()

# =========================
# Date Selection - Simplified
# =========================
if is_detailed_mode:
    st.markdown("## 📅 Time Period")

    # Simple preset buttons
    col1, col2, col3, col4 = st.columns(4)

    date_preset = None
    with col1:
        if st.button("📅 Last 7 Days", use_container_width=True):
            date_preset = "7d"
    with col2:
        if st.button("📅 Last 30 Days", use_container_width=True):
            date_preset = "30d"
    with col3:
        if st.button("📅 Last 3 Months", use_container_width=True):
            date_preset = "90d"
    with col4:
        if st.button("📅 All Available", use_container_width=True):
            date_preset = "all"

    # Set date range based on selection
    if not corridor_df.empty:
        max_date = corridor_df["local_datetime"].dt.date.max()
        min_date = corridor_df["local_datetime"].dt.date.min()

        if date_preset == "7d":
            start_date = max(min_date, max_date - timedelta(days=7))
            end_date = max_date
        elif date_preset == "30d":
            start_date = max(min_date, max_date - timedelta(days=30))
            end_date = max_date
        elif date_preset == "90d":
            start_date = max(min_date, max_date - timedelta(days=90))
            end_date = max_date
        elif date_preset == "all":
            start_date = min_date
            end_date = max_date
        else:
            # Default to last 30 days
            start_date = max(min_date, max_date - timedelta(days=30))
            end_date = max_date

        selected_date_range = (start_date, end_date)
    else:
        selected_date_range = (datetime.now().date() - timedelta(days=30), datetime.now().date())
else:
    # In simple mode, default to last 30 days
    if not corridor_df.empty:
        max_date = corridor_df["local_datetime"].dt.date.max()
        min_date = corridor_df["local_datetime"].dt.date.min()
        start_date = max(min_date, max_date - timedelta(days=30))
        selected_date_range = (start_date, max_date)
    else:
        selected_date_range = (datetime.now().date() - timedelta(days=30), datetime.now().date())

# =========================
# Process Route Data
# =========================
if st.session_state.route_selected:
    # Filter data for selected route
    try:
        route_data = corridor_df.copy()
        if not route_data.empty:
            # Filter to segments that match the route
            route_segments = []
            start_idx = next((i for i, loc in enumerate(CORRIDOR_LOCATIONS) if loc == start_location), 0)
            end_idx = next((i for i, loc in enumerate(CORRIDOR_LOCATIONS) if loc == end_location),
                           len(CORRIDOR_LOCATIONS) - 1)

            if start_idx < end_idx:
                for i in range(start_idx, end_idx):
                    segment = f"{CORRIDOR_LOCATIONS[i]} → {CORRIDOR_LOCATIONS[i + 1]}"
                    route_segments.append(segment)

            if route_segments:
                route_data = route_data[route_data['segment_name'].isin(route_segments)]

            # Process the data
            processed_data = process_traffic_data(
                route_data,
                selected_date_range,
                "Hourly"  # Always use hourly for simplicity
            )

        else:
            processed_data = pd.DataFrame()

    except Exception as e:
        st.error(f"⚠️ Error processing route data: {e}")
        processed_data = pd.DataFrame()

# =========================
# Main Traffic Summary
# =========================
st.markdown("## 🚦 Your Commute Summary")

if not processed_data.empty:
    # Calculate simple metrics
    try:
        # Ensure numeric columns
        for col in ['average_traveltime', 'average_delay', 'average_speed']:
            if col in processed_data.columns:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')

        avg_travel_time = processed_data[
            'average_traveltime'].mean() if 'average_traveltime' in processed_data.columns else 0
        avg_delay = processed_data['average_delay'].mean() if 'average_delay' in processed_data.columns else 0
        max_travel_time = processed_data['average_traveltime'].quantile(
            0.95) if 'average_traveltime' in processed_data.columns else avg_travel_time

        # Simple status calculation
        delay_status, delay_class = get_traffic_status(avg_delay, 30, 60)  # seconds
        travel_status, travel_class = get_traffic_status(avg_travel_time, 8, 12)  # minutes

        # Display summary cards
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="summary-card">
                <h3>⏱️ Typical Trip</h3>
                <div class="metric-large">{format_time_friendly(avg_travel_time)}</div>
                <div class="metric-subtitle">Average travel time</div>
                <div class="status-badge status-{travel_class}">{travel_status}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            buffer_time = max_travel_time - avg_travel_time
            st.markdown(f"""
            <div class="summary-card">
                <h3>📅 Plan For</h3>
                <div class="metric-large">{format_time_friendly(max_travel_time)}</div>
                <div class="metric-subtitle">To arrive on time 95% of trips</div>
                <div class="status-badge status-good">Leave {format_time_friendly(buffer_time)} earlier</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="summary-card">
                <h3>🚨 Delays</h3>
                <div class="metric-large">{avg_delay:.0f} sec</div>
                <div class="metric-subtitle">Average delay</div>
                <div class="status-badge status-{delay_class}">{delay_status}</div>
            </div>
            """, unsafe_allow_html=True)

        # What This Means section
        st.markdown(f"""
        <div class="insight-simple">
            <h4>💡 What This Means for You</h4>
            <p><strong>Your {start_location} → {end_location} route typically takes {format_time_friendly(avg_travel_time)}.</strong></p>
            <p>To arrive on time for important appointments, plan for <strong>{format_time_friendly(max_travel_time)}</strong> 
            (leave an extra {format_time_friendly(buffer_time)} earlier than the typical time).</p>
            {f"<p>⚠️ This route experiences moderate delays averaging {avg_delay:.0f} seconds. Consider alternative times if possible.</p>" if avg_delay > 30 else ""}
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error calculating traffic summary: {e}")
        st.info("Please try selecting different locations or check back later.")

else:
    st.info(
        f"⚠️ No traffic data available for the route {start_location} → {end_location}. Try selecting different locations.")

# =========================
# Interactive Map
# =========================
st.markdown("## 🗺️ Route Map")

try:
    map_fig = build_corridor_map(start_location, end_location)
    if map_fig:
        map_fig.update_layout(height=MAP_HEIGHT, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(map_fig, use_container_width=True, config=PLOTLY_CONFIG)
        st.caption(f"📍 Your route: {start_location} → {end_location}")
    else:
        st.info("🗺️ Map not available for this route.")
except Exception:
    st.info("🗺️ Map temporarily unavailable.")

# =========================
# Detailed Analysis (if enabled)
# =========================
if is_detailed_mode:
    st.markdown("## 📊 Detailed Analysis")

    if not processed_data.empty:
        # Advanced controls in expander
        with st.expander("⚙️ Advanced Options", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                time_filter = st.selectbox(
                    "Time Period Focus",
                    ["All Hours", "Morning Rush (7-9 AM)", "Evening Rush (4-6 PM)", "Off-Peak Hours"],
                    help="Focus analysis on specific time periods"
                )

            with col2:
                granularity = st.selectbox(
                    "Data Detail Level",
                    ["Hourly", "Daily", "Weekly"],
                    help="How detailed should the analysis be?"
                )

        # Performance trends
        st.subheader("📈 Traffic Trends")

        if len(processed_data) > 1:
            col1, col2 = st.columns(2)

            with col1:
                # Travel time trend
                fig_travel = px.line(
                    processed_data.sort_values('local_datetime'),
                    x='local_datetime',
                    y='average_traveltime',
                    title='Travel Time Over Time',
                    labels={'average_traveltime': 'Travel Time (minutes)', 'local_datetime': 'Date'}
                )
                fig_travel.update_layout(height=400)
                st.plotly_chart(fig_travel, use_container_width=True, config=PLOTLY_CONFIG)

            with col2:
                # Delay trend
                fig_delay = px.line(
                    processed_data.sort_values('local_datetime'),
                    x='local_datetime',
                    y='average_delay',
                    title='Delays Over Time',
                    labels={'average_delay': 'Delay (seconds)', 'local_datetime': 'Date'},
                    color_discrete_sequence=['#e74c3c']
                )
                fig_delay.update_layout(height=400)
                st.plotly_chart(fig_delay, use_container_width=True, config=PLOTLY_CONFIG)

        # Data table
        with st.expander("📋 Raw Data", expanded=False):
            display_data = processed_data[
                ['local_datetime', 'average_traveltime', 'average_delay', 'average_speed']].copy()
            display_data.columns = ['Time', 'Travel Time (min)', 'Delay (sec)', 'Speed (mph)']
            st.dataframe(display_data, use_container_width=True)

            # Download button
            csv_data = display_data.to_csv(index=False)
            st.download_button(
                "⬇️ Download Data",
                csv_data,
                f"traffic_data_{start_location}_to_{end_location}.csv",
                "text/csv"
            )

# =========================
# Volume Analysis (Simplified)
# =========================
if not volume_df.empty and is_detailed_mode:
    st.markdown("## 🚗 Traffic Volume Analysis")

    # Simple intersection selector
    intersections = ["All Intersections"] + sorted(volume_df["intersection_name"].dropna().unique().tolist())
    selected_intersection = st.selectbox(
        "🚦 Choose intersection to analyze:",
        intersections,
        help="Select a specific intersection for detailed volume analysis"
    )

    # Process volume data
    volume_data = volume_df.copy()
    if selected_intersection != "All Intersections":
        volume_data = volume_data[volume_data["intersection_name"] == selected_intersection]

    # Filter by date range
    volume_data = volume_data[
        (volume_data["local_datetime"].dt.date >= selected_date_range[0]) &
        (volume_data["local_datetime"].dt.date <= selected_date_range[1])
        ]

    if not volume_data.empty:
        # Simple volume metrics
        volume_data["total_volume"] = pd.to_numeric(volume_data["total_volume"], errors="coerce")
        avg_volume = volume_data["total_volume"].mean()
        peak_volume = volume_data["total_volume"].max()
        utilization = (peak_volume / CAPACITY_LIMIT * 100) if CAPACITY_LIMIT else 0

        # Volume summary
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📊 Average Volume", f"{avg_volume:.0f} vehicles/hour")

        with col2:
            st.metric("🔥 Peak Volume", f"{peak_volume:.0f} vehicles/hour")

        with col3:
            utilization_status = "🟢 Good" if utilization < 60 else "🟡 Busy" if utilization < 80 else "🔴 Congested"
            st.metric("⚡ Peak Capacity Used", f"{utilization:.0f}%", delta=utilization_status)

        # Volume trend chart
        if len(volume_data) > 1:
            fig_volume = px.line(
                volume_data.sort_values('local_datetime'),
                x='local_datetime',
                y='total_volume',
                color='intersection_name' if selected_intersection == "All Intersections" else None,
                title='Traffic Volume Over Time',
                labels={'total_volume': 'Volume (vehicles/hour)', 'local_datetime': 'Date'}
            )

            # Add capacity line
            fig_volume.add_hline(
                y=CAPACITY_LIMIT,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Capacity Limit ({CAPACITY_LIMIT} vph)"
            )

            fig_volume.update_layout(height=400)
            st.plotly_chart(fig_volume, use_container_width=True, config=PLOTLY_CONFIG)

# =========================
# Recommendations Section
# =========================
st.markdown("## 🎯 Recommendations")

recommendations = []

if not processed_data.empty:
    avg_delay = processed_data['average_delay'].mean() if 'average_delay' in processed_data.columns else 0
    avg_travel_time = processed_data[
        'average_traveltime'].mean() if 'average_traveltime' in processed_data.columns else 0

    if avg_delay > 60:  # More than 1 minute delay
        recommendations.append("🚨 **High delays detected** - Consider alternative routes during peak hours")

    if avg_travel_time > 10:  # More than 10 minutes
        recommendations.append("⏰ **Long travel times** - Allow extra time for important trips")

    if avg_delay < 30 and avg_travel_time < 8:
        recommendations.append("✅ **Route performing well** - Current timing should work reliably")

if not recommendations:
    recommendations.append("📊 Analyzing your route... Check back with more data for personalized recommendations")

for i, rec in enumerate(recommendations, 1):
    st.markdown(f"{i}. {rec}")

# =========================
# Footer - Simplified
# =========================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666;">
    <p>🛣️ <strong>Washington Street Traffic Dashboard</strong></p>
    <p>Real-time traffic intelligence for better commute planning</p>
    <p style="font-size: 0.9rem;">© 2025 ADVANTEC Consulting Engineers, Inc.</p>
</div>
""", unsafe_allow_html=True)

# =========================
# Mobile Responsiveness Check
# =========================
# Add JavaScript to detect mobile and adjust layout
st.markdown("""
<script>
function checkMobile() {
    if (window.innerWidth < 768) {
        // Mobile adjustments
        document.querySelectorAll('.metric-large').forEach(el => {
            el.style.fontSize = '2rem';
        });

        document.querySelectorAll('.summary-card').forEach(el => {
            el.style.padding = '1rem';
            el.style.margin = '0.5rem 0';
        });
    }
}

// Run on load and resize
window.addEventListener('load', checkMobile);
window.addEventListener('resize', checkMobile);
</script>
""", unsafe_allow_html=True)