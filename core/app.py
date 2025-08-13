# Python
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Plotly (figures are created in helpers; keeping imports is harmless)
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
    volume_charts,
    date_range_preset_controls,
)

# Cycle length section (moved out)
from cycle_length_recommendations import render_cycle_length_section

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Active Transportation & Operations Management Dashboard",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Constants / Config
# =========================
THEORETICAL_LINK_CAPACITY_VPH = 1800
HIGH_VOLUME_THRESHOLD_VPH = 1200
CRITICAL_DELAY_SEC = 120
HIGH_DELAY_SEC = 60

# =========================
# CSS
# =========================
st.markdown("""
<style>
    .main-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 15px; padding: 2rem; margin: 1rem 0; color: white;
        box-shadow: 0 8px 32px rgba(30, 60, 114, 0.3);
    }
    .context-header {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem; border-radius: 15px; margin: 1rem 0 2rem; color: white; text-align: center;
        box-shadow: 0 8px 32px rgba(79, 172, 254, 0.3); backdrop-filter: blur(10px);
    }
    .context-header h2 { margin: 0; font-size: 2rem; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.3); }
    .context-header p { margin: 1rem 0 0; font-size: 1.1rem; opacity: 0.9; font-weight: 300; }
    @media (prefers-color-scheme: dark) { .context-header { background: linear-gradient(135deg, #2980b9 0%, #3498db 100%); } }

    .metric-container { background: rgba(79, 172, 254, 0.1); border: 1px solid rgba(79, 172, 254, 0.3);
        border-radius: 15px; padding: 1.5rem; margin: 1rem 0; backdrop-filter: blur(10px); transition: all 0.3s ease; }
    .metric-container:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(79, 172, 254, 0.2); }

    .insight-box {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.15) 0%, rgba(0, 242, 254, 0.15) 100%);
        border-left: 5px solid #4facfe; border-radius: 12px; padding: 1.25rem 1.5rem; margin: 1.25rem 0;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.1);
    }
    .insight-box h4 { color: #1e3c72; margin-top: 0; font-weight: 600; }

    .performance-badge { display: inline-block; padding: 0.35rem 0.9rem; border-radius: 25px; font-size: 0.85rem;
        font-weight: 600; margin: 0.2rem; border: 2px solid transparent; transition: all 0.3s ease; }
    .performance-badge:hover { transform: scale(1.05); border-color: rgba(255,255,255,0.25); }
    .badge-excellent { background: linear-gradient(45deg, #2ecc71, #27ae60); color: white; }
    .badge-good { background: linear-gradient(45deg, #3498db, #2980b9); color: white; }
    .badge-fair { background: linear-gradient(45deg, #f39c12, #e67e22); color: white; }
    .badge-poor { background: linear-gradient(45deg, #e74c3c, #c0392b); color: white; }
    .badge-critical { background: linear-gradient(45deg, #e74c3c, #8e44ad); color: white; animation: pulse 2s infinite; }
    @keyframes pulse { 0% {opacity:1} 50% {opacity:.7} 100% {opacity:1} }

    .stTabs [data-baseweb="tab-list"] { gap: 16px; }
    .stTabs [data-baseweb="tab"] { height: 56px; padding: 0 18px; border-radius: 12px;
        background: rgba(79, 172, 254, 0.1); border: 1px solid rgba(79, 172, 254, 0.2); }

    .chart-container { background: rgba(79, 172, 254, 0.05); border-radius: 15px; padding: 1rem; margin: 1rem 0;
        border: 1px solid rgba(79, 172, 254, 0.1); }

    .volume-metric { background: linear-gradient(135deg, rgba(52, 152, 219, 0.1), rgba(41, 128, 185, 0.1));
        border: 1px solid rgba(52, 152, 219, 0.3); border-radius: 12px; padding: 1rem; margin: 0.5rem 0; }

    .modebar { filter: saturate(0.85) opacity(0.9); }
</style>
""", unsafe_allow_html=True)

# =========================
# Title / Intro
# =========================
st.markdown("""
<div class="main-container">
    <h1 style="text-align:center; margin:0; font-size:2.5rem; font-weight:800;">
        🛣️ Active Transportation & Operations Management Dashboard
    </h1>
    <p style="text-align:center; margin-top:1rem; font-size:1.1rem; opacity:0.9;">
        Advanced Traffic Engineering & Operations Management Platform
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    font-size: 1.05rem; font-weight: 400; color: var(--text-color);
    background: linear-gradient(135deg, rgba(79, 172, 254, 0.1), rgba(0, 242, 254, 0.05));
    padding: 1.5rem; border-radius: 18px; box-shadow: 0 8px 32px rgba(79,172,254,0.08);
    margin: 1.25rem 0; line-height: 1.7; border: 1px solid rgba(79,172,254,0.2); backdrop-filter: blur(8px);
">
    <div style="text-align:center; margin-bottom: 0.5rem;">
        <strong style="font-size: 1.2rem; color: #2980b9;">🚀 The ADVANTEC Platform</strong>
    </div>
    <p>Leverages <strong>millions of data points</strong> trained on advanced Machine Learning algorithms to optimize traffic flow, reduce travel time, minimize fuel consumption, and decrease greenhouse gas emissions across the Coachella Valley transportation network.</p>
    <p><strong>Key Capabilities:</strong> Real-time anomaly detection • Intelligent cycle length optimization • Predictive traffic modeling • Performance analytics</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 1.1rem; border-radius: 15px;
    margin: 1rem 0; text-align: center; box-shadow: 0 6px 20px rgba(52, 152, 219, 0.25);">
    <h3 style="margin:0; font-weight:600;">🔍 Research Question</h3>
    <p style="margin: 0.45rem 0 0; font-size: 1.0rem;">What are the main bottlenecks (slowest intersections) on Washington St that are most prone to causing increased travel times?</p>
</div>
""", unsafe_allow_html=True)

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["1️⃣ ITERIS CLEARGUIDE DATA", "2️⃣ KINETIC MOBILITY DATA"])

# -------------------------
# TAB 1: Performance / Travel Time
# -------------------------
with tab1:
    st.header("*🚧 Analyzing Speed, Delay, and Travel Time*")

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Loading corridor performance data...")
    progress_bar.progress(25)

    corridor_df = get_corridor_df()
    progress_bar.progress(100)

    if corridor_df.empty:
        st.error("❌ Failed to load corridor data. Please check your data sources.")
    else:
        status_text.text("✅ Data loaded successfully!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

        with st.sidebar:
            with st.expander("🚧 Performance Analysis Controls", expanded=False):
                seg_options = ["All Segments"] + sorted(corridor_df["segment_name"].dropna().unique().tolist())
                corridor = st.selectbox(
                    "🛣️ Select Corridor Segment",
                    seg_options,
                    help="Choose a specific segment or analyze all segments",
                )

                min_date = corridor_df["local_datetime"].dt.date.min()
                max_date = corridor_df["local_datetime"].dt.date.max()

                st.markdown("#### 📅 Analysis Period")
                date_range = date_range_preset_controls(min_date, max_date, key_prefix="perf")

                st.markdown("#### ⏰ Analysis Settings")
                granularity = st.selectbox(
                    "Data Aggregation",
                    ["Hourly", "Daily", "Weekly", "Monthly"],
                    index=0,
                    help="Higher aggregation smooths trends but may hide peaks",
                )

                time_filter, start_hour, end_hour = None, None, None
                if granularity == "Hourly":
                    time_filter = st.selectbox(
                        "Time Period Focus",
                        [
                            "All Hours",
                            "Peak Hours (7–9 AM, 4–6 PM)",
                            "AM Peak (7–9 AM)",
                            "PM Peak (4–6 PM)",
                            "Off-Peak",
                            "Custom Range",
                        ],
                    )
                    if time_filter == "Custom Range":
                        c1, c2 = st.columns(2)
                        with c1:
                            start_hour = st.number_input("Start Hour (0–23)", 0, 23, 7, step=1)
                        with c2:
                            end_hour = st.number_input("End Hour (1–24)", 1, 24, 18, step=1)

        if len(date_range) == 2:
            try:
                base_df = corridor_df.copy()
                if corridor != "All Segments":
                    base_df = base_df[base_df["segment_name"] == corridor]

                if base_df.empty:
                    st.warning("⚠️ No data for the selected segment.")
                else:
                    filtered_data = process_traffic_data(
                        base_df,
                        date_range,
                        granularity,
                        time_filter if granularity == "Hourly" else None,
                        start_hour,
                        end_hour,
                    )

                    if filtered_data.empty:
                        st.warning("⚠️ No data available for the selected filters.")
                    else:
                        total_records = len(filtered_data)
                        data_span = (date_range[1] - date_range[0]).days + 1
                        time_context = f" • {time_filter}" if (granularity == "Hourly" and time_filter) else ""

                        st.markdown(
                            f"""
                        <div class="context-header">
                            <h2>📊 {corridor}</h2>
                            <p>📅 {date_range[0].strftime('%b %d, %Y')} to {date_range[1].strftime('%b %d, %Y')}
                            ({data_span} days) • {granularity} Aggregation{time_context}</p>
                            <p>📈 Analyzing {total_records:,} data points across the selected period</p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                        raw_data = base_df[
                            (base_df["local_datetime"].dt.date >= date_range[0])
                            & (base_df["local_datetime"].dt.date <= date_range[1])
                        ].copy()

                        if raw_data.empty:
                            st.info("No raw hourly data in this window.")
                        else:
                            for col in ["average_delay", "average_traveltime", "average_speed"]:
                                if col in raw_data:
                                    raw_data[col] = pd.to_numeric(raw_data[col], errors="coerce")

                            col1, col2, col3, col4, col5 = st.columns(5)

                            with col1:
                                worst_delay = (
                                    float(np.nanmax(raw_data["average_delay"]))
                                    if raw_data["average_delay"].notna().any()
                                    else 0.0
                                )
                                p95_delay = (
                                    float(np.nanpercentile(raw_data["average_delay"].dropna(), 95))
                                    if raw_data["average_delay"].notna().any()
                                    else 0.0
                                )
                                rating, badge = get_performance_rating(100 - min(worst_delay / 2, 100))
                                st.metric("🚨 Peak Delay", f"{worst_delay:.1f}s", delta=f"95th: {p95_delay:.1f}s")
                                st.markdown(
                                    f'<span class="performance-badge {badge}">{rating}</span>',
                                    unsafe_allow_html=True,
                                )

                            with col2:
                                worst_tt = (
                                    float(np.nanmax(raw_data["average_traveltime"]))
                                    if raw_data["average_traveltime"].notna().any()
                                    else 0.0
                                )
                                avg_tt = (
                                    float(np.nanmean(raw_data["average_traveltime"]))
                                    if raw_data["average_traveltime"].notna().any()
                                    else 0.0
                                )
                                tt_delta = ((worst_tt - avg_tt) / avg_tt * 100) if avg_tt > 0 else 0
                                impact_rating, badge = get_performance_rating(100 - min(max(tt_delta, 0), 100))
                                st.metric("⏱️ Peak Travel Time", f"{worst_tt:.1f}min", delta=f"+{tt_delta:.0f}% vs avg")
                                st.markdown(
                                    f'<span class="performance-badge {badge}">{impact_rating}</span>',
                                    unsafe_allow_html=True,
                                )

                            with col3:
                                slowest = (
                                    float(np.nanmin(raw_data["average_speed"]))
                                    if raw_data["average_speed"].notna().any()
                                    else 0.0
                                )
                                avg_speed = (
                                    float(np.nanmean(raw_data["average_speed"]))
                                    if raw_data["average_speed"].notna().any()
                                    else 0.0
                                )
                                speed_drop = ((avg_speed - slowest) / avg_speed * 100) if avg_speed > 0 else 0
                                speed_rating, badge = get_performance_rating(min(slowest * 2, 100))
                                st.metric("🐌 Minimum Speed", f"{slowest:.1f}mph", delta=f"-{speed_drop:.0f}% vs avg")
                                st.markdown(
                                    f'<span class="performance-badge {badge}">{speed_rating}</span>',
                                    unsafe_allow_html=True,
                                )

                            with col4:
                                if avg_tt > 0:
                                    cv_tt = float(np.nanstd(raw_data["average_traveltime"]) / avg_tt) * 100
                                else:
                                    cv_tt = 0.0
                                reliability = max(0, 100 - cv_tt)
                                rel_rating, badge = get_performance_rating(reliability)
                                st.metric("🎯 Reliability Index", f"{reliability:.0f}%", delta=f"CV: {cv_tt:.1f}%")
                                st.markdown(
                                    f'<span class="performance-badge {badge}">{rel_rating}</span>',
                                    unsafe_allow_html=True,
                                )

                            with col5:
                                high_delay_pct = (
                                    (raw_data["average_delay"] > HIGH_DELAY_SEC).mean() * 100
                                    if raw_data["average_delay"].notna().any()
                                    else 0.0
                                )
                                hours = (
                                    int((raw_data["average_delay"] > HIGH_DELAY_SEC).sum())
                                    if raw_data["average_delay"].notna().any()
                                    else 0
                                )
                                freq_rating, badge = get_performance_rating(100 - high_delay_pct)
                                st.metric("⚠️ Congestion Frequency", f"{high_delay_pct:.1f}%", delta=f"{hours} hours")
                                st.markdown(
                                    f'<span class="performance-badge {badge}">{freq_rating}</span>',
                                    unsafe_allow_html=True,
                                )

                        if len(filtered_data) > 1:
                            st.subheader("📈 Performance Trends")
                            v1, v2 = st.columns(2)
                            with v1:
                                dc = performance_chart(filtered_data, "delay")
                                if dc:
                                    st.plotly_chart(dc, use_container_width=True)
                            with v2:
                                tc = performance_chart(filtered_data, "travel")
                                if tc:
                                    st.plotly_chart(tc, use_container_width=True)

                        if not raw_data.empty:
                            worst_delay = (
                                float(np.nanmax(raw_data["average_delay"]))
                                if raw_data["average_delay"].notna().any()
                                else 0.0
                            )
                            avg_tt = (
                                float(np.nanmean(raw_data["average_traveltime"]))
                                if raw_data["average_traveltime"].notna().any()
                                else 0.0
                            )
                            worst_tt = (
                                float(np.nanmax(raw_data["average_traveltime"]))
                                if raw_data["average_traveltime"].notna().any()
                                else 0.0
                            )
                            tt_delta = ((worst_tt - avg_tt) / avg_tt * 100) if avg_tt > 0 else 0
                            if avg_tt > 0:
                                cv_tt = float(np.nanstd(raw_data["average_traveltime"]) / avg_tt) * 100
                            else:
                                cv_tt = 0.0
                            reliability = max(0, 100 - cv_tt)
                            high_delay_pct = (
                                (raw_data["average_delay"] > HIGH_DELAY_SEC).mean() * 100
                                if raw_data["average_delay"].notna().any()
                                else 0.0
                            )
                            st.markdown(
                                f"""
                            <div class="insight-box">
                                <h4>💡 Advanced Performance Insights</h4>
                                <p><strong>📊 Data Overview:</strong> {len(filtered_data):,} {granularity.lower()} observations across {(date_range[1] - date_range[0]).days + 1} days.</p>
                                <p><strong>🚨 Peaks:</strong> Delay up to {worst_delay:.0f}s ({worst_delay / 60:.1f} min) • Travel time up to {worst_tt:.1f} min (+{tt_delta:.0f}% vs avg).</p>
                                <p><strong>🎯 Reliability:</strong> {reliability:.0f}% travel time reliability • Delays > {HIGH_DELAY_SEC}s occur {high_delay_pct:.1f}% of hours.</p>
                                <p><strong>📌 Action:</strong> {"Critical intervention needed" if worst_delay > CRITICAL_DELAY_SEC else "Optimization recommended" if worst_delay > HIGH_DELAY_SEC else "Monitor trends"}.</p>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                        st.subheader("🚨 Comprehensive Bottleneck Analysis")
                        if not raw_data.empty:
                            try:
                                g = raw_data.groupby(["segment_name", "direction"]).agg(
                                    average_delay_mean=("average_delay", "mean"),
                                    average_delay_max=("average_delay", "max"),
                                    average_traveltime_mean=("average_traveltime", "mean"),
                                    average_traveltime_max=("average_traveltime", "max"),
                                    average_speed_mean=("average_speed", "mean"),
                                    average_speed_min=("average_speed", "min"),
                                    n=("average_delay", "count"),
                                ).reset_index()

                                def _norm(s):
                                    s = s.astype(float)
                                    mn, mx = np.nanmin(s), np.nanmax(s)
                                    if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
                                        return (s - mn) / (mx - mn)
                                    return pd.Series(np.zeros(len(s)), index=s.index)

                                score = (
                                    0.45 * _norm(g["average_delay_max"])
                                    + 0.35 * _norm(g["average_delay_mean"])
                                    + 0.20 * _norm(g["average_traveltime_max"])
                                ) * 100
                                g["Bottleneck_Score"] = score.round(1)

                                bins = [-0.1, 20, 40, 60, 80, 200]
                                labels = ["🟢 Excellent", "🔵 Good", "🟡 Fair", "🟠 Poor", "🔴 Critical"]
                                g["🎯 Performance Rating"] = pd.cut(g["Bottleneck_Score"], bins=bins, labels=labels)

                                final = g[
                                    [
                                        "segment_name",
                                        "direction",
                                        "🎯 Performance Rating",
                                        "Bottleneck_Score",
                                        "average_delay_mean",
                                        "average_delay_max",
                                        "average_traveltime_mean",
                                        "average_traveltime_max",
                                        "average_speed_mean",
                                        "average_speed_min",
                                        "n",
                                    ]
                                ].rename(
                                    columns={
                                        "segment_name": "Segment",
                                        "direction": "Dir",
                                        "average_delay_mean": "Avg Delay (s)",
                                        "average_delay_max": "Peak Delay (s)",
                                        "average_traveltime_mean": "Avg Time (min)",
                                        "average_traveltime_max": "Peak Time (min)",
                                        "average_speed_mean": "Avg Speed (mph)",
                                        "average_speed_min": "Min Speed (mph)",
                                        "n": "Obs",
                                    }
                                ).sort_values("Bottleneck_Score", ascending=False)

                                st.dataframe(
                                    final.head(15),
                                    use_container_width=True,
                                    column_config={
                                        "Bottleneck_Score": st.column_config.NumberColumn(
                                            "🚨 Impact Score",
                                            help="Composite (0–100); higher ⇒ worse",
                                            format="%.1f",
                                        )
                                    },
                                )

                                st.download_button(
                                    "⬇️ Download Bottleneck Table (CSV)",
                                    data=final.to_csv(index=False).encode("utf-8"),
                                    file_name="bottlenecks.csv",
                                    mime="text/csv",
                                )
                                st.download_button(
                                    "⬇️ Download Filtered Performance (CSV)",
                                    data=filtered_data.to_csv(index=False).encode("utf-8"),
                                    file_name="performance_filtered.csv",
                                    mime="text/csv",
                                )
                            except Exception as e:
                                st.error(f"❌ Error in performance analysis: {e}")

            except Exception as e:
                st.error(f"❌ Error processing traffic data: {e}")
        else:
            st.warning("⚠️ Please select both start and end dates to proceed.")

# -------------------------
# TAB 2: Volume / Capacity
# -------------------------
with tab2:
    st.header("📊 Advanced Traffic Demand & Capacity Analysis")

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Loading traffic demand data...")
    progress_bar.progress(25)

    volume_df = get_volume_df()
    progress_bar.progress(100)

    if volume_df.empty:
        st.error("❌ Failed to load volume data. Please check your data sources.")
    else:
        status_text.text("✅ Volume data loaded successfully!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

        with st.sidebar:
            with st.expander("📊 Volume Analysis Controls", expanded=False):
                intersections = ["All Intersections"] + sorted(
                    volume_df["intersection_name"].dropna().unique().tolist()
                )
                intersection = st.selectbox("🚦 Select Intersection", intersections)

                min_date = volume_df["local_datetime"].dt.date.min()
                max_date = volume_df["local_datetime"].dt.date.max()

                st.markdown("#### 📅 Analysis Period")
                date_range_vol = date_range_preset_controls(min_date, max_date, key_prefix="vol")

                st.markdown("#### ⏰ Analysis Settings")
                granularity_vol = st.selectbox("Data Aggregation", ["Hourly", "Daily", "Weekly", "Monthly"], index=0)

                direction_options = ["All Directions"] + sorted(volume_df["direction"].dropna().unique().tolist())
                direction_filter = st.selectbox("🔄 Direction Filter", direction_options)

        if len(date_range_vol) == 2:
            try:
                base_df = volume_df.copy()
                if intersection != "All Intersections":
                    base_df = base_df[base_df["intersection_name"] == intersection]
                if direction_filter != "All Directions":
                    base_df = base_df[base_df["direction"] == direction_filter]

                if base_df.empty:
                    st.warning("⚠️ No volume data for the selected filters.")
                else:
                    filtered_volume_data = process_traffic_data(base_df, date_range_vol, granularity_vol)

                    if filtered_volume_data.empty:
                        st.warning("⚠️ No volume data available for the selected range.")
                    else:
                        span = (date_range_vol[1] - date_range_vol[0]).days + 1
                        st.markdown(
                            f"""
                        <div class="context-header">
                            <h2>📊 Volume Analysis: {intersection}</h2>
                            <p>📅 {date_range_vol[0].strftime('%b %d, %Y')} to {date_range_vol[1].strftime('%b %d, %Y')}
                            ({span} days) • {granularity_vol} Aggregation</p>
                            <p>📈 {len(filtered_volume_data):,} observations • Direction: {direction_filter}</p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                        raw = base_df[
                            (base_df["local_datetime"].dt.date >= date_range_vol[0])
                            & (base_df["local_datetime"].dt.date <= date_range_vol[1])
                        ].copy()

                        st.subheader("🚦 Traffic Demand Performance Indicators")
                        if raw.empty:
                            st.info("No raw hourly volume in this window.")
                        else:
                            raw["total_volume"] = pd.to_numeric(raw["total_volume"], errors="coerce")
                            col1, col2, col3, col4, col5 = st.columns(5)

                            with col1:
                                peak = float(np.nanmax(raw["total_volume"])) if raw["total_volume"].notna().any() else 0
                                p95 = (
                                    float(np.nanpercentile(raw["total_volume"].dropna(), 95))
                                    if raw["total_volume"].notna().any()
                                    else 0
                                )
                                util = (peak / THEORETICAL_LINK_CAPACITY_VPH) * 100 if THEORETICAL_LINK_CAPACITY_VPH else 0
                                if util > 90:
                                    badge = "badge-critical"
                                elif util > 75:
                                    badge = "badge-poor"
                                elif util > 60:
                                    badge = "badge-fair"
                                else:
                                    badge = "badge-good"
                                st.metric("🔥 Peak Demand", f"{peak:,.0f} vph", delta=f"95th: {p95:,.0f}")
                                st.markdown(
                                    f'<span class="performance-badge {badge}">{util:.0f}% Capacity</span>',
                                    unsafe_allow_html=True,
                                )

                            with col2:
                                avg = float(np.nanmean(raw["total_volume"])) if raw["total_volume"].notna().any() else 0
                                med = float(np.nanmedian(raw["total_volume"])) if raw["total_volume"].notna().any() else 0
                                st.metric("📊 Average Demand", f"{avg:,.0f} vph", delta=f"Median: {med:,.0f}")
                                avg_util = (avg / THEORETICAL_LINK_CAPACITY_VPH) * 100 if THEORETICAL_LINK_CAPACITY_VPH else 0
                                badge = "badge-good" if avg_util <= 40 else ("badge-fair" if avg_util <= 60 else "badge-poor")
                                st.markdown(
                                    f'<span class="performance-badge {badge}">{avg_util:.0f}% Avg Util</span>',
                                    unsafe_allow_html=True,
                                )

                            with col3:
                                ratio = (peak / avg) if avg > 0 else 0
                                st.metric("📈 Peak/Average Ratio", f"{ratio:.1f}x", help="Higher ⇒ more peaked demand")
                                badge = "badge-good" if ratio <= 2 else ("badge-fair" if ratio <= 3 else "badge-poor")
                                state = "Low" if ratio <= 2 else ("Moderate" if ratio <= 3 else "High")
                                st.markdown(
                                    f'<span class="performance-badge {badge}">{state} Peaking</span>',
                                    unsafe_allow_html=True,
                                )

                            with col4:
                                cv = (float(np.nanstd(raw["total_volume"])) / avg * 100) if avg > 0 else 0
                                st.metric("🎯 Demand Consistency", f"{max(0, 100 - cv):.0f}%", delta=f"CV: {cv:.1f}%")
                                badge = "badge-good" if cv < 30 else ("badge-fair" if cv < 50 else "badge-poor")
                                label = "Consistent" if cv < 30 else ("Variable" if cv < 50 else "Highly Variable")
                                st.markdown(
                                    f'<span class="performance-badge {badge}">{label}</span>',
                                    unsafe_allow_html=True,
                                )

                            with col5:
                                high_hours = int((raw["total_volume"] > HIGH_VOLUME_THRESHOLD_VPH).sum())
                                total_hours = int(raw["total_volume"].count())
                                risk_pct = (high_hours / total_hours * 100) if total_hours > 0 else 0
                                st.metric("⚠️ High Volume Hours", f"{high_hours}", delta=f"{risk_pct:.1f}% of time")
                                if risk_pct > 25:
                                    badge = "badge-critical"
                                elif risk_pct > 15:
                                    badge = "badge-poor"
                                elif risk_pct > 5:
                                    badge = "badge-fair"
                                else:
                                    badge = "badge-good"
                                level = (
                                    "Very High"
                                    if risk_pct > 25
                                    else ("High" if risk_pct > 15 else ("Moderate" if risk_pct > 5 else "Low"))
                                )
                                st.markdown(
                                    f'<span class="performance-badge {badge}">{level} Risk</span>',
                                    unsafe_allow_html=True,
                                )

                        st.subheader("📈 Volume Analysis Visualizations")
                        if len(filtered_volume_data) > 1:
                            chart1, chart2, chart3 = volume_charts(
                                filtered_volume_data,
                                THEORETICAL_LINK_CAPACITY_VPH,
                                HIGH_VOLUME_THRESHOLD_VPH,
                            )
                            if chart1:
                                st.plotly_chart(chart1, use_container_width=True)
                            colA, colB = st.columns(2)
                            with colA:
                                if chart3:
                                    st.plotly_chart(chart3, use_container_width=True)
                            with colB:
                                if chart2:
                                    st.plotly_chart(chart2, use_container_width=True)

                        if not raw.empty:
                            peak = float(np.nanmax(raw["total_volume"])) if raw["total_volume"].notna().any() else 0
                            avg = float(np.nanmean(raw["total_volume"])) if raw["total_volume"].notna().any() else 0
                            ratio = (peak / avg) if avg > 0 else 0
                            cv = (float(np.nanstd(raw["total_volume"])) / avg * 100) if avg > 0 else 0
                            util = (peak / THEORETICAL_LINK_CAPACITY_VPH) * 100 if THEORETICAL_LINK_CAPACITY_VPH else 0
                            high_hours = int((raw["total_volume"] > HIGH_VOLUME_THRESHOLD_VPH).sum())
                            total_hours = int(raw["total_volume"].count())
                            risk_pct = (high_hours / total_hours * 100) if total_hours > 0 else 0

                            action = (
                                "Immediate capacity expansion needed"
                                if util > 90
                                else "Consider signal optimization"
                                if util > 75
                                else "Monitor trends & optimize timing"
                                if util > 60
                                else "Current capacity appears adequate"
                            )

                            st.markdown(
                                f"""
                            <div class="insight-box">
                                <h4>💡 Advanced Volume Analysis Insights</h4>
                                <p><strong>📊 Capacity:</strong> Peak {peak:,.0f} vph ({util:.0f}% of {THEORETICAL_LINK_CAPACITY_VPH:,} vph) • Avg {avg:,.0f} vph.</p>
                                <p><strong>📈 Demand Shape:</strong> {ratio:.1f}× peak-to-average • Consistency {max(0, 100 - cv):.0f}%.</p>
                                <p><strong>⚠️ Risk:</strong> >{HIGH_VOLUME_THRESHOLD_VPH:,} vph occurs {high_hours} hours ({risk_pct:.1f}% of period).</p>
                                <p><strong>🎯 Recommendation:</strong> {action}.</p>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                        st.subheader("🚨 Intersection Volume & Capacity Risk Analysis")
                        try:
                            g = raw.groupby(["intersection_name", "direction"]).agg(
                                total_volume_mean=("total_volume", "mean"),
                                total_volume_max=("total_volume", "max"),
                                total_volume_std=("total_volume", "std"),
                                total_volume_count=("total_volume", "count"),
                            ).reset_index()

                            g["Peak_Capacity_Util"] = (
                                g["total_volume_max"] / THEORETICAL_LINK_CAPACITY_VPH * 100
                            ).round(1)
                            g["Avg_Capacity_Util"] = (
                                g["total_volume_mean"] / THEORETICAL_LINK_CAPACITY_VPH * 100
                            ).round(1)
                            g["Volume_Variability"] = (
                                g["total_volume_std"] / g["total_volume_mean"] * 100
                            ).replace([np.inf, -np.inf], np.nan).fillna(0).round(1)
                            g["Peak_Avg_Ratio"] = (
                                g["total_volume_max"] / g["total_volume_mean"]
                            ).replace([np.inf, -np.inf], 0).fillna(0).round(1)

                            g["🚨 Risk Score"] = (
                                0.5 * g["Peak_Capacity_Util"]
                                + 0.3 * g["Avg_Capacity_Util"]
                                + 0.2 * (g["Peak_Avg_Ratio"] * 10)
                            ).round(1)

                            g["⚠️ Risk Level"] = pd.cut(
                                g["🚨 Risk Score"],
                                bins=[0, 40, 60, 80, 90, 999],
                                labels=["🟢 Low Risk", "🟡 Moderate Risk", "🟠 High Risk", "🔴 Critical Risk", "🚨 Severe Risk"],
                                include_lowest=True,
                            )
                            g["🎯 Action Priority"] = pd.cut(
                                g["Peak_Capacity_Util"],
                                bins=[0, 60, 75, 90, 999],
                                labels=["🟢 Monitor", "🟡 Optimize", "🟠 Upgrade", "🔴 Urgent"],
                                include_lowest=True,
                            )

                            final = g[
                                [
                                    "intersection_name",
                                    "direction",
                                    "⚠️ Risk Level",
                                    "🎯 Action Priority",
                                    "🚨 Risk Score",
                                    "Peak_Capacity_Util",
                                    "Avg_Capacity_Util",
                                    "total_volume_mean",
                                    "total_volume_max",
                                    "Peak_Avg_Ratio",
                                    "total_volume_count",
                                ]
                            ].rename(
                                columns={
                                    "intersection_name": "Intersection",
                                    "direction": "Dir",
                                    "Peak_Capacity_Util": "📊 Peak Capacity %",
                                    "Avg_Capacity_Util": "📊 Avg Capacity %",
                                    "total_volume_mean": "Avg Volume (vph)",
                                    "total_volume_max": "Peak Volume (vph)",
                                    "total_volume_count": "Data Points",
                                }
                            ).sort_values("🚨 Risk Score", ascending=False)

                            st.dataframe(
                                final.head(15),
                                use_container_width=True,
                                column_config={
                                    "🚨 Risk Score": st.column_config.NumberColumn(
                                        "🚨 Capacity Risk Score",
                                        help="Composite of peak/avg util + peaking",
                                        format="%.1f",
                                        min_value=0,
                                        max_value=120,
                                    ),
                                    "📊 Peak Capacity %": st.column_config.NumberColumn("📊 Peak Capacity %", format="%.1f%%"),
                                    "📊 Avg Capacity %": st.column_config.NumberColumn("📊 Avg Capacity %", format="%.1f%%"),
                                },
                            )

                            st.download_button(
                                "⬇️ Download Capacity Risk Table (CSV)",
                                data=final.to_csv(index=False).encode("utf-8"),
                                file_name="capacity_risk.csv",
                                mime="text/csv",
                            )
                            st.download_button(
                                "⬇️ Download Filtered Volume (CSV)",
                                data=filtered_volume_data.to_csv(index=False).encode("utf-8"),
                                file_name="volume_filtered.csv",
                                mime="text/csv",
                            )
                        except Exception as e:
                            st.error(f"❌ Error in volume analysis: {e}")
                            simple = raw.groupby(["intersection_name", "direction"]).agg(
                                Avg=("total_volume", "mean"), Peak=("total_volume", "max")
                            ).reset_index().sort_values("Peak", ascending=False)
                            st.dataframe(simple, use_container_width=True)

                        # Cycle Length Recommendations section (moved to separate module)
                        render_cycle_length_section(raw)

            except Exception as e:
                st.error(f"❌ Error processing volume data: {e}")
                st.info("Please check your data sources and try again.")
        else:
            st.warning("⚠️ Please select both start and end dates to proceed with the volume analysis.")

# =========================
# Footer (refined styling + updated tagline)
# =========================
import streamlit.components.v1 as components

footer_html = """
<div style="text-align:center; padding: 1.25rem;
    background: linear-gradient(135deg, rgba(79,172,254,0.1), rgba(0,242,254,0.05));
    border-radius: 15px; margin-top: 1rem; border: 1px solid rgba(79,172,254,0.2);
    font-family: system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif;">

  <h4 style="color:#2980b9; margin:0 0 .4rem; font-weight:700;">
    🛣️ Active Transportation & Operations Management Dashboard
  </h4>
  <p style="opacity:.85; margin:.1rem 0 0; font-size:1.0rem;">
    Powered by Advanced Machine Learning • Real-time Traffic Intelligence • Intelligent Transportation Solutions
  </p>

  <!-- Social and website row -->
  <div style="display:flex; justify-content:center; align-items:center; gap:14px; margin:12px 0 8px;">
    <!-- Instagram (IG text badge for clarity/consistency) -->
    <a href="https://www.instagram.com/advantec98/" target="_blank" rel="noopener noreferrer" aria-label="Instagram"
       style="width:40px;height:40px;display:grid;place-items:center;border-radius:50%;
              background:#ffffff; box-shadow:0 2px 8px rgba(0,0,0,0.08); text-decoration:none;
              color:#444; border:1px solid rgba(41,128,185,.25); transition:transform .15s ease, box-shadow .15s ease;">
      <span style="font:700 13px/1 system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial;">IG</span>
    </a>

    <!-- LinkedIn -->
    <a href="https://www.linkedin.com/company/advantec-consulting-engineers-inc./posts/?feedView=all"
       target="_blank" rel="noopener noreferrer" aria-label="LinkedIn"
       style="width:40px;height:40px;display:grid;place-items:center;border-radius:50%;
              background:#ffffff; box-shadow:0 2px 8px rgba(0,0,0,0.08); text-decoration:none;
              border:1px solid rgba(41,128,185,.25); transition:transform .15s ease, box-shadow .15s ease;">
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 448 512" aria-hidden="true" role="img">
        <path fill="#0A66C2"
              d="M100.28 448H7.4V148.9h92.88zM53.79 108.1C24.09 108.1 0 83.5 0 53.8 0 24.1 24.1 0 53.79 0s53.8 24.1 53.8 53.8c0 29.7-24.1 54.3-53.8 54.3zM447.9 448h-92.68V302.4c0-34.7-.7-79.3-48.3-79.3-48.3 0-55.7 37.7-55.7 76.6V448h-92.7V148.9h89V185h1.3c12.4-23.6 42.7-48.3 87.8-48.3 93.9 0 111.2 61.8 111.2 142.3V448z"/>
      </svg>
    </a>

    <!-- Facebook -->
    <a href="https://www.facebook.com/advantecconsultingUSA" target="_blank" rel="noopener noreferrer" aria-label="Facebook"
       style="width:40px;height:40px;display:grid;place-items:center;border-radius:50%;
              background:#ffffff; box-shadow:0 2px 8px rgba(0,0,0,0.08); text-decoration:none;
              border:1px solid rgba(41,128,185,.25); transition:transform .15s ease, box-shadow .15s ease;">
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 320 512" aria-hidden="true" role="img">
        <path fill="#1877F2"
              d="M279.14 288l14.22-92.66h-88.91v-60.13c0-25.35 12.42-50.06 52.24-50.06h40.42V6.26S263.61 0 225.36 0c-73.22 0-121 44.38-121 124.72v70.62H22.89V288h81.47v224h100.2V288z"/>
      </svg>
    </a>

    <!-- Website: clear pill button with label for clarity -->
    <a href="https://advantec-usa.com/" target="_blank" rel="noopener noreferrer" aria-label="ADVANTEC Website"
       style="height:40px; display:inline-flex; align-items:center; gap:8px; padding:0 12px;
              border-radius:9999px; background:#ffffff; box-shadow:0 2px 8px rgba(0,0,0,0.08);
              text-decoration:none; border:1px solid #2980b9; color:#2980b9; font-weight:700;
              transition:transform .15s ease, box-shadow .15s ease;">
      <span style="font-size:18px; line-height:1;">🌐</span>
      <span>Website</span>
    </a>
  </div>

  <p style="opacity:.65; margin:.2rem 0 0; font-size:.9rem;">
    © 2025 ADVANTEC Consulting Engineers, Inc. — "Because We Care"
  </p>
</div>

<script>
  // Subtle hover lift for all interactive items in the row
  (function(){
    const items = document.currentScript.previousElementSibling.querySelectorAll('a');
    items.forEach(el => {
      el.addEventListener('mouseenter', () => { el.style.transform = 'translateY(-1px)'; });
      el.addEventListener('mouseleave', () => { el.style.transform = 'translateY(0)'; });
    });
  })();
</script>
"""

components.html(footer_html, height=200)