# Python
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Plotly (figures are created in helpers)
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
    compute_perf_kpis_interpretable,
    render_badge,
)

# Cycle length section (optional)
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

    .insight-box {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.15) 0%, rgba(0, 242, 254, 0.15) 100%);
        border-left: 5px solid #4facfe; border-radius: 12px; padding: 1.25rem 1.5rem; margin: 1.25rem 0;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.1);
    }

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
    <p>Leverages <strong>millions of data points</strong> trained on advanced Machine Learning algorithms to optimize traffic flow, reduce travel time, minimize fuel consumption, and decrease greenhouse gas emissions.</p>
    <p><strong>Key Capabilities:</strong> Real-time anomaly detection • Intelligent cycle length optimization • Predictive traffic modeling • Performance analytics</p>
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

    status_text = st.empty()
    status_text.info("Loading corridor performance data...")
    corridor_df = get_corridor_df()
    status_text.empty()

    if corridor_df.empty:
        st.error("❌ Failed to load corridor data.")
        st.stop()

    with st.sidebar:
        with st.expander("🚧 Performance Analysis Controls", expanded=False):
            seg_options = ["All Segments"] + sorted(corridor_df["segment_name"].dropna().unique().tolist())
            corridor = st.selectbox(
                "🛣️ Select Corridor Segment",
                seg_options,
                help="Choose a specific segment or analyze all segments",
            )

            # Direction selector (do not mix NB/SB)
            dir_options_perf = sorted(corridor_df["direction"].dropna().unique().tolist())
            selected_direction = st.selectbox(
                "🔄 Direction",
                dir_options_perf,
                help="Analyze a single direction (no mixing)",
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
                key="granularity_perf",
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
                    key="time_period_focus_perf",
                )
                if time_filter == "Custom Range":
                    c1, c2 = st.columns(2)
                    with c1:
                        start_hour = st.number_input("Start Hour (0–23)", 0, 23, 7, step=1, key="start_hour_perf")
                    with c2:
                        end_hour = st.number_input("End Hour (1–24)", 1, 24, 18, step=1, key="end_hour_perf")

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
                        # Coerce numeric
                        for col in ["average_delay", "average_traveltime", "average_speed"]:
                            if col in raw_data:
                                raw_data[col] = pd.to_numeric(raw_data[col], errors="coerce")

                        # Build a single-direction dataset for KPIs (never mix NB/SB)
                        df_for_kpi = raw_data.copy()
                        if "direction" in df_for_kpi.columns:
                            df_for_kpi = df_for_kpi[df_for_kpi["direction"] == selected_direction]

                        # If user chose All Segments, sum minutes across segments at each timestamp
                        if corridor == "All Segments" and not df_for_kpi.empty:
                            df_for_kpi = (
                                df_for_kpi.groupby(["local_datetime"], as_index=False)
                                .agg(
                                    average_traveltime=("average_traveltime", "sum"),  # minutes across segments
                                    average_delay=("average_delay", "mean"),            # keep delay as mean
                                    average_speed=("average_speed", "mean"),            # optional
                                )
                            )

                        # --- Five KPI row (interpretable + badges) ---
                        k = compute_perf_kpis_interpretable(df_for_kpi, HIGH_DELAY_SEC)

                        c1, c2, c3, c4, c5 = st.columns(5)
                        with c1:
                            st.metric(
                                "🎯 Reliability Index",
                                f"{k['reliability']['value']:.0f}{k['reliability']['unit']}",
                                help=k['reliability']['help'],
                            )
                            st.markdown(render_badge(k['reliability']['score']), unsafe_allow_html=True)

                        with c2:
                            st.metric(
                                "⚠️ Congestion Frequency",
                                f"{k['congestion_freq']['value']:.1f}{k['congestion_freq']['unit']}",
                                help=k['congestion_freq']['help'],
                            )
                            st.caption(k['congestion_freq'].get('extra', ''))
                            st.markdown(render_badge(k['congestion_freq']['score']), unsafe_allow_html=True)

                        with c3:
                            st.metric(
                                "⏱️ Average Travel Time",
                                f"{k['avg_tt']['value']:.1f} {k['avg_tt']['unit']}",
                                help=k['avg_tt']['help'],
                            )
                            st.markdown(render_badge(k['avg_tt']['score']), unsafe_allow_html=True)

                        with c4:
                            st.metric(
                                "📈 Planning Time (95th)",
                                f"{k['planning_time']['value']:.1f} {k['planning_time']['unit']}",
                                help=k['planning_time']['help'],
                            )
                            st.markdown(render_badge(k['planning_time']['score']), unsafe_allow_html=True)

                        with c5:
                            st.metric(
                                "🧭 Buffer Index",
                                f"{k['buffer_index']['value']:.1f}{k['buffer_index']['unit']}",
                                help=k['buffer_index']['help'],
                            )
                            st.markdown(render_badge(k['buffer_index']['score']), unsafe_allow_html=True)

                    # Optional trends
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

                    # Optional: cycle length section (uncomment if desired)
                    # render_cycle_length_section(raw_data)

        except Exception as e:
            st.error(f"❌ Error processing traffic data: {e}")
    else:
        st.warning("⚠️ Please select both start and end dates to proceed.")

# -------------------------
# TAB 2: Volume / Capacity (existing logic retained)
# -------------------------
with tab2:
    st.header("📊 Advanced Traffic Demand & Capacity Analysis")

    status_text = st.empty()
    status_text.info("Loading traffic demand data...")
    volume_df = get_volume_df()
    status_text.empty()

    if volume_df.empty:
        st.error("❌ Failed to load volume data.")
        st.stop()

    with st.sidebar:
        with st.expander("📊 Volume Analysis Controls", expanded=False):
            intersections = ["All Intersections"] + sorted(
                volume_df["intersection_name"].dropna().unique().tolist()
            )
            intersection = st.selectbox("🚦 Select Intersection", intersections, key="intersection_vol")

            min_date = volume_df["local_datetime"].dt.date.min()
            max_date = volume_df["local_datetime"].dt.date.max()

            st.markdown("#### 📅 Analysis Period")
            date_range_vol = date_range_preset_controls(min_date, max_date, key_prefix="vol")

            st.markdown("#### ⏰ Analysis Settings")
            granularity_vol = st.selectbox(
                "Data Aggregation", ["Hourly", "Daily", "Weekly", "Monthly"], index=0, key="granularity_vol"
            )

            direction_options = ["All Directions"] + sorted(volume_df["direction"].dropna().unique().tolist())
            direction_filter = st.selectbox("🔄 Direction Filter", direction_options, key="direction_filter_vol")

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

                    # Simple KPIs
                    if not base_df.empty:
                        raw = base_df[
                            (base_df["local_datetime"].dt.date >= date_range_vol[0])
                            & (base_df["local_datetime"].dt.date <= date_range_vol[1])
                        ].copy()
                        if not raw.empty:
                            raw["total_volume"] = pd.to_numeric(raw["total_volume"], errors="coerce")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                peak = float(np.nanmax(raw["total_volume"])) if raw["total_volume"].notna().any() else 0
                                st.metric("🔥 Peak Demand", f"{peak:,.0f} vph")
                            with col2:
                                avg = float(np.nanmean(raw["total_volume"])) if raw["total_volume"].notna().any() else 0
                                st.metric("📊 Average Demand", f"{avg:,.0f} vph")
                            with col3:
                                hv_hours = int((raw["total_volume"] > HIGH_VOLUME_THRESHOLD_VPH).sum())
                                total_hours = int(raw["total_volume"].count())
                                hv_pct = (hv_hours / total_hours * 100) if total_hours > 0 else 0
                                st.metric("⚠️ High Volume Time", f"{hv_pct:.1f}%")

                    chart1, chart2, chart3 = volume_charts(
                        filtered_volume_data, THEORETICAL_LINK_CAPACITY_VPH, HIGH_VOLUME_THRESHOLD_VPH
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

        except Exception as e:
            st.error(f"❌ Error processing volume data: {e}")
    else:
        st.warning("⚠️ Please select both start and end dates to proceed.")

# =========================
# FOOTER (adaptive colors)
# =========================
FOOTER = """
<style>
  .footer-title { color:#2980b9; margin:0 0 .4rem; font-weight:700; }
  .social-btn {
    width: 40px; height: 40px; display:grid; place-items:center; border-radius:50%;
    background:#ffffff; border:1px solid rgba(41,128,185,.25);
    box-shadow:0 2px 8px rgba(0,0,0,.08); text-decoration:none;
    transition: transform .15s ease, box-shadow .15s ease;
  }
  .social-btn:hover { transform: translateY(-1px); box-shadow:0 4px 14px rgba(0,0,0,.12); }
  .website-pill {
    height:40px; display:inline-flex; align-items:center; gap:8px; padding:0 12px;
    border-radius:9999px; background:#ffffff; border:1px solid #2980b9; color:#2980b9;
    font-weight:700; text-decoration:none; box-shadow:0 2px 8px rgba(0,0,0,.08);
    transition: transform .15s ease, box-shadow .15s ease;
  }
  .website-pill:hover { transform: translateY(-1px); box-shadow:0 4px 14px rgba(0,0,0,.12); }
</style>

<div class="footer-card" style="text-align:center; padding: 1.25rem;
    background: linear-gradient(135deg, rgba(79,172,254,0.1), rgba(0,242,254,0.05));
    border-radius: 15px; margin-top: 1rem; border: 1px solid rgba(79,172,254,0.2);
    font-family: system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif;">

  <h4 class="footer-title">🛣️ Active Transportation & Operations Management Dashboard</h4>
  <p class="footer-sub" style="margin:.1rem 0 0; font-size:1.0rem; color:#0f2f52;">
    Powered by Advanced Machine Learning • Real-time Traffic Intelligence • Intelligent Transportation Solutions (ITS)
  </p>

  <div style="display:flex; justify-content:center; align-items:center; gap:14px; margin:12px 0 8px;">
    <a class="social-btn" href="https://www.instagram.com/advantec98/" target="_blank" rel="noopener noreferrer" aria-label="Instagram"><span style="font:700 13px/1 system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial; color:#444;">IG</span></a>
    <a class="social-btn" href="https://www.linkedin.com/company/advantec-consulting-engineers-inc./posts/?feedView=all" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 448 512" aria-hidden="true"><path fill="#0A66C2" d="M100.28 448H7.4V148.9h92.88zM53.79 108.1C24.09 108.1 0 83.5 0 53.8 0 24.1 24.1 0 53.79 0s53.8 24.1 53.8 53.8c0 29.7-24.1 54.3-53.8 54.3zM447.9 448h-92.68V302.4c0-34.7-.7-79.3-48.3-79.3-48.3 0-55.7 37.7-55.7 76.6V448h-92.7V148.9h89V185h1.3c12.4-23.6 42.7-48.3 87.8-48.3 93.9 0 111.2 61.8 111.2 142.3V448z"/></svg>
    </a>
    <a class="social-btn" href="https://www.facebook.com/advantecconsultingUSA" target="_blank" rel="noopener noreferrer" aria-label="Facebook">
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 320 512" aria-hidden="true"><path fill="#1877F2" d="M279.14 288l14.22-92.66h-88.91v-60.13c0-25.35 12.42-50.06 52.24-50.06h40.42V6.26S263.61 0 225.36 0c-73.22 0-121 44.38-121 124.72v70.62H22.89V288h81.47v224h100.2V288z"/></svg>
    </a>
    <a class="website-pill" href="https://advantec-usa.com/" target="_blank" rel="noopener noreferrer" aria-label="ADVANTEC Website"><span style="font-size:18px; line-height:1;">🌐</span><span>Website</span></a>
  </div>

  <p class="footer-copy" style="margin:.2rem 0 0; font-size:.9rem; color:#0f2f52;">
    © 2025 ADVANTEC Consulting Engineers, Inc. — "Because We Care"
  </p>
</div>

<script>
(function() {
  function updateFooterColors() {
    const body = document.body;
    const computed = getComputedStyle(body);
    const bgColor = computed.backgroundColor || computed.getPropertyValue('--background-color') || '#ffffff';

    let r=255,g=255,b=255;
    const m = bgColor.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
    if (m) { r = +m[1]; g = +m[2]; b = +m[3]; }
    const luminance = (0.299*r + 0.587*g + 0.114*b)/255;
    const isDark = luminance < 0.5;

    const sub = document.querySelector('.footer-sub');
    const copy = document.querySelector('.footer-copy');
    const title = document.querySelector('.footer-title');
    if (sub && copy) {
      if (isDark) { sub.style.color = '#ffffff'; copy.style.color = '#ffffff'; if (title) title.style.color = '#7ec3ff'; }
      else { sub.style.color = '#0f2f52'; copy.style.color = '#0f2f52'; if (title) title.style.color = '#2980b9'; }
    }
  }
  updateFooterColors();
  new MutationObserver(updateFooterColors).observe(document.documentElement, {attributes:true, attributeFilter:['data-theme','class']});
  new MutationObserver(updateFooterColors).observe(document.body, {attributes:true, attributeFilter:['data-theme','class','style']});
  setInterval(updateFooterColors, 1000);
})();
</script>
"""
st.markdown(FOOTER, unsafe_allow_html=True)