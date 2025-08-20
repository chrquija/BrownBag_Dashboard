# Python
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import streamlit.components.v1 as components

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
    compute_perf_kpis_interpretable,
    render_badge,
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

# Canonical bottom → top node order (ensure labels match your dataset exactly)
DESIRED_NODE_ORDER_BOTTOM_UP = [
    "Avenue 52",
    "Calle Tampico",
    "Village Shopping Ctr",
    "Avenue 50",
    "Sagebrush Ave",
    "Eisenhower Dr",
    "Avenue 48",
    "Avenue 47",
    "Point Happy Simon",
    "Hwy 111",
]

# Build ordered node list from segment_name like "A → B"
def _build_node_order(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "segment_name" not in df.columns:
        return []
    segs = df["segment_name"].dropna().tolist()
    order: list[str] = []
    for s in segs:
        parts = [p.strip() for p in s.split("→")]
        if len(parts) != 2:
            continue
        a, b = parts[0], parts[1]
        if not order:
            order.append(a)
            order.append(b)
        else:
            if order[-1] == a:
                order.append(b)
            elif a not in order and b not in order:
                order.append(a)
                order.append(b)
    # de-duplicate preserving order
    seen, out = set(), []
    for n in order:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out

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
    <p>Leverages <strong>millions of data points</strong> trained on advanced Machine Learning algorithms to optimize traffic flow, reduce travel time, minimize fuel consumption, and decrease greenhouse gas emissions across the transportation network.</p>
    <p><strong>Key Capabilities:</strong> Real-time anomaly detection • Intelligent cycle length optimization • Predictive traffic modeling • Performance analytics</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 1.1rem; border-radius: 15px;
    margin: 1rem 0; text-align: center; box-shadow: 0 6px 20px rgba(52, 152, 219, 0.25);">
    <h3 style="margin:0; font-weight:600;">🔍 Research Question</h3>
    <p style="margin: 0.45rem 0 0; font-size: 1.0rem;">What are the main bottlenecks on Washington St that most increase travel times?</p>
</div>
""", unsafe_allow_html=True)

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["1️⃣ ITERIS CLEARGUIDE", "2️⃣ KINETIC MOBILITY"])

# -------------------------
# TAB 1: Performance / Travel Time
# -------------------------
with tab1:
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

        # Sidebar logos + Controls
        with st.sidebar:
            st.image("Logos/ACE-logo-HiRes.jpg", width=210)
            st.image("Logos/CV Sync__.jpg", width=205)

            with st.expander("TAB 1️⃣ CONTROLS", expanded=False):
                st.caption("Variables: Speed, Delay, and Travel Time")

                # O-D mode (origin → destination)
                od_mode = st.checkbox(
                    "Origin - Destination Mode",
                    value=True,
                    help="Compute KPIs using summed per-hour O-D trip times along the selected path.",
                )
                origin, destination = None, None
                if od_mode:
                    node_list_raw = _build_node_order(corridor_df)
                    known_in_data = [n for n in DESIRED_NODE_ORDER_BOTTOM_UP if n in node_list_raw]
                    extras = [n for n in node_list_raw if n not in DESIRED_NODE_ORDER_BOTTOM_UP]
                    node_list = known_in_data + extras

                    if len(node_list) >= 2:
                        cA, cB = st.columns(2)
                        with cA:
                            origin = st.selectbox("Origin", node_list, index=0, key="od_origin")
                        with cB:
                            destination = st.selectbox("Destination", node_list, index=len(node_list) - 1, key="od_destination")
                    else:
                        st.info("Not enough nodes found to build O-D options.")

                # Analysis Period
                min_date = corridor_df["local_datetime"].dt.date.min()
                max_date = corridor_df["local_datetime"].dt.date.max()
                st.markdown("#### 📅 Analysis Period")
                date_range = date_range_preset_controls(min_date, max_date, key_prefix="perf")

                # Analysis Settings
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

        # Main tab content
        if len(date_range) == 2:
            try:
                base_df = corridor_df.copy()
                if base_df.empty:
                    st.warning("⚠️ No data for the selected segment.")
                else:
                    # O-D subset first
                    working_df = base_df.copy()
                    route_label = "All Segments"
                    if od_mode and origin and destination:
                        node_order = _build_node_order(base_df)
                        if origin in node_order and destination in node_order:
                            i0, i1 = node_order.index(origin), node_order.index(destination)
                            if i0 < i1:
                                # Forward path (bottom → top) → use consecutive segments and keep NB only
                                path_segments = [f"{node_order[i]} → {node_order[i + 1]}" for i in range(i0, i1)]
                                seg_df = base_df[base_df["segment_name"].isin(path_segments)].copy()
                                if not seg_df.empty:
                                    # Direction filter: normalize and keep NB rows only
                                    if "direction" in seg_df.columns:
                                        dir_norm = (
                                            seg_df["direction"].astype(str).str.strip().str.lower()
                                            .replace({"northbound": "nb", "southbound": "sb"})
                                        )
                                        seg_df = seg_df.loc[dir_norm == "nb"]
                                    working_df = seg_df.copy()
                                    route_label = f"{origin} → {destination}"
                            else:
                                # Reverse path (top → bottom) → build reversed segments and keep SB only
                                path_segments = [f"{node_order[i - 1]} → {node_order[i]}" for i in range(i1, i0, -1)]
                                seg_df = base_df[base_df["segment_name"].isin(path_segments)].copy()
                                if not seg_df.empty:
                                    if "direction" in seg_df.columns:
                                        dir_norm = (
                                            seg_df["direction"].astype(str).str.strip().str.lower()
                                            .replace({"northbound": "nb", "southbound": "sb"})
                                        )
                                        seg_df = seg_df.loc[dir_norm == "sb"]
                                    working_df = seg_df.copy()
                                    route_label = f"{origin} → {destination}"
                                else:
                                    st.info(
                                        "Selected O-D is opposite to the dataset direction. Add reverse-direction data to analyze that path.")

                    # Filter + aggregate once for charts/tables at requested granularity
                    filtered_data = process_traffic_data(
                        working_df,
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

                        # Big banner title
                        st.markdown(
                            f"""
                        <div style="
                            background: linear-gradient(135deg, #2b77e5 0%, #19c3e6 100%);
                            border-radius:16px; padding:18px 20px; color:#fff; margin:8px 0 14px;
                            box-shadow:0 10px 26px rgba(25,115,210,.25); text-align:left;
                            font-family: inherit;">
                          <div style="display:flex; align-items:center; gap:10px;">
                            <div style="width:36px;height:36px;border-radius:10px;background:rgba(255,255,255,.18);
                                        display:flex;align-items:center;justify-content:center;
                                        box-shadow:inset 0 0 0 1px rgba(255,255,255,.15);">📊</div>
                            <div style="font-size:1.9rem;font-weight:800;letter-spacing:.2px;">
                              Travel Time Analysis: {route_label}
                            </div>
                          </div>
                          <div style="margin-top:10px;display:flex;flex-direction:column;gap:6px;">
                            <div>📅 {date_range[0].strftime('%b %d, %Y')} to {date_range[1].strftime('%b %d, %Y')} ({data_span} days) • {granularity} Aggregation{time_context}</div>
                            <div>✅ Analyzing {total_records:,} data points across the selected period</div>
                          </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                        # Build per-hour O-D series (sum across segments per hour) for KPIs
                        od_hourly = process_traffic_data(
                            working_df,
                            date_range,
                            "Hourly",  # force hourly to avoid averaging averages
                            time_filter,
                            start_hour,
                            end_hour,
                        )
                        if not od_hourly.empty:
                            od_series = (
                                od_hourly.groupby("local_datetime", as_index=False)
                                .agg({"average_traveltime": "sum", "average_delay": "sum"})
                            )
                            raw_data = od_series.copy()
                        else:
                            # Fallback to filtered_data if hourly O-D can't be built
                            od_series = pd.DataFrame()  # ensure variable exists
                            raw_data = filtered_data.copy()

                        if raw_data.empty:
                            st.info("No data in this window.")
                        else:
                            for col in ["average_delay", "average_traveltime", "average_speed"]:
                                if col in raw_data:
                                    raw_data[col] = pd.to_numeric(raw_data[col], errors="coerce")

                            st.subheader("🚦 Corridor Performance Metrics")
                            k = compute_perf_kpis_interpretable(raw_data, HIGH_DELAY_SEC)

                            # Compute Buffer Time in minutes from existing KPIs
                            buffer_minutes = max(0.0, k["planning_time"]["value"] - k["avg_tt"]["value"])
                            buffer_help = (
                                "Extra minutes to leave earlier so you arrive on time 95% of the time.\n"
                                "Formula: Planning Time (95th) − Average Travel Time."
                            )

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
                                    "📈 Planning Time (95t Percentile)",
                                    f"{k['planning_time']['value']:.1f} {k['planning_time']['unit']}",
                                    help=k['planning_time']['help'],
                                )
                                st.markdown(render_badge(k['planning_time']['score']), unsafe_allow_html=True)
                            with c5:
                                st.metric(
                                    "🧭 Buffer Index",
                                    f"{buffer_minutes:.1f} min",
                                    help=buffer_help,
                                )
                                # Reuse existing buffer-index score badge until backend returns a minutes-based score
                                st.markdown(render_badge(k['buffer_index']['score']), unsafe_allow_html=True)

                        if len(filtered_data) > 1:
                            st.subheader("📈 Performance Trends")
                            v1, v2 = st.columns(2)

                            # Use O-D series for trends if available; otherwise fall back
                            trends_df = od_series if not od_series.empty else filtered_data

                            # If user selected Daily/Weekly/Monthly, aggregate the O-D trends to match the selection
                            if not od_series.empty and granularity in ("Daily", "Weekly", "Monthly"):
                                tmp = od_series.copy()
                                tmp["local_datetime"] = pd.to_datetime(tmp["local_datetime"])
                                if granularity == "Daily":
                                    tmp["date_group"] = tmp["local_datetime"].dt.date
                                    trends_df = (
                                        tmp.groupby("date_group", as_index=False)
                                        .agg({"average_traveltime": "mean", "average_delay": "mean"})
                                        .rename(columns={"date_group": "local_datetime"})
                                    )
                                    trends_df["local_datetime"] = pd.to_datetime(trends_df["local_datetime"])
                                elif granularity == "Weekly":
                                    tmp["week_group"] = tmp["local_datetime"].dt.to_period("W").dt.start_time
                                    trends_df = (
                                        tmp.groupby("week_group", as_index=False)
                                        .agg({"average_traveltime": "mean", "average_delay": "mean"})
                                        .rename(columns={"week_group": "local_datetime"})
                                    )
                                elif granularity == "Monthly":
                                    tmp["month_group"] = tmp["local_datetime"].dt.to_period("M").dt.start_time
                                    trends_df = (
                                        tmp.groupby("month_group", as_index=False)
                                        .agg({"average_traveltime": "mean", "average_delay": "mean"})
                                        .rename(columns={"month_group": "local_datetime"})
                                    )

                            with v1:
                                dc = performance_chart(trends_df, "delay")
                                if dc:
                                    st.plotly_chart(dc, use_container_width=True)
                            with v2:
                                tc = performance_chart(trends_df, "travel")
                                if tc:
                                    st.plotly_chart(tc, use_container_width=True)

                            # Optional: Corridor O-D summary table (Hourly)
                            if not od_series.empty:
                                st.subheader("🔍Which Dates/Times have the highest Travel Time and Delay?")
                                st.dataframe(
                                    od_series.rename(
                                        columns={
                                            "local_datetime": "Timestamp",
                                            "average_traveltime": "O-D Travel Time (min)",
                                            "average_delay": "O-D Delay (s)",
                                        }
                                    ),
                                    use_container_width=True,
                                )

                        if not raw_data.empty:
                            worst_delay = (
                                float(np.nanmax(raw_data["average_delay"]))
                                if "average_delay" in raw_data and raw_data["average_delay"].notna().any()
                                else 0.0
                            )

                        if not raw_data.empty:
                            worst_delay = (
                                float(np.nanmax(raw_data["average_delay"]))
                                if "average_delay" in raw_data and raw_data["average_delay"].notna().any()
                                else 0.0
                            )
                            avg_tt = (
                                float(np.nanmean(raw_data["average_traveltime"]))
                                if "average_traveltime" in raw_data and raw_data["average_traveltime"].notna().any()
                                else 0.0
                            )
                            worst_tt = (
                                float(np.nanmax(raw_data["average_traveltime"]))
                                if "average_traveltime" in raw_data and raw_data["average_traveltime"].notna().any()
                                else 0.0
                            )
                            tt_delta = ((worst_tt - avg_tt) / avg_tt * 100) if avg_tt > 0 else 0
                            if avg_tt > 0 and "average_traveltime" in raw_data:
                                cv_tt = float(np.nanstd(raw_data["average_traveltime"]) / avg_tt) * 100
                            else:
                                cv_tt = 0.0
                            reliability = max(0, 100 - cv_tt)
                            high_delay_pct = (
                                (raw_data["average_delay"] > HIGH_DELAY_SEC).mean() * 100
                                if "average_delay" in raw_data and raw_data["average_delay"].notna().any()
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
                        if not raw_data.empty and "segment_name" in working_df.columns:
                            try:
                                g = working_df[
                                    (working_df["local_datetime"].dt.date >= date_range[0])
                                    & (working_df["local_datetime"].dt.date <= date_range[1])
                                ].groupby(["segment_name", "direction"]).agg(
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
            with st.expander("TAB 2️⃣ CONTROLS", expanded=False):
                st.caption("Variables: Vehicle Volume")
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
                    "Data Aggregation",
                    ["Hourly", "Daily", "Weekly", "Monthly"],
                    index=0,
                    key="granularity_vol",
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
                        total_obs = len(filtered_volume_data)

                        st.markdown(
                            f"""
                        <div style="
                            background: linear-gradient(135deg, #2b77e5 0%, #19c3e6 100%);
                            border-radius:16px; padding:18px 20px; color:#fff; margin:8px 0 14px;
                            box-shadow:0 10px 26px rgba(25,115,210,.25); text-align:left;
                            font-family: system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
                          <div style="display:flex; align-items:center; gap:10px;">
                            <div style="width:36px;height:36px;border-radius:10px;background:rgba(255,255,255,.18);
                                        display:flex;align-items:center;justify-content:center;
                                        box-shadow:inset 0 0 0 1px rgba(255,255,255,.15);">📊</div>
                            <div style="font-size:1.9rem;font-weight:800;letter-spacing:.2px;">
                              Volume Analysis: {intersection}
                            </div>
                          </div>
                          <div style="margin-top:10px;display:flex;flex-direction:column;gap:6px;">
                            <div>📅 {date_range_vol[0].strftime('%b %d, %Y')} to {date_range_vol[1].strftime('%b %d, %Y')} ({span} days) • {granularity_vol} Aggregation</div>
                            <div>✅ {total_obs:,} observations • Direction: {direction_filter}</div>
                          </div>
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
                                st.metric("🔥 Peak Demand", f"{peak:,.0f} vehicles", delta=f"95th: {p95:,.0f}")
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
                                    f'<span class="performance-badge {badge}' + f'">{avg_util:.0f}% Avg Util</span>',
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

                        # Cycle Length Recommendations section
                        render_cycle_length_section(raw)

            except Exception as e:
                st.error(f"❌ Error processing volume data: {e}")
                st.info("Please check your data sources and try again.")
        else:
            st.warning("⚠️ Please select both start and end dates to proceed with the volume analysis.")

# =========================
# FOOTER
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
    <a class="social-btn" href="https://www.instagram.com/advantec98/" target="_blank" rel="noopener noreferrer" aria-label="Instagram">
      <span style="font:700 13px/1 system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial; color:#444;">IG</span>
    </a>
    <a class="social-btn" href="https://www.linkedin.com/company/advantec-consulting-engineers-inc./posts/?feedView=all"
       target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 448 512" aria-hidden="true"><path fill="#0A66C2" d="M100.28 448H7.4V148.9h92.88zM53.79 108.1C24.09 108.1 0 83.5 0 53.8 0 24.1 24.1 0 53.79 0s53.8 24.1 53.8 53.8c0 29.7-24.1 54.3-53.8 54.3zM447.9 448h-92.68V302.4c0-34.7-.7-79.3-48.3-79.3-48.3 0-55.7 37.7-55.7 76.6V448h-92.7V148.9h89V185h1.3c12.4-23.6 42.7-48.3 87.8-48.3 93.9 0 111.2 61.8 111.2 142.3V448z"/></svg>
    </a>
    <a class="social-btn" href="https://www.facebook.com/advantecconsultingUSA" target="_blank" rel="noopener noreferrer" aria-label="Facebook">
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 320 512" aria-hidden="true"><path fill="#1877F2" d="M279.14 288l14.22-92.66h-88.91v-60.13c0-25.35 12.42-50.06 52.24-50.06h40.42V6.26S263.61 0 225.36 0c-73.22 0-121 44.38-121 124.72v70.62H22.89V288h81.47v224h100.2V288z"/></svg>
    </a>
    <a class="website-pill" href="https://advantec-usa.com/" target="_blank" rel="noopener noreferrer" aria-label="ADVANTEC Website">
      <span style="font-size:18px; line-height:1;">🌐</span>
      <span>Website</span>
    </a>
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
    const bgColor = computed.backgroundColor || getComputedStyle(document.documentElement).getPropertyValue('--background-color') || '#ffffff';

    let r=255,g=255,b=255;
    if (bgColor.startsWith('rgb')) {
      const m = bgColor.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
      if (m) { r = parseInt(m[1]); g = parseInt(m[2]); b = parseInt(m[3]); }
    }
    const luminance = (0.299*r + 0.587*g + 0.114*b) / 255;
    const isDark = luminance < 0.5;

    const subtitle = document.querySelector('.footer-sub');
    const copyright = document.querySelector('.footer-copy');
    const title = document.querySelector('.footer-title');

    if (subtitle && copyright) {
      if (isDark) {
        subtitle.style.color = '#ffffff';
        copyright.style.color = '#ffffff';
        if (title) title.style.color = '#7ec3ff';
      } else {
        subtitle.style.color = '#0f2f52';
        copyright.style.color = '#0f2f52';
        if (title) title.style.color = '#2980b9';
      }
    }
  }
  updateFooterColors();
  const observer = new MutationObserver(updateFooterColors);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'class'] });
  observer.observe(document.body, { attributes: true, attributeFilter: ['data-theme', 'class', 'style'] });
  setInterval(updateFooterColors, 1000);
})();
</script>
"""

st.markdown(FOOTER, unsafe_allow_html=True)