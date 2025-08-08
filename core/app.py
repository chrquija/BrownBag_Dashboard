import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# === External project functions ===
from sidebar_functions import process_traffic_data, load_traffic_data, load_volume_data

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Active Transportation & Operations Management Dashboard",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Hybrid Threshold Defaults (fallbacks)
# =========================
DEFAULT_CAPACITY_VPH = 1600      # fallback if practical capacity can't be computed
DEFAULT_HIGH_UTIL = 0.70         # high utilization band if cap is unknown
DEFAULT_CRITICAL_UTIL = 0.90     # critical utilization band if cap is unknown

# HCM-aligned delay (s/veh)
HIGH_DELAY_SEC = 55
CRITICAL_DELAY_SEC = 80

# Travel-time & speed bands (relative to free-flow baselines)
TTI_MODERATE = 1.15
TTI_HIGH = 1.30
TTI_CRITICAL = 1.50

SPEED_RATIO_WARN = 0.85
SPEED_RATIO_HIGH = 0.70
SPEED_RATIO_CRITICAL = 0.60

# =========================
# CSS (stakeholder polish)
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
    .insight-box {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.15) 0%, rgba(0, 242, 254, 0.15) 100%);
        border-left: 5px solid #4facfe; border-radius: 12px; padding: 1.1rem 1.3rem; margin: 1.1rem 0;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.1);
    }
    .legend-box {
        background: linear-gradient(135deg, rgba(46, 204, 113, .08), rgba(52, 152, 219, .08));
        border-left: 4px solid #2ecc71; border-radius: 10px; padding: .85rem 1rem; margin: .75rem 0 0;
        font-size: .92rem;
    }
    .chip {
        display:inline-block; padding:.15rem .55rem; border-radius:999px; font-size:.8rem; margin:.1rem .25rem;
        border:1px solid rgba(0,0,0,.06); background:rgba(255,255,255,.65);
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
# Helpers / Utilities
# =========================
def _safe_to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def get_corridor_df() -> pd.DataFrame:
    df = load_traffic_data()
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = _safe_to_datetime(df.copy(), "local_datetime")
    return df

@st.cache_data(show_spinner=False)
def get_volume_df() -> pd.DataFrame:
    df = load_volume_data()
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = _safe_to_datetime(df.copy(), "local_datetime")
    return df

def get_performance_rating(score: float):
    if score > 80: return "🟢 Excellent", "badge-excellent"
    if score > 60: return "🔵 Good", "badge-good"
    if score > 40: return "🟡 Fair", "badge-fair"
    if score > 20: return "🟠 Poor", "badge-poor"
    return "🔴 Critical", "badge-critical"

# --- Auto-calibration using local data ---
def compute_practical_capacity(volume_df: pd.DataFrame, window_days: int = 90):
    if volume_df.empty:
        return pd.DataFrame(columns=["intersection_name", "direction", "practical_capacity_vph"])
    df = volume_df.copy()
    df["local_datetime"] = pd.to_datetime(df["local_datetime"], errors="coerce")
    df = df.dropna(subset=["intersection_name", "direction", "total_volume"])
    cutoff = df["local_datetime"].max() - pd.Timedelta(days=window_days)
    df = df[df["local_datetime"] >= cutoff]
    grp = df.groupby(["intersection_name", "direction"])["total_volume"]
    cap = grp.quantile(0.95).reset_index(name="practical_capacity_vph")
    cap["practical_capacity_vph"] = cap["practical_capacity_vph"].clip(lower=600)
    return cap

def compute_freeflow_baselines(corridor_df: pd.DataFrame):
    if corridor_df.empty:
        return pd.DataFrame(columns=["segment_name", "direction", "ff_traveltime_min", "ff_speed_mph"])
    df = corridor_df.copy()
    df["local_datetime"] = pd.to_datetime(df["local_datetime"], errors="coerce")
    df["hour"] = df["local_datetime"].dt.hour
    offpeak = df[df["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5])]

    def _agg(g):
        tt = pd.to_numeric(g["average_traveltime"], errors="coerce").dropna()
        sp = pd.to_numeric(g["average_speed"], errors="coerce").dropna()
        return pd.Series({
            "ff_traveltime_min": (np.percentile(tt, 10) if len(tt) else np.nan),
            "ff_speed_mph": (np.percentile(sp, 90) if len(sp) else np.nan),
        })

    base = (offpeak
            .groupby(["segment_name", "direction"], dropna=True)
            .apply(_agg)
            .reset_index())
    return base

def get_capacity_for_selection(cap_df: pd.DataFrame,
                               intersection: str,
                               direction: str):
    if cap_df.empty:
        cap = DEFAULT_CAPACITY_VPH
        return {"cap_vph": cap, "high_vph": DEFAULT_HIGH_UTIL*cap, "critical_vph": DEFAULT_CRITICAL_UTIL*cap}

    if intersection != "All Intersections" and direction != "All Directions":
        r = cap_df[(cap_df["intersection_name"] == intersection) & (cap_df["direction"] == direction)]
        if not r.empty:
            cap = float(r["practical_capacity_vph"].iloc[0])
            return {"cap_vph": cap, "high_vph": 0.70*cap, "critical_vph": 0.90*cap}

    if intersection != "All Intersections":
        subset = cap_df[cap_df["intersection_name"] == intersection]["practical_capacity_vph"]
    elif direction != "All Directions":
        subset = cap_df[cap_df["direction"] == direction]["practical_capacity_vph"]
    else:
        subset = cap_df["practical_capacity_vph"]

    cap = float(np.nanpercentile(subset, 95)) if len(subset) else DEFAULT_CAPACITY_VPH
    return {"cap_vph": cap, "high_vph": 0.70*cap, "critical_vph": 0.90*cap}

def get_ff_for_segment(ff_df: pd.DataFrame, segment: str, direction: str):
    if ff_df.empty:
        return np.nan, np.nan
    r = ff_df[(ff_df["segment_name"] == segment) & (ff_df["direction"] == direction)]
    if not r.empty:
        return float(r["ff_traveltime_min"].iloc[0]), float(r["ff_speed_mph"].iloc[0])
    return np.nan, np.nan

def performance_chart(data: pd.DataFrame, metric_type: str = "delay"):
    if data.empty: return None
    metric_type = metric_type.lower().strip()
    if metric_type == "delay":
        y_col, title, color = "average_delay", "Traffic Delay Analysis", "#e74c3c"
    else:
        y_col, title, color = "average_traveltime", "Travel Time Analysis", "#3498db"

    dd = data.dropna(subset=["local_datetime", y_col]).sort_values("local_datetime")

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Time Series Analysis", "Distribution Analysis"),
                        vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=dd["local_datetime"], y=dd[y_col],
                             mode="lines+markers", name=f"{metric_type.title()} Trend",
                             line=dict(color=color, width=2), marker=dict(size=4)), row=1, col=1)
    fig.add_trace(go.Histogram(x=dd[y_col], nbinsx=30, name=f"{metric_type.title()} Distribution",
                               marker_color=color, opacity=0.75), row=2, col=1)
    fig.update_layout(height=600, title=title, showlegend=True, template="plotly_white",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig

def volume_charts(data: pd.DataFrame, cap_hint: float = None):
    if data.empty: return None, None, None
    dd = data.dropna(subset=["local_datetime", "total_volume", "intersection_name"]).copy()
    dd.sort_values("local_datetime", inplace=True)

    fig1 = px.line(dd, x="local_datetime", y="total_volume", color="intersection_name",
                   title="📈 Traffic Volume Trends by Intersection",
                   labels={"total_volume": "Volume (vehicles/hour)", "local_datetime": "Date/Time"},
                   template="plotly_white")
    fig1.update_layout(height=500, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    if cap_hint:
        fig1.add_hline(y=cap_hint, line_dash="dash", line_color="red", annotation_text=f"Capacity ≈ {int(cap_hint):,} vph")

    fig2 = make_subplots(rows=2, cols=1,
                         subplot_titles=("Volume Distribution by Intersection", "Hourly Avg Volume Heatmap"),
                         vertical_spacing=0.12)
    for name, g in dd.groupby("intersection_name", sort=False):
        fig2.add_trace(go.Box(y=g["total_volume"], name=name, boxpoints="outliers"), row=1, col=1)

    dd["hour"] = dd["local_datetime"].dt.hour
    hourly_avg = dd.groupby(["hour", "intersection_name"], as_index=False)["total_volume"].mean()
    hourly_pivot = hourly_avg.pivot(index="intersection_name", columns="hour", values="total_volume").sort_index()
    fig2.add_trace(go.Heatmap(z=hourly_pivot.values, x=hourly_pivot.columns, y=hourly_pivot.index,
                              colorscale="Blues", showscale=True, colorbar=dict(title="Avg Volume (vph)")),
                   row=2, col=1)
    fig2.update_layout(height=800, title="📊 Volume Distribution & Capacity Analysis", template="plotly_white",
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")

    hourly_volume = dd.groupby(["hour", "intersection_name"], as_index=False)["total_volume"].mean()
    fig3 = px.line(hourly_volume, x="hour", y="total_volume", color="intersection_name",
                   title="🕐 Average Hourly Volume Patterns",
                   labels={"total_volume": "Average Volume (vph)", "hour": "Hour of Day"},
                   template="plotly_white")
    if cap_hint:
        fig3.add_hline(y=cap_hint, line_dash="dash", line_color="red", annotation_text=f"Capacity ≈ {int(cap_hint):,} vph")
        fig3.add_hline(y=0.7*cap_hint, line_dash="dot", line_color="orange", annotation_text="High Util (~70% cap)")
    fig3.update_layout(height=500, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig1, fig2, fig3

def date_range_preset_controls(min_date: datetime.date, max_date: datetime.date, key_prefix: str):
    k_range = f"{key_prefix}_range"
    if k_range not in st.session_state:
        default_start = max(min_date, max_date - timedelta(days=30))
        st.session_state[k_range] = (default_start, max_date)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📅 Last 7 Days", key=f"{key_prefix}_7d"):
            st.session_state[k_range] = (max(min_date, max_date - timedelta(days=7)), max_date)
    with c2:
        if st.button("📅 Last 30 Days", key=f"{key_prefix}_30d"):
            st.session_state[k_range] = (max(min_date, max_date - timedelta(days=30)), max_date)
    with c3:
        if st.button("📅 Full Range", key=f"{key_prefix}_full"):
            st.session_state[k_range] = (min_date, max_date)

    custom = st.date_input("Custom Date Range",
                           value=st.session_state[k_range],
                           min_value=min_date, max_value=max_date,
                           key=f"{key_prefix}_custom")
    if custom != st.session_state[k_range]:
        st.session_state[k_range] = custom
    return st.session_state[k_range]

def prior_period(date_range):
    """Return previous period with same length directly before current window."""
    start, end = date_range
    delta = (end - start)
    prev_end = start - timedelta(days=1)
    prev_start = prev_end - delta
    return (prev_start, prev_end)

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["🚧 Performance & Delay Analysis", "📊 Traffic Demand & Capacity Analysis"])

# -------------------------
# TAB 1: Performance / Travel Time
# -------------------------
with tab1:
    st.header("🚧 Comprehensive Performance & Travel Time Analysis")

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
        time.sleep(0.4); progress_bar.empty(); status_text.empty()

        # ---- Sidebar / View Mode ----
        with st.sidebar:
            with st.expander("🎛️ View & Filters", expanded=False):
                view_mode = st.radio("View Mode", ["Executive", "Analyst"], horizontal=True, index=0,
                                     help="Executive shows summary & top actions. Analyst reveals full detail.")
            with st.expander("🚧 Performance Analysis Controls", expanded=False):
                seg_options = ["All Segments"] + sorted(corridor_df["segment_name"].dropna().unique().tolist())
                corridor = st.selectbox("🛣️ Select Corridor Segment", seg_options,
                                        help="Choose a specific segment or analyze all segments")
                min_date = corridor_df["local_datetime"].dt.date.min()
                max_date = corridor_df["local_datetime"].dt.date.max()
                st.markdown("#### 📅 Analysis Period")
                date_range = date_range_preset_controls(min_date, max_date, key_prefix="perf")
                st.markdown("#### ⏰ Analysis Settings")
                granularity = st.selectbox("Data Aggregation", ["Hourly", "Daily", "Weekly", "Monthly"],
                                           index=0, help="Higher aggregation smooths trends but may hide peaks")
                time_filter, start_hour, end_hour = None, None, None
                if granularity == "Hourly":
                    time_filter = st.selectbox(
                        "Time Period Focus",
                        ["All Hours", "Peak Hours (7–9 AM, 4–6 PM)", "AM Peak (7–9 AM)", "PM Peak (4–6 PM)", "Off-Peak", "Custom Range"]
                    )
                    if time_filter == "Custom Range":
                        c1, c2 = st.columns(2)
                        with c1: start_hour = st.number_input("Start Hour (0–23)", 0, 23, 7, step=1)
                        with c2: end_hour   = st.number_input("End Hour (1–24)", 1, 24, 18, step=1)

        if len(date_range) == 2:
            try:
                base_df = corridor_df.copy()
                if corridor != "All Segments":
                    base_df = base_df[base_df["segment_name"] == corridor]

                if base_df.empty:
                    st.warning("⚠️ No data for the selected segment.")
                else:
                    ff_df = compute_freeflow_baselines(base_df)

                    filtered_data = process_traffic_data(
                        base_df, date_range, granularity,
                        time_filter if granularity == "Hourly" else None,
                        start_hour, end_hour
                    )

                    if filtered_data.empty:
                        st.warning("⚠️ No data available for the selected filters.")
                    else:
                        total_records = len(filtered_data)
                        data_span = (date_range[1] - date_range[0]).days + 1
                        time_context = f" • {time_filter}" if (granularity == "Hourly" and time_filter) else ""

                        st.markdown(f"""
                        <div class="context-header">
                            <h2>📊 Performance Dashboard: {corridor}</h2>
                            <p>📅 {date_range[0].strftime('%b %d, %Y')} to {date_range[1].strftime('%b %d, %Y')}
                            ({data_span} days) • {granularity} Aggregation{time_context}</p>
                            <p>📈 Analyzing {total_records:,} data points across the selected period</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # ===== Executive Summary (topline KPIs + deltas) =====
                        # prepare raw windows for current & previous periods
                        raw_cur = base_df[(base_df["local_datetime"].dt.date >= date_range[0]) &
                                          (base_df["local_datetime"].dt.date <= date_range[1])].copy()
                        prev_range = prior_period(date_range)
                        raw_prev = base_df[(base_df["local_datetime"].dt.date >= prev_range[0]) &
                                           (base_df["local_datetime"].dt.date <= prev_range[1])].copy()

                        for col in ["average_delay", "average_traveltime", "average_speed"]:
                            if col in raw_cur: raw_cur[col] = pd.to_numeric(raw_cur[col], errors="coerce")
                            if col in raw_prev: raw_prev[col] = pd.to_numeric(raw_prev[col], errors="coerce")

                        # compute TTI/speed ratios
                        cur_tmp = raw_cur.merge(ff_df, on=["segment_name", "direction"], how="left")
                        prev_tmp = raw_prev.merge(ff_df, on=["segment_name", "direction"], how="left")
                        raw_cur["TTI"] = cur_tmp["average_traveltime"] / cur_tmp["ff_traveltime_min"]
                        raw_cur["SpeedRatio"] = cur_tmp["average_speed"] / cur_tmp["ff_speed_mph"]
                        raw_prev["TTI"] = prev_tmp["average_traveltime"] / prev_tmp["ff_traveltime_min"]
                        raw_prev["SpeedRatio"] = prev_tmp["average_speed"] / prev_tmp["ff_speed_mph"]

                        def _med(series):
                            s = pd.to_numeric(series, errors="coerce").dropna()
                            return float(np.nanmedian(s)) if len(s) else np.nan
                        def _mean(series):
                            s = pd.to_numeric(series, errors="coerce").dropna()
                            return float(np.nanmean(s)) if len(s) else np.nan

                        # KPIs current
                        kpi_delay95 = float(np.nanpercentile(raw_cur["average_delay"].dropna(), 95)) if raw_cur["average_delay"].notna().any() else np.nan
                        kpi_tti_med = _med(raw_cur["TTI"])
                        kpi_rel = 100 - (np.nanstd(raw_cur["average_traveltime"]) / _mean(raw_cur["average_traveltime"]) * 100 if _mean(raw_cur["average_traveltime"]) > 0 else 0)
                        kpi_sr_med = _med(raw_cur["SpeedRatio"])
                        kpi_high_delay_pct = (raw_cur["average_delay"] > HIGH_DELAY_SEC).mean() * 100 if raw_cur["average_delay"].notna().any() else 0

                        # KPIs previous
                        prev_delay95 = float(np.nanpercentile(raw_prev["average_delay"].dropna(), 95)) if raw_prev["average_delay"].notna().any() else np.nan
                        prev_tti_med = _med(raw_prev["TTI"])
                        prev_rel = 100 - (np.nanstd(raw_prev["average_traveltime"]) / _mean(raw_prev["average_traveltime"]) * 100 if _mean(raw_prev["average_traveltime"]) > 0 else 0)
                        prev_sr_med = _med(raw_prev["SpeedRatio"])
                        prev_high_delay_pct = (raw_prev["average_delay"] > HIGH_DELAY_SEC).mean() * 100 if raw_prev["average_delay"].notna().any() else 0

                        # Executive strip
                        if view_mode == "Executive":
                            c1, c2, c3, c4 = st.columns(4)
                            with c1:
                                st.metric("95th % Delay (s)", f"{kpi_delay95:.0f}" if np.isfinite(kpi_delay95) else "—",
                                          delta=None if not np.isfinite(prev_delay95) else f"{kpi_delay95 - prev_delay95:+.0f} vs prev")
                            with c2:
                                st.metric("Median TTI", f"{kpi_tti_med:.2f}" if np.isfinite(kpi_tti_med) else "—",
                                          delta=None if not np.isfinite(prev_tti_med) else f"{kpi_tti_med - prev_tti_med:+.2f}")
                            with c3:
                                st.metric("Reliability (%)", f"{kpi_rel:.0f}" if np.isfinite(kpi_rel) else "—",
                                          delta=None if not np.isfinite(prev_rel) else f"{kpi_rel - prev_rel:+.0f}")
                            with c4:
                                st.metric("% Hours > 55s", f"{kpi_high_delay_pct:.1f}",
                                          delta=f"{kpi_high_delay_pct - prev_high_delay_pct:+.1f} pp")

                        # Analyst KPIs detail (always visible; Execs can glance/ignore)
                        if not raw_cur.empty:
                            col1, col2, col3, col4, col5 = st.columns(5)

                            with col1:
                                worst_delay = float(np.nanmax(raw_cur["average_delay"])) if raw_cur["average_delay"].notna().any() else 0.0
                                p95_delay = kpi_delay95 if np.isfinite(kpi_delay95) else 0.0
                                rating, badge = get_performance_rating(100 - min(worst_delay / 2, 100))
                                st.metric("🚨 Peak Delay", f"{worst_delay:.0f}s", delta=f"95th: {p95_delay:.0f}s")
                                st.markdown(f'<span class="performance-badge {badge}">{rating}</span>', unsafe_allow_html=True)

                            with col2:
                                worst_tt = float(np.nanmax(raw_cur["average_traveltime"])) if raw_cur["average_traveltime"].notna().any() else 0.0
                                avg_tt = _mean(raw_cur["average_traveltime"])
                                tt_delta = ((worst_tt - avg_tt) / avg_tt * 100) if avg_tt and avg_tt > 0 else 0
                                impact_rating, badge = get_performance_rating(100 - min(max(tt_delta,0), 100))
                                st.metric("⏱️ Peak Travel Time", f"{worst_tt:.1f}min", delta=f"+{tt_delta:.0f}% vs avg")
                                st.markdown(f'<span class="performance-badge {badge}">{impact_rating}</span>', unsafe_allow_html=True)

                            with col3:
                                slowest = float(np.nanmin(raw_cur["average_speed"])) if raw_cur["average_speed"].notna().any() else 0.0
                                avg_speed = _mean(raw_cur["average_speed"])
                                speed_drop = ((avg_speed - slowest) / avg_speed * 100) if avg_speed and avg_speed > 0 else 0
                                st.metric("🐌 Minimum Speed", f"{slowest:.1f}mph", delta=f"-{speed_drop:.0f}% vs avg")
                                sr_med = kpi_sr_med
                                if np.isfinite(sr_med):
                                    if sr_med < SPEED_RATIO_CRITICAL: badge = "badge-critical"
                                    elif sr_med < SPEED_RATIO_HIGH: badge = "badge-poor"
                                    elif sr_med < SPEED_RATIO_WARN: badge = "badge-fair"
                                    else: badge = "badge-good"
                                    st.markdown(f'<span class="performance-badge {badge}">Speed/FF ≈ {sr_med:.2f}</span>', unsafe_allow_html=True)

                            with col4:
                                reliability = kpi_rel if np.isfinite(kpi_rel) else 0
                                rel_rating, badge = get_performance_rating(reliability)
                                st.metric("🎯 Reliability Index", f"{reliability:.0f}%", delta=None if not np.isfinite(prev_rel) else f"{reliability - prev_rel:+.0f} pp")
                                st.markdown(f'<span class="performance-badge {badge}">{rel_rating}</span>', unsafe_allow_html=True)

                            with col5:
                                high_delay_pct = kpi_high_delay_pct
                                hours = int((raw_cur["average_delay"] > HIGH_DELAY_SEC).sum()) if raw_cur["average_delay"].notna().any() else 0
                                freq_rating, badge = get_performance_rating(100 - high_delay_pct)
                                st.metric("⚠️ % Hours > 55s", f"{high_delay_pct:.1f}%", delta=f"{high_delay_pct - prev_high_delay_pct:+.1f} pp")
                                st.markdown(f'<span class="performance-badge {badge}">{freq_rating}</span>', unsafe_allow_html=True)

                            # TTI badge
                            if np.isfinite(kpi_tti_med):
                                if kpi_tti_med >= TTI_CRITICAL: badge = "badge-critical"
                                elif kpi_tti_med >= TTI_HIGH: badge = "badge-poor"
                                elif kpi_tti_med >= TTI_MODERATE: badge = "badge-fair"
                                else: badge = "badge-good"
                                st.markdown(f'<span class="performance-badge {badge}">TTI ≈ {kpi_tti_med:.2f}</span>', unsafe_allow_html=True)

                        # Legend for stakeholders
                        st.markdown(f"""
                        <div class="legend-box">
                            <b>How to read this:</b>
                            <span class="chip">TTI = Travel Time / Free-Flow Time</span>
                            <span class="chip">Speed/FF = Speed / Free-Flow Speed</span>
                            <span class="chip">Free-Flow = 10th % travel time & 90th % speed during 22:00–05:00</span><br/>
                            Delay bands reflect HCM: <b>High ≥ {HIGH_DELAY_SEC}s</b>, <b>Critical ≥ {CRITICAL_DELAY_SEC}s</b>. Reliability uses CV of travel time (higher = steadier).
                        </div>
                        """, unsafe_allow_html=True)

                        # Charts (hide in Executive mode if you want ultra-compact)
                        if view_mode == "Analyst" and len(filtered_data) > 1:
                            st.subheader("📈 Performance Trends")
                            v1, v2 = st.columns(2)
                            with v1:
                                dc = performance_chart(filtered_data, "delay")
                                if dc: st.plotly_chart(dc, use_container_width=True)
                            with v2:
                                tc = performance_chart(filtered_data, "travel")
                                if tc: st.plotly_chart(tc, use_container_width=True)

                        # Bottlenecks + executive takeaways
                        st.subheader("🚨 Comprehensive Bottleneck Analysis")
                        if not raw_cur.empty:
                            try:
                                g = raw_cur.groupby(["segment_name", "direction"]).agg(
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

                                score = (0.45*_norm(g["average_delay_max"]) +
                                         0.35*_norm(g["average_delay_mean"]) +
                                         0.20*_norm(g["average_traveltime_max"])) * 100
                                g["Bottleneck_Score"] = score.round(1)

                                bins = [-0.1, 20, 40, 60, 80, 200]
                                labels = ['🟢 Excellent', '🔵 Good', '🟡 Fair', '🟠 Poor', '🔴 Critical']
                                g["🎯 Performance Rating"] = pd.cut(g["Bottleneck_Score"], bins=bins, labels=labels)

                                final = g[[
                                    "segment_name","direction","🎯 Performance Rating","Bottleneck_Score",
                                    "average_delay_mean","average_delay_max",
                                    "average_traveltime_mean","average_traveltime_max",
                                    "average_speed_mean","average_speed_min","n"
                                ]].rename(columns={
                                    "segment_name": "Segment","direction": "Dir",
                                    "average_delay_mean": "Avg Delay (s)","average_delay_max": "Peak Delay (s)",
                                    "average_traveltime_mean": "Avg Time (min)","average_traveltime_max": "Peak Time (min)",
                                    "average_speed_mean": "Avg Speed (mph)","average_speed_min": "Min Speed (mph)",
                                    "n": "Obs"
                                }).sort_values("Bottleneck_Score", ascending=False)

                                # Executive summary bullets: Top 3 bottlenecks
                                top3 = final.head(3)
                                bullets = "<br/>".join(
                                    f"• <b>{r['Segment']}</b> {r['Dir']} — Impact {r['Bottleneck_Score']:.1f}, Peak Delay {r['Peak Delay (s)']:.0f}s"
                                    for _, r in top3.iterrows()
                                )
                                st.markdown(f"""
                                <div class="insight-box">
                                    <h4>💼 Executive Summary</h4>
                                    <p><strong>Top Bottlenecks (current period):</strong><br/>{bullets if len(top3)>0 else "No bottlenecks detected"}</p>
                                    <p><strong>Operational Focus:</strong> Prioritize signal optimization and incident response on the above segments during peak hours. Track TTI trend and %Hours&gt;55s to verify improvement.</p>
                                </div>
                                """, unsafe_allow_html=True)

                                st.dataframe(
                                    final.head(15),
                                    use_container_width=True,
                                    column_config={
                                        "Bottleneck_Score": st.column_config.NumberColumn(
                                            "🚨 Impact Score", help="Composite (0–100); higher ⇒ worse", format="%.1f"
                                        )
                                    }
                                )
                                st.download_button("⬇️ Download Bottleneck Table (CSV)",
                                                   data=final.to_csv(index=False).encode("utf-8"),
                                                   file_name="bottlenecks.csv", mime="text/csv")
                                st.download_button("⬇️ Download Filtered Performance (CSV)",
                                                   data=filtered_data.to_csv(index=False).encode("utf-8"),
                                                   file_name="performance_filtered.csv", mime="text/csv")
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
        time.sleep(0.4); progress_bar.empty(); status_text.empty()

        cap_df = compute_practical_capacity(volume_df, window_days=90)

        with st.sidebar:
            with st.expander("🎛️ View & Filters", expanded=False):
                view_mode_vol = st.radio("View Mode (Volume)", ["Executive", "Analyst"], horizontal=True, index=0)
            with st.expander("📊 Volume Analysis Controls", expanded=False):
                intersections = ["All Intersections"] + sorted(volume_df["intersection_name"].dropna().unique().tolist())
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
                        st.markdown(f"""
                        <div class="context-header">
                            <h2>📊 Volume Analysis: {intersection}</h2>
                            <p>📅 {date_range_vol[0].strftime('%b %d, %Y')} to {date_range_vol[1].strftime('%b %d, %Y')}
                            ({span} days) • {granularity_vol} Aggregation</p>
                            <p>📈 {len(filtered_volume_data):,} observations • Direction: {direction_filter}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        raw = base_df[(base_df["local_datetime"].dt.date >= date_range_vol[0]) &
                                      (base_df["local_datetime"].dt.date <= date_range_vol[1])].copy()
                        prev_range = prior_period(date_range_vol)
                        raw_prev = base_df[(base_df["local_datetime"].dt.date >= prev_range[0]) &
                                           (base_df["local_datetime"].dt.date <= prev_range[1])].copy()

                        st.subheader("🚦 Traffic Demand Performance Indicators")
                        if raw.empty:
                            st.info("No raw hourly volume in this window.")
                        else:
                            raw["total_volume"] = pd.to_numeric(raw["total_volume"], errors="coerce")
                            raw_prev["total_volume"] = pd.to_numeric(raw_prev["total_volume"], errors="coerce")

                            thr = get_capacity_for_selection(cap_df, intersection=intersection, direction=direction_filter)
                            cap_vph = thr["cap_vph"]; high_vph = thr["high_vph"]; critical_vph = thr["critical_vph"]

                            peak = float(np.nanmax(raw["total_volume"])) if raw["total_volume"].notna().any() else 0
                            p95 = float(np.nanpercentile(raw["total_volume"].dropna(), 95)) if raw["total_volume"].notna().any() else 0
                            avg = float(np.nanmean(raw["total_volume"])) if raw["total_volume"].notna().any() else 0
                            med = float(np.nanmedian(raw["total_volume"])) if raw["total_volume"].notna().any() else 0
                            ratio = (peak / avg) if avg > 0 else 0
                            cv = (float(np.nanstd(raw["total_volume"])) / avg * 100) if avg > 0 else 0
                            high_hours = int((raw["total_volume"] >= high_vph).sum())
                            total_hours = int(raw["total_volume"].count())
                            risk_pct = (high_hours / total_hours * 100) if total_hours > 0 else 0
                            util = (peak / cap_vph * 100) if cap_vph > 0 else 0

                            # previous period
                            peak_prev = float(np.nanmax(raw_prev["total_volume"])) if raw_prev["total_volume"].notna().any() else np.nan
                            avg_prev = float(np.nanmean(raw_prev["total_volume"])) if raw_prev["total_volume"].notna().any() else np.nan
                            high_hours_prev = int((raw_prev["total_volume"] >= high_vph).sum()) if raw_prev["total_volume"].notna().any() else np.nan
                            total_hours_prev = int(raw_prev["total_volume"].count())
                            risk_pct_prev = (high_hours_prev / total_hours_prev * 100) if total_hours_prev > 0 and np.isfinite(high_hours_prev) else np.nan

                            # Executive strip
                            if view_mode_vol == "Executive":
                                c1, c2, c3, c4 = st.columns(4)
                                with c1:
                                    st.metric("Peak Demand (vph)", f"{peak:,.0f}",
                                              delta=None if not np.isfinite(peak_prev) else f"{peak - peak_prev:+.0f} vs prev")
                                with c2:
                                    st.metric("Avg Demand (vph)", f"{avg:,.0f}",
                                              delta=None if not np.isfinite(avg_prev) else f"{avg - avg_prev:+.0f}")
                                with c3:
                                    st.metric("Peak Util (% of cap)", f"{util:.0f}%",
                                              delta=None)  # cap may change; we avoid misleading delta here
                                with c4:
                                    st.metric("High-Util Hours", f"{high_hours}",
                                              delta=None if not np.isfinite(risk_pct_prev) else f"{risk_pct - risk_pct_prev:+.1f} pp")

                            # Detailed KPI cards
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                if peak >= critical_vph: badge = "badge-critical"
                                elif peak >= high_vph: badge = "badge-poor"
                                elif peak > 0.6*cap_vph: badge = "badge-fair"
                                else: badge = "badge-good"
                                st.metric("🔥 Peak Demand", f"{peak:,.0f} vph", delta=f"95th: {p95:,.0f}")
                                st.markdown(f'<span class="performance-badge {badge}">{util:.0f}% of cap (~{int(cap_vph):,} vph)</span>', unsafe_allow_html=True)

                            with col2:
                                avg_util = (avg / cap_vph * 100) if cap_vph > 0 else 0
                                badge = "badge-good" if avg_util <= 40 else ("badge-fair" if avg_util <= 60 else "badge-poor")
                                st.metric("📊 Average Demand", f"{avg:,.0f} vph", delta=f"Median: {med:,.0f}")
                                st.markdown(f'<span class="performance-badge {badge}">{avg_util:.0f}% avg util</span>', unsafe_allow_html=True)

                            with col3:
                                badge = "badge-good" if ratio <= 2 else ("badge-fair" if ratio <= 3 else "badge-poor")
                                state = "Low" if ratio <= 2 else ("Moderate" if ratio <= 3 else "High")
                                st.metric("📈 Peak/Average Ratio", f"{ratio:.1f}x")
                                st.markdown(f'<span class="performance-badge {badge}">{state} Peaking</span>', unsafe_allow_html=True)

                            with col4:
                                badge = "badge-good" if cv < 30 else ("badge-fair" if cv < 50 else "badge-poor")
                                label = "Consistent" if cv < 30 else ("Variable" if cv < 50 else "Highly Variable")
                                st.metric("🎯 Demand Consistency", f"{max(0, 100 - cv):.0f}%", delta=f"CV: {cv:.1f}%")
                                st.markdown(f'<span class="performance-badge {badge}">{label}</span>', unsafe_allow_html=True)

                            with col5:
                                if risk_pct > 25: badge = "badge-critical"
                                elif risk_pct > 15: badge = "badge-poor"
                                elif risk_pct > 5: badge = "badge-fair"
                                else: badge = "badge-good"
                                level = "Very High" if risk_pct > 25 else ("High" if risk_pct > 15 else ("Moderate" if risk_pct > 5 else "Low"))
                                st.metric("⚠️ High Util Hours", f"{high_hours}",
                                          delta=None if not np.isfinite(risk_pct_prev) else f"{risk_pct - risk_pct_prev:+.1f} pp")
                                st.markdown(f'<span class="performance-badge {badge}">{level} Risk</span>', unsafe_allow_html=True)

                        # Legend
                        st.markdown(f"""
                        <div class="legend-box">
                            <b>How we set capacity:</b> Practical capacity is the <b>95th percentile hourly volume</b> by intersection &amp; direction over the last 90 days. 
                            <span class="chip">High Util ≥ 70% cap</span><span class="chip">Critical ≥ 90% cap</span>
                        </div>
                        """, unsafe_allow_html=True)

                        # Charts (compact for Execs)
                        if view_mode_vol == "Analyst" and len(filtered_volume_data) > 1:
                            chart1, chart2, chart3 = volume_charts(filtered_volume_data, cap_hint=cap_vph)
                            if chart1: st.plotly_chart(chart1, use_container_width=True)
                            colA, colB = st.columns(2)
                            with colA:
                                if chart3: st.plotly_chart(chart3, use_container_width=True)
                            with colB:
                                if chart2: st.plotly_chart(chart2, use_container_width=True)

                        # Ranking & actions
                        st.subheader("🚨 Intersection Volume & Capacity Risk Analysis")
                        try:
                            g = raw.groupby(["intersection_name", "direction"]).agg(
                                total_volume_mean=("total_volume", "mean"),
                                total_volume_max=("total_volume", "max"),
                                total_volume_std=("total_volume", "std"),
                                total_volume_count=("total_volume", "count")
                            ).reset_index()

                            g = g.merge(cap_df, on=["intersection_name", "direction"], how="left")
                            g["practical_capacity_vph"] = g["practical_capacity_vph"].fillna(DEFAULT_CAPACITY_VPH)

                            g["Peak_Capacity_Util"] = (g["total_volume_max"] / g["practical_capacity_vph"] * 100).round(1)
                            g["Avg_Capacity_Util"] = (g["total_volume_mean"] / g["practical_capacity_vph"] * 100).round(1)
                            g["Volume_Variability"] = (g["total_volume_std"] / g["total_volume_mean"] * 100).replace([np.inf, -np.inf], np.nan).fillna(0).round(1)
                            g["Peak_Avg_Ratio"] = (g["total_volume_max"] / g["total_volume_mean"]).replace([np.inf, -np.inf], 0).fillna(0).round(1)

                            g["🚨 Risk Score"] = (0.5 * g["Peak_Capacity_Util"] +
                                                  0.3 * g["Avg_Capacity_Util"] +
                                                  0.2 * (g["Peak_Avg_Ratio"] * 10)).round(1)

                            g["⚠️ Risk Level"] = pd.cut(g["🚨 Risk Score"],
                                bins=[0, 40, 60, 80, 90, 999],
                                labels=["🟢 Low Risk", "🟡 Moderate Risk", "🟠 High Risk", "🔴 Critical Risk", "🚨 Severe Risk"],
                                include_lowest=True
                            )
                            g["🎯 Action Priority"] = pd.cut(g["Peak_Capacity_Util"],
                                bins=[0, 70, 90, 999],
                                labels=["🟢 Monitor/Optimize", "🟠 Upgrade Timing", "🔴 Capacity Add"], include_lowest=True
                            )

                            final = g[[
                                "intersection_name","direction","⚠️ Risk Level","🎯 Action Priority","🚨 Risk Score",
                                "Peak_Capacity_Util","Avg_Capacity_Util","practical_capacity_vph",
                                "total_volume_mean","total_volume_max","Peak_Avg_Ratio","total_volume_count"
                            ]].rename(columns={
                                "intersection_name":"Intersection","direction":"Dir",
                                "Peak_Capacity_Util":"📊 Peak Capacity %","Avg_Capacity_Util":"📊 Avg Capacity %",
                                "practical_capacity_vph":"Practical Cap (vph)",
                                "total_volume_mean":"Avg Volume (vph)","total_volume_max":"Peak Volume (vph)",
                                "total_volume_count":"Data Points"
                            }).sort_values("🚨 Risk Score", ascending=False)

                            # Executive bullets
                            top3 = final.head(3)
                            bullets = "<br/>".join(
                                f"• <b>{r['Intersection']}</b> {r['Dir']} — Peak {r['Peak Volume (vph)']:.0f} ({r['📊 Peak Capacity %']:.0f}% of cap)"
                                for _, r in top3.iterrows()
                            )
                            st.markdown(f"""
                            <div class="insight-box">
                                <h4>💼 Executive Summary</h4>
                                <p><strong>Highest Capacity Stress:</strong><br/>{bullets if len(top3)>0 else "No high-stress intersections detected"}</p>
                                <p><strong>Operational Focus:</strong> Time-of-day plan refinement at high-stress locations; monitor High-Util hours and revisit splits.</p>
                            </div>
                            """, unsafe_allow_html=True)

                            st.dataframe(final.head(15), use_container_width=True,
                                column_config={
                                    "🚨 Risk Score": st.column_config.NumberColumn(
                                        "🚨 Capacity Risk Score", help="Composite of peak/avg util + peaking", format="%.1f", min_value=0, max_value=120
                                    ),
                                    "📊 Peak Capacity %": st.column_config.NumberColumn("📊 Peak Capacity %", format="%.1f%%"),
                                    "📊 Avg Capacity %": st.column_config.NumberColumn("📊 Avg Capacity %", format="%.1f%%"),
                                }
                            )
                            st.download_button("⬇️ Download Capacity Risk Table (CSV)",
                                               data=final.to_csv(index=False).encode("utf-8"),
                                               file_name="capacity_risk.csv", mime="text/csv")
                            st.download_button("⬇️ Download Filtered Volume (CSV)",
                                               data=filtered_volume_data.to_csv(index=False).encode("utf-8"),
                                               file_name="volume_filtered.csv", mime="text/csv")
                        except Exception as e:
                            st.error(f"❌ Error in volume analysis: {e}")
                            simple = raw.groupby(["intersection_name", "direction"]).agg(
                                Avg=("total_volume", "mean"), Peak=("total_volume", "max")
                            ).reset_index().sort_values("Peak", ascending=False)
                            st.dataframe(simple, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error processing volume data: {e}")
                st.info("Please check your data sources and try again.")
        else:
            st.warning("⚠️ Please select both start and end dates to proceed with the volume analysis.")

# =========================
# Footer
# =========================
st.markdown("""
---
<div style="text-align:center; padding: 1.25rem; background: linear-gradient(135deg, rgba(79,172,254,0.1), rgba(0,242,254,0.05));
    border-radius: 15px; margin-top: 1rem; border: 1px solid rgba(79,172,254,0.2);">
    <h4 style="color:#2980b9; margin-bottom: 0.5rem;">🛣️ Active Transportation & Operations Management Dashboard</h4>
    <p style="opacity:0.8; margin:0;">Powered by Advanced Machine Learning • Real-time Traffic Intelligence • Sustainable Transportation Solutions</p>
    <p style="opacity:0.6; margin-top: 0.25rem; font-size: 0.9rem;">© 2025 ADVANTEC Platform - Optimizing Transportation Networks</p>
</div>
""", unsafe_allow_html=True)
