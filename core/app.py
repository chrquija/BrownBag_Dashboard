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
DEFAULT_CAPACITY_VPH = 1600
DEFAULT_HIGH_UTIL = 0.70
DEFAULT_CRITICAL_UTIL = 0.90

# HCM-aligned delay references (not used for KPI titles, kept for context)
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
# CSS (polish)
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
        padding: 1.5rem; border-radius: 15px; margin: 1rem 0 1.25rem; color: white; text-align: center;
        box-shadow: 0 8px 32px rgba(79, 172, 254, 0.3); backdrop-filter: blur(10px);
    }
    .insight-box {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.15) 0%, rgba(0, 242, 254, 0.15) 100%);
        border-left: 5px solid #4facfe; border-radius: 12px; padding: 1rem 1.2rem; margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.1);
    }
    .legend-box {
        background: linear-gradient(135deg, rgba(46, 204, 113, .08), rgba(52, 152, 219, .08));
        border-left: 4px solid #2ecc71; border-radius: 10px; padding: .75rem .9rem; margin: .6rem 0 0;
        font-size: .92rem;
    }
    .chip {
        display:inline-block; padding:.15rem .55rem; border-radius:999px; font-size:.8rem; margin:.1rem .25rem;
        border:1px solid rgba(0,0,0,.06); background:rgba(255,255,255,.65);
    }
    .performance-badge { display: inline-block; padding: 0.35rem 0.9rem; border-radius: 25px; font-size: 0.85rem;
        font-weight: 600; margin: 0.2rem; border: 2px solid transparent; transition: all 0.3s ease; }
    .badge-excellent { background: linear-gradient(45deg, #2ecc71, #27ae60); color: white; }
    .badge-good { background: linear-gradient(45deg, #3498db, #2980b9); color: white; }
    .badge-fair { background: linear-gradient(45deg, #f39c12, #e67e22); color: white; }
    .badge-poor { background: linear-gradient(45deg, #e74c3c, #c0392b); color: white; }
    .badge-critical { background: linear-gradient(45deg, #e74c3c, #8e44ad); color: white; animation: pulse 2s infinite; }
    @keyframes pulse { 0% {opacity:1} 50% {opacity:.7} 100% {opacity:1} }
    .stTabs [data-baseweb="tab-list"] { gap: 16px; }
    .stTabs [data-baseweb="tab"] { height: 56px; padding: 0 18px; border-radius: 12px;
        background: rgba(79, 172, 254, 0.1); border: 1px solid rgba(79, 172, 254, 0.2); }
    .modebar { filter: saturate(0.9) opacity(0.95); }
</style>
""", unsafe_allow_html=True)

# =========================
# Title / Intro
# =========================
st.markdown("""
<div class="main-container">
    <h1 style="text-align:center; margin:0; font-size:2.3rem; font-weight:800;">
        🛣️ Active Transportation & Operations Management Dashboard
    </h1>
    <p style="text-align:center; margin-top:0.8rem; font-size:1.05rem; opacity:0.9;">
        Advanced Traffic Engineering & Operations Management Platform
    </p>
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
    return _safe_to_datetime(df.copy(), "local_datetime")

@st.cache_data(show_spinner=False)
def get_volume_df() -> pd.DataFrame:
    df = load_volume_data()
    if df is None or len(df) == 0:
        return pd.DataFrame()
    return _safe_to_datetime(df.copy(), "local_datetime")

# --- Corridor ordering helpers ---
CORRIDOR_ORDER_KEYS = [
    "Ave 52", "Ave 50", "Ave 48", "Calle Tampico",
    "Fred Waring", "Fred Waring Dr", "Miles Ave",
    "Point Happy", "Point Happy Simon", "Hwy 111", "Highway 111", "Ave 47"
]

def _key_index(name: str) -> int:
    s = name or ""
    for i, k in enumerate(CORRIDOR_ORDER_KEYS):
        if k.lower() in s.lower():
            return i
    return 10_000

def order_segments_for_dropdown(names: list[str]) -> list[str]:
    def seg_pos(seg: str):
        parts = [p.strip() for p in seg.split("->")]
        left = parts[0] if parts else seg
        right = parts[1] if len(parts) > 1 else seg
        return (_key_index(left), _key_index(right), seg.lower())
    return sorted(names, key=seg_pos)

def order_intersections_for_dropdown(names: list[str]) -> list[str]:
    def xstreet(n: str):
        if "&" in n:
            return n.split("&", 1)[1].strip()
        return n
    def int_pos(n: str):
        return (_key_index(xstreet(n)), n.lower())
    return sorted(names, key=int_pos)

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

def get_capacity_for_selection(cap_df: pd.DataFrame, intersection: str, direction: str):
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

# =========================
# Charts
# =========================
def performance_chart(data: pd.DataFrame, metric_type: str = "delay"):
    if data.empty: return None
    metric_type = metric_type.lower().strip()
    if metric_type == "delay":
        y_col, title = "average_delay", "Traffic Delay Analysis"
        y_label = "Average Delay (s/veh)"
    else:
        y_col, title = "average_traveltime", "Travel Time Analysis"
        y_label = "Average Travel Time (min)"

    dd = data.dropna(subset=["local_datetime", y_col]).sort_values("local_datetime")

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Time Series", "Distribution"),
        vertical_spacing=0.12
    )
    fig.add_trace(
        go.Scatter(
            x=dd["local_datetime"], y=dd[y_col],
            mode="lines+markers", name="Series"
        ), row=1, col=1
    )
    fig.add_trace(
        go.Histogram(
            x=dd[y_col], nbinsx=30, name="Histogram", opacity=0.85
        ), row=2, col=1
    )
    fig.update_layout(
        height=600, title=title, showlegend=False, template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    fig.update_xaxes(title_text="Date / Time", row=1, col=1)
    fig.update_yaxes(title_text=y_label,       row=1, col=1)
    fig.update_xaxes(title_text=y_label,       row=2, col=1)
    fig.update_yaxes(title_text="Frequency",   row=2, col=1)
    return fig

def create_volume_charts(data, cap_hint: float = None):
    """Clean volume charts with explicit axis labels and layout that won't overlap."""
    if data.empty:
        return None, None, None

    dd = data.dropna(subset=["local_datetime", "total_volume", "intersection_name"]).copy()
    dd.sort_values("local_datetime", inplace=True)

    # 1) Hourly pattern (line)
    dd["hour"] = dd["local_datetime"].dt.hour
    hourly_volume = dd.groupby(["hour", "intersection_name"], as_index=False)["total_volume"].mean()

    fig_hourly = px.line(
        hourly_volume,
        x="hour", y="total_volume", color="intersection_name",
        title="Average Hourly Volume Pattern",
        labels={"hour": "Hour of Day", "total_volume": "Average Volume (vph)", "intersection_name": "Intersection"},
        template="plotly_white"
    )
    if cap_hint:
        fig_hourly.add_hline(y=cap_hint, line_dash="dash", annotation_text=f"Practical Capacity ≈ {int(cap_hint):,} vph")
        fig_hourly.add_hline(y=0.7*cap_hint, line_dash="dot", annotation_text="High Util (~70% cap)")
    fig_hourly.update_layout(
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=40)
    )
    fig_hourly.update_xaxes(dtick=1)

    # 2) Distribution by intersection (box)
    fig_box = px.box(
        dd, x="intersection_name", y="total_volume", points="outliers",
        title="Volume Distribution by Intersection",
        labels={"intersection_name": "Intersection", "total_volume": "Volume (vph)"},
        template="plotly_white"
    )
    fig_box.update_layout(
        height=480,
        margin=dict(l=20, r=20, t=60, b=140)
    )
    fig_box.update_xaxes(tickangle=35)

    # 3) Hourly heatmap
    hourly_avg = dd.groupby(["intersection_name", "hour"], as_index=False)["total_volume"].mean()
    heat_pivot = hourly_avg.pivot(index="intersection_name", columns="hour", values="total_volume").sort_index()

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=heat_pivot.values,
            x=heat_pivot.columns,
            y=heat_pivot.index,
            colorscale="Blues",
            colorbar=dict(title="Avg Volume (vph)")
        )
    )
    fig_heat.update_layout(
        title="Hourly Average Volume (Heatmap)",
        xaxis_title="Hour of Day",
        yaxis_title="Intersection",
        template="plotly_white",
        height=520,
        margin=dict(l=80, r=20, t=60, b=40)
    )

    return fig_hourly, fig_box, fig_heat

# =========================
# Insight helpers
# =========================
def generate_bottleneck_sentence(raw_df: pd.DataFrame,
                                 ff_df: pd.DataFrame,
                                 date_range: tuple,
                                 high_delay_sec: int = 55,
                                 tti_congested: float = 1.25,
                                 speed_ff_thresh: float = 0.70) -> str:
    """One plain-English sentence describing the most likely source of delay and why."""
    if raw_df.empty:
        return "No performance issues detected in the selected period."

    df = raw_df.copy()
    df["local_datetime"] = pd.to_datetime(df["local_datetime"], errors="coerce")
    df = df.dropna(subset=["segment_name", "direction", "average_traveltime", "average_speed", "average_delay"])
    base = df.merge(ff_df, on=["segment_name", "direction"], how="left")
    base["TTI"] = base["average_traveltime"] / base["ff_traveltime_min"]
    base["SpeedRatio"] = base["average_speed"] / base["ff_speed_mph"]
    base["hour"] = base["local_datetime"].dt.hour

    am_mask = base["hour"].between(7, 9, inclusive="both")
    pm_mask = base["hour"].between(16, 18, inclusive="both")

    def pct_cong(sub, mask):
        s = sub[mask]
        if s.empty: return 0.0
        return float(((s["TTI"] >= tti_congested) | (s["SpeedRatio"] <= speed_ff_thresh)).mean() * 100)

    g = (base.groupby(["segment_name", "direction"])
              .agg(p95_delay=("average_delay", lambda s: np.nanpercentile(pd.to_numeric(s, errors="coerce").dropna(), 95) if pd.to_numeric(s, errors="coerce").notna().any() else np.nan),
                   med_tti=("TTI", "median"),
                   med_speed_ratio=("SpeedRatio", "median"),
                   cv_tt=("average_traveltime", lambda s: (np.nanstd(s) / np.nanmean(s) * 100) if np.nanmean(s) > 0 else np.nan),
                   hi_delay_pct=("average_delay", lambda s: (pd.to_numeric(s, errors="coerce") > high_delay_sec).mean() * 100 if pd.to_numeric(s, errors="coerce").notna().any() else 0),
                   n=("average_delay","count"))).reset_index()

    # Add AM/PM congestion %
    g["pm_cong_pct"] = g.apply(lambda r: pct_cong(base[(base["segment_name"]==r["segment_name"]) &
                                                       (base["direction"]==r["direction"])], pm_mask), axis=1)
    g["am_cong_pct"] = g.apply(lambda r: pct_cong(base[(base["segment_name"]==r["segment_name"]) &
                                                       (base["direction"]==r["direction"])], am_mask), axis=1)

    g["score"] = 0.6*g["p95_delay"].fillna(0) + 0.4*(g["med_tti"].fillna(1.0)*100)
    worst = g.sort_values("score", ascending=False).head(1)
    if worst.empty:
        return "No performance issues detected in the selected period."

    r = worst.iloc[0]
    seg = str(r["segment_name"])
    d   = str(r["direction"])

    causes = []
    if r["pm_cong_pct"] >= 20: causes.append(f"recurrent PM-peak congestion (TTI≥{tti_congested:.2f} / Speed≤{speed_ff_thresh:.2f} in {r['pm_cong_pct']:.0f}% of PM hours)")
    if r["am_cong_pct"] >= 20: causes.append(f"recurrent AM-peak congestion (TTI≥{tti_congested:.2f} / Speed≤{speed_ff_thresh:.2f} in {r['am_cong_pct']:.0f}% of AM hours)")
    if pd.notna(r["med_speed_ratio"]) and r["med_speed_ratio"] < 0.80: causes.append(f"low Speed/FF ≈ {r['med_speed_ratio']:.2f}")
    if pd.notna(r["cv_tt"]) and r["cv_tt"] >= 35: causes.append(f"high travel-time variability (CV {r['cv_tt']:.0f}%)")
    if pd.notna(r["hi_delay_pct"]) and r["hi_delay_pct"] >= 10: causes.append(f"frequent high-delay hours (>{high_delay_sec}s in {r['hi_delay_pct']:.0f}% of hours)")
    if not causes: causes.append("isolated peak-period spikes")

    start, end = date_range
    period_text = f"{start.strftime('%b %d, %Y')} – {end.strftime('%b %d, %Y')}"
    return f"Most likely source of delay: <b>{seg} ({d})</b> during <b>{period_text}</b>, due to {', '.join(causes[:2])}."

def generate_volume_insights(raw_df: pd.DataFrame, cap_vph: float, high_vph: float, date_range: tuple) -> str:
    if raw_df.empty:
        return "<div class='insight-box'><p>No volume data for the selected period.</p></div>"

    df = raw_df.copy()
    df["hour"] = pd.to_datetime(df["local_datetime"], errors="coerce").dt.hour
    df["total_volume"] = pd.to_numeric(df["total_volume"], errors="coerce")

    by_int = df.groupby("intersection_name", as_index=False)["total_volume"].mean().dropna()
    if by_int.empty:
        return "<div class='insight-box'><p>No volume data for the selected period.</p></div>"

    top = by_int.sort_values("total_volume", ascending=False).head(1)
    low = by_int.sort_values("total_volume", ascending=True).head(1)

    top_name = top.iloc[0]["intersection_name"]
    top_hourly = (df[df["intersection_name"] == top_name]
                    .groupby("hour", as_index=False)["total_volume"].mean()
                    .sort_values("total_volume", ascending=False))
    top_peak_hour = int(top_hourly.iloc[0]["hour"]) if not top_hourly.empty else None
    top_avg = float(top.iloc[0]["total_volume"])

    low_name = low.iloc[0]["intersection_name"]
    low_avg = float(low.iloc[0]["total_volume"])

    corridor_hourly = df.groupby("hour", as_index=False)["total_volume"].mean().sort_values("total_volume", ascending=False)
    corridor_peak_hour = int(corridor_hourly.iloc[0]["hour"]) if not corridor_hourly.empty else None

    int_hourly_peak = df.groupby(["intersection_name", "hour"], as_index=False)["total_volume"].mean()
    int_peak = int_hourly_peak.groupby("intersection_name", as_index=False)["total_volume"].max()
    n_high = int((int_peak["total_volume"] >= high_vph).sum())
    n_cap  = int((int_peak["total_volume"] >= cap_vph).sum())

    start, end = date_range
    period_text = f"{start.strftime('%b %d, %Y')} – {end.strftime('%b %d, %Y')}"
    return f"""
    <div class="insight-box">
        <h4>Volume Analysis Insights <span style="font-weight:400;">({period_text})</span></h4>
        <ul style="margin:.4rem 0 0 1.1rem;">
            <li>Highest average: <b>{top_name}</b> (~{top_avg:,.0f} vph; peak ~{top_peak_hour}:00).</li>
            <li>Lowest average: <b>{low_name}</b> (~{low_avg:,.0f} vph).</li>
            <li>Corridor peak: around <b>{corridor_peak_hour}:00</b>; {n_high} intersections over the high-volume threshold (~{int(high_vph):,} vph) and {n_cap} near capacity (~{int(cap_vph):,} vph).</li>
        </ul>
    </div>
    """

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

    custom = st.date_input(
        "Custom Date Range",
        value=st.session_state[k_range],
        min_value=min_date, max_value=max_date,
        key=f"{key_prefix}_custom"
    )
    if custom != st.session_state[k_range]:
        st.session_state[k_range] = custom
    return st.session_state[k_range]

def prior_period(date_range):
    start, end = date_range
    delta = (end - start)
    prev_end = start - timedelta(days=1)
    prev_start = prev_end - delta
    return (prev_start, prev_end)

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["🚧 Performance & Delay Analysis", "📊 Traffic Demand & Capacity"])

# -------------------------
# TAB 1: Performance / Travel Time
# -------------------------
with tab1:
    st.header("Performance")

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Loading performance data…")
    progress_bar.progress(25)

    corridor_df = get_corridor_df()
    progress_bar.progress(100)

    if corridor_df.empty:
        st.error("❌ Failed to load performance data.")
    else:
        status_text.text("✅ Data loaded")
        time.sleep(0.3); progress_bar.empty(); status_text.empty()

        # ----- Sidebar controls -----
        with st.sidebar:
            with st.expander("🚧 Performance Controls", expanded=False):
                # Ordered segment dropdown
                seg_all = list(corridor_df["segment_name"].dropna().unique())
                seg_ordered = order_segments_for_dropdown(seg_all)
                seg_options = ["All Segments"] + seg_ordered
                corridor = st.selectbox("🛣️ Segment", seg_options, help="South → North order", key="perf_segment")

                min_date = corridor_df["local_datetime"].dt.date.min()
                max_date = corridor_df["local_datetime"].dt.date.max()
                st.markdown("#### 📅 Period")
                date_range = date_range_preset_controls(min_date, max_date, key_prefix="perf")

                st.markdown("#### ⏰ Settings")
                granularity = st.selectbox("Aggregation", ["Hourly", "Daily", "Weekly", "Monthly"], index=0, key="perf_aggregation")
                time_filter, start_hour, end_hour = None, None, None
                if granularity == "Hourly":
                    time_filter = st.selectbox(
                        "Time Focus",
                        ["All Hours", "Peak Hours (7–9 AM, 4–6 PM)", "AM Peak (7–9 AM)", "PM Peak (4–6 PM)", "Off-Peak", "Custom Range"],
                        key="perf_time_focus"
                    )
                    if time_filter == "Custom Range":
                        c1, c2 = st.columns(2)
                        with c1: start_hour = st.number_input("Start Hour", 0, 23, 7, key="perf_start_hour")
                        with c2: end_hour   = st.number_input("End Hour", 1, 24, 18, key="perf_end_hour")

                st.markdown("#### 🚦 Congestion Definition")
                cong_logic = st.selectbox(
                    "Define Congestion Using",
                    ["TTI only", "Speed/FF only", "TTI OR Speed/FF", "TTI AND Speed/FF"],
                    index=2, key="perf_cong_logic"
                )
                tti_thresh = st.slider("TTI threshold (≥)", 1.05, 2.00, 1.25, 0.05, key="perf_tti_thresh")
                speedff_thresh = st.slider("Speed / Free-Flow threshold (≤)", 0.40, 1.00, 0.70, 0.05, key="perf_speedff_thresh")

        if len(date_range) == 2:
            try:
                base_df = corridor_df.copy()
                if corridor != "All Segments":
                    base_df = base_df[base_df["segment_name"] == corridor]

                if base_df.empty:
                    st.warning("⚠️ No data for the selected segment.")
                else:
                    # Free-flow baselines
                    ff_df = compute_freeflow_baselines(base_df)

                    # Aggregate for charts
                    filtered_data = process_traffic_data(
                        base_df, date_range, granularity,
                        time_filter if granularity == "Hourly" else None,
                        start_hour, end_hour
                    )

                    # Header (minimal)
                    st.markdown(f"""
                    <div class="context-header">
                        <h2>📊 {corridor}</h2>
                        <p>{date_range[0].strftime('%b %d, %Y')} – {date_range[1].strftime('%b %d, %Y')}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if filtered_data.empty:
                        st.warning("No data for the selected filters.")
                    else:
                        # Raw window for KPIs
                        raw_cur = base_df[(base_df["local_datetime"].dt.date >= date_range[0]) &
                                          (base_df["local_datetime"].dt.date <= date_range[1])].copy()
                        prev_range = prior_period(date_range)
                        raw_prev = base_df[(base_df["local_datetime"].dt.date >= prev_range[0]) &
                                           (base_df["local_datetime"].dt.date <= prev_range[1])].copy()

                        for col in ["average_delay", "average_traveltime", "average_speed"]:
                            if col in raw_cur: raw_cur[col] = pd.to_numeric(raw_cur[col], errors="coerce")
                            if col in raw_prev: raw_prev[col] = pd.to_numeric(raw_prev[col], errors="coerce")

                        cur_tmp = raw_cur.merge(ff_df, on=["segment_name", "direction"], how="left")
                        prev_tmp = raw_prev.merge(ff_df, on=["segment_name", "direction"], how="left")
                        raw_cur["TTI"] = cur_tmp["average_traveltime"] / cur_tmp["ff_traveltime_min"]
                        raw_cur["SpeedRatio"] = cur_tmp["average_speed"] / cur_tmp["ff_speed_mph"]
                        raw_prev["TTI"] = prev_tmp["average_traveltime"] / prev_tmp["ff_traveltime_min"]
                        raw_prev["SpeedRatio"] = prev_tmp["average_speed"] / prev_tmp["ff_speed_mph"]

                        # ======= PERFORMANCE KPIs (your requested set) =======
                        def _nanmean(s):
                            s = pd.to_numeric(s, errors="coerce").dropna()
                            return float(np.nanmean(s)) if len(s) else np.nan
                        def _nanmin(s):
                            s = pd.to_numeric(s, errors="coerce").dropna()
                            return float(np.nanmin(s)) if len(s) else np.nan
                        def _nanmax(s):
                            s = pd.to_numeric(s, errors="coerce").dropna()
                            return float(np.nanmax(s)) if len(s) else np.nan
                        def _p95(s):
                            s = pd.to_numeric(s, errors="coerce").dropna()
                            return float(np.nanpercentile(s, 95)) if len(s) else np.nan

                        avg_tt  = _nanmean(raw_cur["average_traveltime"])
                        min_tt  = _nanmin(raw_cur["average_traveltime"])
                        peak_tt = _nanmax(raw_cur["average_traveltime"])
                        p95_tt  = _p95(raw_cur["average_traveltime"])
                        # Reliability Index = 95th / average
                        ri = (p95_tt / avg_tt) if (avg_tt and np.isfinite(avg_tt) and avg_tt > 0 and np.isfinite(p95_tt)) else np.nan

                        # Congestion Frequency per definition
                        def cong_mask(tti, sr, logic):
                            if logic == "TTI only":
                                return tti >= tti_thresh
                            if logic == "Speed/FF only":
                                return sr <= speedff_thresh
                            if logic == "TTI OR Speed/FF":
                                return (tti >= tti_thresh) | (sr <= speedff_thresh)
                            return (tti >= tti_thresh) & (sr <= speedff_thresh)

                        cong_pct = 0.0
                        if raw_cur[["TTI","SpeedRatio"]].notna().any().any():
                            cong_pct = float(cong_mask(raw_cur["TTI"], raw_cur["SpeedRatio"], cong_logic).mean() * 100)

                        # Previous period for deltas (optional)
                        avg_tt_prev = _nanmean(raw_prev["average_traveltime"])
                        p95_tt_prev = _p95(raw_prev["average_traveltime"])
                        ri_prev = (p95_tt_prev / avg_tt_prev) if (avg_tt_prev and np.isfinite(avg_tt_prev) and avg_tt_prev > 0 and np.isfinite(p95_tt_prev)) else np.nan
                        cong_pct_prev = (cong_mask(raw_prev["TTI"], raw_prev["SpeedRatio"], cong_logic).mean() * 100) if raw_prev[["TTI","SpeedRatio"]].notna().any().any() else np.nan

                        # KPI row (minimal, stakeholder-friendly)
                        c1, c2, c3, c4, c5 = st.columns(5)
                        with c1:
                            st.metric("Peak Travel Time", f"{peak_tt:.1f} min" if np.isfinite(peak_tt) else "—")
                        with c2:
                            st.metric("Minimum Travel Time", f"{min_tt:.1f} min" if np.isfinite(min_tt) else "—")
                        with c3:
                            st.metric("Average Travel Time", f"{avg_tt:.1f} min" if np.isfinite(avg_tt) else "—")
                        with c4:
                            st.metric("Reliability Index", f"{ri:.2f}" if np.isfinite(ri) else "—",
                                      delta=None if not np.isfinite(ri_prev) else f"{ri - ri_prev:+.2f}")
                        with c5:
                            st.metric("Congestion Frequency", f"{cong_pct:.1f}%",
                                      delta=None if not np.isfinite(cong_pct_prev) else f"{cong_pct - cong_pct_prev:+.1f} pp")

                        # Legend
                        st.markdown(f"""
                        <div class="legend-box">
                            <b>Method:</b> Free-flow = 10th percentile travel time & 90th percentile speed during 22:00–05:00.
                            <span class="chip">TTI = Travel Time / Free-Flow</span>
                            <span class="chip">Speed/FF = Speed / Free-Flow</span>
                            <span class="chip">Congestion if {cong_logic} (TTI ≥ {tti_thresh:.2f}, Speed/FF ≤ {speedff_thresh:.2f})</span>
                        </div>
                        """, unsafe_allow_html=True)

                        # Charts
                        if len(filtered_data) > 1:
                            st.subheader("Trends")
                            v1, v2 = st.columns(2)
                            with v1:
                                dc = performance_chart(filtered_data, "delay")
                                if dc: st.plotly_chart(dc, use_container_width=True)
                            with v2:
                                tc = performance_chart(filtered_data, "travel")
                                if tc: st.plotly_chart(tc, use_container_width=True)

                        # Advanced Performance Insights (one-liner)
                        insight_sentence = generate_bottleneck_sentence(
                            raw_df=raw_cur, ff_df=ff_df, date_range=date_range,
                            high_delay_sec=HIGH_DELAY_SEC, tti_congested=tti_thresh, speed_ff_thresh=speedff_thresh
                        )
                        st.markdown(f"""
                        <div class="insight-box">
                            <h4>Advanced Performance Insights</h4>
                            <p>{insight_sentence}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Bottlenecks table (kept as before)
                        st.subheader("Bottlenecks")
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

                                final = g[[
                                    "segment_name","direction","Bottleneck_Score",
                                    "average_delay_mean","average_delay_max",
                                    "average_traveltime_mean","average_traveltime_max",
                                    "average_speed_mean","average_speed_min","n"
                                ]].rename(columns={
                                    "segment_name":"Segment","direction":"Dir",
                                    "average_delay_mean":"Avg Delay (s)","average_delay_max":"Peak Delay (s)",
                                    "average_traveltime_mean":"Avg Time (min)","average_traveltime_max":"Peak Time (min)",
                                    "average_speed_mean":"Avg Speed (mph)","average_speed_min":"Min Speed (mph)",
                                    "n":"Obs"
                                }).sort_values("Bottleneck_Score", ascending=False)

                                top3 = final.head(3)
                                bullets = "<br/>".join(
                                    f"• <b>{r['Segment']}</b> ({r['Dir']}) — Impact {r['Bottleneck_Score']:.1f}, Peak Delay {r['Peak Delay (s)']:.0f}s"
                                    for _, r in top3.iterrows()
                                )
                                st.markdown(f"""
                                <div class="insight-box">
                                    <h4>Executive Summary</h4>
                                    <p><strong>Top Bottlenecks (current period: {date_range[0].strftime('%b %d, %Y')} – {date_range[1].strftime('%b %d, %Y')}):</strong><br/>{bullets if len(top3)>0 else "No bottlenecks detected."}</p>
                                    <p><em>Note:</em> “Intersections” and segment names refer to Washington St at key cross streets within the Ave 52 → Hwy 111 study area.</p>
                                </div>
                                """, unsafe_allow_html=True)

                                st.dataframe(
                                    final.head(15),
                                    use_container_width=True,
                                    column_config={
                                        "Bottleneck_Score": st.column_config.NumberColumn(
                                            "Impact Score (0–100)", help="Composite severity score; higher = worse", format="%.1f"
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
                st.error(f"❌ Error processing performance data: {e}")
        else:
            st.warning("⚠️ Please select both start and end dates to proceed.")

# -------------------------
# TAB 2: Volume / Capacity
# -------------------------
with tab2:
    st.header("Demand & Capacity")

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Loading volume data…")
    progress_bar.progress(25)

    volume_df = get_volume_df()
    progress_bar.progress(100)

    if volume_df.empty:
        st.error("❌ Failed to load volume data.")
    else:
        status_text.text("✅ Data loaded")
        time.sleep(0.3); progress_bar.empty(); status_text.empty()

        cap_df = compute_practical_capacity(volume_df, window_days=90)

        with st.sidebar:
            with st.expander("📊 Volume Controls", expanded=False):
                ints_all = list(volume_df["intersection_name"].dropna().unique())
                ints_ordered = order_intersections_for_dropdown(ints_all)
                intersection_options = ["All Intersections"] + ints_ordered
                intersection = st.selectbox("🚦 Intersection", intersection_options, help="South → North order", key="vol_intersection")

                min_date = volume_df["local_datetime"].dt.date.min()
                max_date = volume_df["local_datetime"].dt.date.max()
                st.markdown("#### 📅 Period")
                date_range_vol = date_range_preset_controls(min_date, max_date, key_prefix="vol")

                st.markdown("#### ⏰ Settings")
                granularity_vol = st.selectbox("Aggregation", ["Hourly", "Daily", "Weekly", "Monthly"], index=0, key="vol_aggregation")
                direction_options = ["All Directions"] + sorted(volume_df["direction"].dropna().unique().tolist())
                direction_filter = st.selectbox("Direction", direction_options, key="vol_direction")

        if len(date_range_vol) == 2:
            try:
                base_df = volume_df.copy()
                if intersection != "All Intersections":
                    base_df = base_df[base_df["intersection_name"] == intersection]
                if direction_filter != "All Directions":
                    base_df = base_df[base_df["direction"] == direction_filter]

                st.markdown(f"""
                <div class="context-header">
                    <h2>📊 {intersection}</h2>
                    <p>{date_range_vol[0].strftime('%b %d, %Y')} – {date_range_vol[1].strftime('%b %d, %Y')}</p>
                </div>
                """, unsafe_allow_html=True)

                if base_df.empty:
                    st.warning("⚠️ No volume data for the selected filters.")
                else:
                    filtered_volume_data = process_traffic_data(base_df, date_range_vol, granularity_vol)
                    if filtered_volume_data.empty:
                        st.warning("No data for the selected range.")
                    else:
                        raw = base_df[(base_df["local_datetime"].dt.date >= date_range_vol[0]) &
                                      (base_df["local_datetime"].dt.date <= date_range_vol[1])].copy()
                        prev_range = prior_period(date_range_vol)
                        raw_prev = base_df[(base_df["local_datetime"].dt.date >= prev_range[0]) &
                                           (base_df["local_datetime"].dt.date <= prev_range[1])].copy()

                        if raw.empty:
                            st.info("No raw hourly volume in this window.")
                        else:
                            raw["total_volume"] = pd.to_numeric(raw["total_volume"], errors="coerce")
                            raw_prev["total_volume"] = pd.to_numeric(raw_prev["total_volume"], errors="coerce")

                            thr = get_capacity_for_selection(cap_df, intersection=intersection, direction=direction_filter)
                            cap_vph = thr["cap_vph"]; high_vph = thr["high_vph"]; critical_vph = thr["critical_vph"]

                            # ====== KEEPING YOUR EXISTING VOLUME KPIs ======
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

                            c1, c2, c3, c4, c5 = st.columns(5)
                            with c1:
                                st.metric("Peak (vph)", f"{peak:,.0f}",
                                          delta=None if not np.isfinite(peak_prev) else f"{peak - peak_prev:+.0f}")
                            with c2:
                                st.metric("Average (vph)", f"{avg:,.0f}",
                                          delta=None if not np.isfinite(avg_prev) else f"{avg - avg_prev:+.0f}")
                            with c3:
                                st.metric("Peak Utilization", f"{util:.0f}%", help=f"Share of practical capacity (~{int(cap_vph):,} vph)")
                            with c4:
                                st.metric("Consistency", f"{max(0, 100 - cv):.0f}%", delta=f"CV {cv:.1f}%")
                            with c5:
                                st.metric("High-Util Hours", f"{high_hours}",
                                          delta=None if not np.isfinite(risk_pct_prev) else f"{risk_pct - risk_pct_prev:+.1f} pp")

                            # Charts (no overlap)
                            st.subheader("Trends")
                            chart1, chart2, chart3 = create_volume_charts(filtered_volume_data, cap_hint=cap_vph)
                            if chart1: st.plotly_chart(chart1, use_container_width=True)
                            colA, colB = st.columns(2, gap="large")
                            with colA:
                                if chart2: st.plotly_chart(chart2, use_container_width=True)
                            with colB:
                                if chart3: st.plotly_chart(chart3, use_container_width=True)

                            # Insights (simple bullets)
                            st.markdown(
                                generate_volume_insights(raw, cap_vph=cap_vph, high_vph=high_vph, date_range=date_range_vol),
                                unsafe_allow_html=True
                            )

                            # Capacity risk table (unchanged)
                            st.subheader("Capacity Risk")
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

                                g["Risk_Score"] = (0.5 * g["Peak_Capacity_Util"] +
                                                   0.3 * g["Avg_Capacity_Util"] +
                                                   0.2 * (g["Peak_Avg_Ratio"] * 10)).round(1)

                                final = g[[
                                    "intersection_name","direction","Risk_Score",
                                    "Peak_Capacity_Util","Avg_Capacity_Util","practical_capacity_vph",
                                    "total_volume_mean","total_volume_max","Peak_Avg_Ratio","total_volume_count"
                                ]].rename(columns={
                                    "intersection_name":"Intersection","direction":"Dir",
                                    "Peak_Capacity_Util":"Peak Capacity %","Avg_Capacity_Util":"Avg Capacity %",
                                    "practical_capacity_vph":"Practical Cap (vph)",
                                    "total_volume_mean":"Avg Volume (vph)","total_volume_max":"Peak Volume (vph)",
                                    "total_volume_count":"Data Points"
                                }).sort_values("Risk_Score", ascending=False)

                                st.dataframe(
                                    final.head(15),
                                    use_container_width=True,
                                    column_config={
                                        "Risk_Score": st.column_config.NumberColumn(
                                            "Capacity Risk (0–120)", help="Composite of peak/avg utilization & demand peaking", format="%.1f", min_value=0, max_value=120
                                        ),
                                        "Peak Capacity %": st.column_config.NumberColumn("Peak Capacity %", format="%.1f%%"),
                                        "Avg Capacity %": st.column_config.NumberColumn("Avg Capacity %", format="%.1f%%"),
                                    }
                                )
                                st.download_button("⬇️ Download Capacity Risk Table (CSV)",
                                                   data=final.to_csv(index=False).encode("utf-8"),
                                                   file_name="capacity_risk.csv", mime="text/csv")
                                st.download_button("⬇️ Download Filtered Volume (CSV)",
                                                   data=filtered_volume_data.to_csv(index=False).encode("utf-8"),
                                                   file_name="volume_filtered.csv", mime="text/csv")
                            except Exception as e:
                                st.error(f"❌ Error in capacity analysis: {e}")
            except Exception as e:
                st.error(f"❌ Error processing volume data: {e}")
        else:
            st.warning("⚠️ Please select both start and end dates to proceed.")

# =========================
# Footer
# =========================
st.markdown("""
---
<div style="text-align:center; padding: 1.1rem; background: linear-gradient(135deg, rgba(79,172,254,0.1), rgba(0,242,254,0.05));
    border-radius: 15px; margin-top: 0.8rem; border: 1px solid rgba(79,172,254,0.2);">
    <h4 style="color:#2980b9; margin-bottom: 0.4rem;">🛣️ Active Transportation & Operations Management Dashboard</h4>
    <p style="opacity:0.8; margin:0;">Data-driven corridor operations • Real-time insights • Sustainable mobility</p>
    <p style="opacity:0.6; margin-top: 0.2rem; font-size: 0.9rem;">© 2025 ADVANTEC Platform</p>
</div>
""", unsafe_allow_html=True)
