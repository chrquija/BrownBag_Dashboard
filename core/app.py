# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import streamlit.components.v1 as components

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
    # volume_charts,  # <- no longer used in TAB 2
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

# -------- Canonical helpers (used for robust O-D path building) --------
def _nodes_present_in_data(df: pd.DataFrame) -> set:
    """All node labels that appear in any 'A → B' segment_name."""
    if "segment_name" not in df.columns or df.empty:
        return set()
    parts = df["segment_name"].dropna().str.split("→")
    left = parts.apply(lambda x: x[0].strip() if isinstance(x, list) and len(x) == 2 else None)
    right = parts.apply(lambda x: x[1].strip() if isinstance(x, list) and len(x) == 2 else None)
    return set(pd.concat([left, right], ignore_index=True).dropna().unique())

def _canonical_order_in_data(df: pd.DataFrame) -> list[str]:
    """Canonical corridor order, restricted to nodes that actually exist in the data."""
    present = _nodes_present_in_data(df)
    return [n for n in DESIRED_NODE_ORDER_BOTTOM_UP if n in present]

# =========================
# Robust direction normalization (string-only)
# =========================
def normalize_dir(s: pd.Series) -> pd.Series:
    """
    Vectorized normalizer returning only 'nb', 'sb', or 'unk' (dtype=object).
    Safe for mixed dtype inputs; never returns NaN.
    """
    ser = s.astype(str).str.lower().str.strip()
    ser = ser.str.replace(r"[\s\-\(\)_/\\]+", " ", regex=True)
    nb_mask = ser.str.contains(r"\b(nb|north|northbound)\b", regex=True)
    sb_mask = ser.str.contains(r"\b(sb|south|southbound)\b", regex=True)
    return pd.Series(
        np.where(nb_mask, "nb", np.where(sb_mask, "sb", "unk")),
        index=ser.index,
        dtype="object",
    )

def normalize_dir_value(v) -> str:
    """Scalar helper if ever needed; string-only returns."""
    if v is None:
        return "unk"
    try:
        s = str(v).lower().strip()
    except Exception:
        return "unk"
    s = " ".join([tok for tok in s.replace("-", " ").replace("_", " ").split()])
    if any(t in s for t in [" nb", "nb ", " northbound", " north "]):
        return "nb"
    if any(t in s for t in [" sb", "sb ", " southbound", " south "]):
        return "sb"
    return "unk"

# =========================
# Extra CSS
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
    .context-header h2 { margin: 0; font-size: 2rem; font-weight: 700; }
    .context-header p { margin: 1rem 0 0; font-size: 1.1rem; opacity: 0.9; font-weight: 300; }

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
    .badge-poor { background: linear-gradient(45deg, #e74c3c, #8e44ad); color: white; }
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
        From Data to Decisions: Itelligent Traffic Management for CVAG
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
# --------- NEW TAB 2 HELPERS (aggregation-aware) ----------
# =========================

AGG_META = {
    "Hourly":  {"unit": "vph", "bucket": "H", "label": "hour",  "fixed_hours": 1},
    "Daily":   {"unit": "vpd", "bucket": "D", "label": "day",   "fixed_hours": 24},
    "Weekly":  {"unit": "vpw", "bucket": "W", "label": "week",  "fixed_hours": 24*7},
    "Monthly": {"unit": "vpm", "bucket": "M", "label": "month", "fixed_hours": None},  # varies by month
}

def _prep_bucket(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """
    Aggregate hourly records to the selected bucket (sum of hourly volumes).
    Returns: df with columns [local_datetime, intersection_name, total_volume, bucket_hours].
    """
    if df.empty:
        return df.copy()

    g = granularity
    meta = AGG_META[g]
    d = df.copy()
    d["local_datetime"] = pd.to_datetime(d["local_datetime"])

    if g == "Hourly":
        d["bucket"] = d["local_datetime"].dt.floor("H")
    elif g == "Daily":
        d["bucket"] = d["local_datetime"].dt.floor("D")
    elif g == "Weekly":
        d["bucket"] = d["local_datetime"].dt.to_period("W").dt.start_time
    else:  # Monthly
        d["bucket"] = d["local_datetime"].dt.to_period("M").dt.start_time

    agg = (
        d.groupby(["bucket", "intersection_name"], as_index=False)
         .agg(total_volume=("total_volume", "sum"))
         .rename(columns={"bucket": "local_datetime"})
    )

    # Hours in the bucket (for capacity/threshold scaling)
    if g == "Monthly":
        agg["bucket_hours"] = pd.to_datetime(agg["local_datetime"]).dt.days_in_month * 24
    else:
        agg["bucket_hours"] = meta["fixed_hours"]
    return agg

def _cap_series_for_x(x_df: pd.DataFrame, cap_vph: float, high_vph: float) -> pd.DataFrame:
    """Given unique x (local_datetime) and bucket_hours, produce y series for capacity/threshold."""
    xs = x_df[["local_datetime", "bucket_hours"]].drop_duplicates().sort_values("local_datetime")
    xs["capacity"] = xs["bucket_hours"] * float(cap_vph)
    xs["high"] = xs["bucket_hours"] * float(high_vph)
    return xs

def _fmt_period(ts: pd.Timestamp, granularity: str) -> str:
    ts = pd.to_datetime(ts)
    if granularity == "Hourly":
        return ts.strftime("%b %d, %Y %H:%M")
    if granularity == "Daily":
        return ts.strftime("%b %d, %Y")
    if granularity == "Weekly":
        wk = ts.to_period("W")
        return f"Week of {wk.start_time.strftime('%b %d, %Y')}"
    return ts.strftime("%b %Y")

def improved_volume_charts_for_tab2(
    raw_hourly_df: pd.DataFrame,
    granularity: str,
    cap_vph: float,
    high_vph: float,
    top_k: int = 8
):
    """
    Returns (fig_trend, fig_box, fig_matrix)
    - fig_trend: Time series per intersection (lines+markers for non-hourly, lines for hourly)
                 with scaled capacity/high-threshold overlays.
    - fig_box:   Distribution of bucket totals by intersection.
    - fig_matrix: Average bucket total by intersection (compact ranking).
    """
    if raw_hourly_df.empty:
        return None, None, None

    # Aggregate to the selected bucket
    agg = _prep_bucket(raw_hourly_df, granularity)
    if agg.empty:
        return None, None, None

    # Limit to top intersections by mean demand to keep charts readable
    order = agg.groupby("intersection_name")["total_volume"].mean().sort_values(ascending=False)
    keep = order.index[:max(1, min(top_k, len(order)))]

    plot_df = agg[agg["intersection_name"].isin(keep)].copy().sort_values("local_datetime")
    unit = AGG_META[granularity]["unit"]
    label = AGG_META[granularity]["label"]

    # ---------- Trend ----------
    fig_trend = go.Figure()
    mode = "lines" if granularity == "Hourly" else "lines+markers"

    # Choose date format for hover
    xfmt = "%Y-%m-%d %H:%M" if granularity == "Hourly" else "%Y-%m-%d"

    for name, g in plot_df.groupby("intersection_name"):
        fig_trend.add_trace(
            go.Scatter(
                x=g["local_datetime"],
                y=g["total_volume"],
                mode=mode,
                name=name,
                # Escape Plotly tokens inside f-strings with double braces
                hovertemplate=(
                    f"<b>%{{fullData.name}}</b><br>"
                    f"%{{x|{xfmt}}}<br>"
                    f"Volume: %{{y:,.0f}} {unit}<extra></extra>"
                ),
            )
        )

    # Capacity overlays (scaled by hours per bucket)
    xs = _cap_series_for_x(plot_df, cap_vph, high_vph)

    fig_trend.add_trace(
        go.Scatter(
            x=xs["local_datetime"],
            y=xs["capacity"],
            name=f"Theoretical Capacity ({unit})",
            mode="lines",
            line=dict(dash="dash"),
            hovertemplate=(
                f"%{{x|{xfmt}}}<br>"
                f"Capacity: %{{y:,.0f}} {unit}<extra></extra>"
            ),
        )
    )

    fig_trend.add_trace(
        go.Scatter(
            x=xs["local_datetime"],
            y=xs["high"],
            name=f"High Volume Threshold ({unit})",
            mode="lines",
            line=dict(dash="dot"),
            hovertemplate=(
                f"%{{x|{xfmt}}}<br>"
                f"Threshold: %{{y:,.0f}} {unit}<extra></extra>"
            ),
        )
    )

    fig_trend.update_layout(
        xaxis_title="Date/Time",
        yaxis_title=f"Volume ({unit})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    # ---------- Box distribution ----------
    cat_order = order[order.index.isin(keep)].index.tolist()
    fig_box = px.box(
        plot_df, x="intersection_name", y="total_volume",
        category_orders={"intersection_name": cat_order},
        points=False, title=f"Volume Distribution by Intersection — {granularity}"
    )
    fig_box.update_layout(
        xaxis_title="Intersection",
        yaxis_title=f"Volume per {label} ({unit})",
        margin=dict(l=10, r=10, t=40, b=10)
    )

    # ---------- Matrix (compact ranking) ----------
    mat = (
        plot_df.groupby("intersection_name", as_index=False)["total_volume"]
               .mean()
               .rename(columns={"total_volume": f"Avg {label} Volume"})
    )
    mat["Rank"] = mat[f"Avg {label} Volume"].rank(ascending=False, method="dense").astype(int)
    mat = mat.sort_values("Rank")
    fig_matrix = px.bar(
        mat, y="intersection_name", x=f"Avg {label} Volume",
        orientation="h", text=f"Avg {label} Volume",
        title=f"Average {label.capitalize()} Vehicle Volume by Intersection"
    )
    fig_matrix.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
    fig_matrix.update_layout(
        xaxis_title=f"Average {label} volume ({unit})",
        yaxis_title="",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig_trend, fig_box, fig_matrix

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
                    # Use canonical order but only keep nodes present in the data
                    nodes_in_data = _canonical_order_in_data(corridor_df)
                    # Fallback to discovered order if canonical matching yields <2 nodes
                    node_list = nodes_in_data if len(nodes_in_data) >= 2 else _build_node_order(corridor_df)

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
                    # --- BEGIN O-D SUBSET (handles NB and SB robustly) ---
                    working_df = base_df.copy()
                    route_label = "All Segments"

                    # ensure numeric types early to avoid dtype gotchas later
                    for c in ["average_traveltime", "average_delay", "average_speed"]:
                        if c in working_df.columns:
                            working_df[c] = pd.to_numeric(working_df[c], errors="coerce")

                    desired_dir: str | None = None    # ensure scope for later "final guard"
                    path_segments: list[str] = []

                    if od_mode and origin and destination:
                        # Use canonical order (restricted to nodes present)
                        canonical = _canonical_order_in_data(base_df)
                        # Fallback to discovered order if needed
                        if len(canonical) < 2:
                            canonical = _build_node_order(base_df)

                        if origin in canonical and destination in canonical:
                            i0, i1 = canonical.index(origin), canonical.index(destination)

                            if i0 < i1:
                                desired_dir = "nb"
                            elif i0 > i1:
                                desired_dir = "sb"
                            else:
                                desired_dir = None  # same node

                            # Build segment labels in NB orientation (lower index → higher index)
                            imin, imax = (i0, i1) if i0 < i1 else (i1, i0)
                            candidate_segments = [f"{canonical[j]} → {canonical[j + 1]}" for j in range(imin, imax)]

                            # Keep only segments that actually exist in the data
                            seg_names_in_data = set(base_df["segment_name"].dropna().unique().tolist())
                            path_segments = [s for s in candidate_segments if s in seg_names_in_data]

                            if path_segments:
                                seg_df = base_df[base_df["segment_name"].isin(path_segments)].copy()

                                # Filter rows to desired_dir using robust normalizer (avoid NB+SB mix)
                                if "direction" in seg_df.columns and desired_dir is not None:
                                    dnorm = normalize_dir(seg_df["direction"])
                                    seg_df = seg_df.loc[dnorm == desired_dir].copy()

                                working_df = seg_df.copy()
                                route_label = f"{origin} → {destination}"
                            else:
                                st.info("No matching segments found for the selected O-D on the canonical path.")

                    # === DEBUG 1: Show direction counts in working_df (after path & direction filter) ===
                    if "direction" in working_df.columns:
                        try:
                            dir_counts_working = normalize_dir(working_df["direction"]).value_counts(dropna=False).to_dict()
                            st.write("Direction counts in working_df:", dir_counts_working)
                        except Exception:
                            st.write("Direction counts in working_df: <unavailable>")

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

                        # =========================
                        # Build per-hour O-D series (average per segment-hour first, then sum)
                        # =========================
                        od_hourly = process_traffic_data(
                            working_df,
                            date_range,
                            "Hourly",  # force hourly to avoid averaging averages wrongly
                            time_filter,
                            start_hour,
                            end_hour,
                        )

                        if not od_hourly.empty:
                            # Final guard: filter to desired_dir again using robust normalization
                            if "direction" in od_hourly.columns and desired_dir is not None:
                                dnorm2 = normalize_dir(od_hourly["direction"])
                                od_hourly = od_hourly.loc[dnorm2 == desired_dir].copy()

                            # Coerce to numeric BEFORE aggregations
                            for c in ["average_traveltime", "average_delay"]:
                                if c in od_hourly.columns:
                                    od_hourly[c] = pd.to_numeric(od_hourly[c], errors="coerce")

                            # If multiple records exist for same segment & hour, average them first
                            if "segment_name" in od_hourly.columns and "local_datetime" in od_hourly.columns:
                                od_hourly = (
                                    od_hourly.groupby(["local_datetime", "segment_name"], as_index=False)
                                    .agg({"average_traveltime": "mean", "average_delay": "mean"})
                                )

                            # === DEBUG 2: Direction counts in od_hourly (post-guard, pre-sum) ===
                            if "direction" in od_hourly.columns:
                                try:
                                    dir_counts_od = normalize_dir(od_hourly["direction"]).value_counts(dropna=False).to_dict()
                                    st.write("Direction counts in od_hourly (pre-sum):", dir_counts_od)
                                except Exception:
                                    st.write("Direction counts in od_hourly (pre-sum): <unavailable>")

                            # Sum across segments for each hour to form the O-D series
                            od_series = (
                                od_hourly.groupby("local_datetime", as_index=False)
                                .agg({"average_traveltime": "sum", "average_delay": "sum"})
                            )
                            raw_data = od_series.copy()
                        else:
                            od_series = pd.DataFrame()
                            raw_data = filtered_data.copy()

                        # Ensure numeric types for downstream KPIs
                        if not raw_data.empty:
                            for col in ["average_delay", "average_traveltime", "average_speed"]:
                                if col in raw_data.columns:
                                    raw_data[col] = pd.to_numeric(raw_data[col], errors="coerce")

                        # --- END O-D CONSTRUCTION ---

                        if raw_data.empty:
                            st.info("No data in this window.")
                        else:
                            st.subheader("🚦 KPI's (Key Performance Indicators)")
                            k = compute_perf_kpis_interpretable(raw_data, HIGH_DELAY_SEC)

                            # Compute Buffer Time in minutes
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
                                    "📈 Planning Time (95th Percentile)",
                                    f"{k['planning_time']['value']:.1f} {k['planning_time']['unit']}",
                                    help=k['planning_time']['help'],
                                )
                                st.markdown(render_badge(k['planning_time']['score']), unsafe_allow_html=True)
                            with c5:
                                st.metric(
                                    "🧭 Buffer Time (leave this much earlier)",
                                    f"{buffer_minutes:.1f} min",
                                    help=buffer_help,
                                )
                                st.markdown(render_badge(k['buffer_index']['score']), unsafe_allow_html=True)

                        if len(filtered_data) > 1:
                            st.subheader("📈 Performance Trends")
                            v1, v2 = st.columns(2)

                            # Use O-D series for trends if available; otherwise fall back
                            trends_df = od_series if 'od_series' in locals() and not od_series.empty else filtered_data

                            # If user selected Daily/Weekly/Monthly, aggregate the O-D trends to match the selection
                            if 'od_series' in locals() and not od_series.empty and granularity in ("Daily", "Weekly", "Monthly"):
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

                            # Corridor O-D summary table (always hourly)
                            if 'od_series' in locals() and not od_series.empty:
                                st.subheader("🔍Which Dates/Times have the highest Travel Time and Delay?")
                                st.dataframe(
                                    od_series.rename(
                                        columns={
                                            "local_datetime": "Timestamp",
                                            "average_traveltime": "O-D Travel Time (min)",
                                            "average_delay": "O-D Delay (min)",
                                        }
                                    ),
                                    use_container_width=True,
                                )

                        # =========================
                        # 🚨 Comprehensive Bottleneck Analysis (improved)
                        # =========================
                        st.subheader("🚨 Comprehensive Bottleneck Analysis")
                        if 'raw_data' in locals() and not raw_data.empty and "segment_name" in working_df.columns:
                            try:
                                # Filter to analysis window
                                analysis_df = working_df[
                                    (working_df["local_datetime"].dt.date >= date_range[0])
                                    & (working_df["local_datetime"].dt.date <= date_range[1])
                                ].copy()

                                # Normalize direction for clear grouping
                                if "direction" in analysis_df.columns:
                                    analysis_df["dir_norm"] = normalize_dir(analysis_df["direction"])
                                else:
                                    analysis_df["dir_norm"] = "unk"

                                # When O-D mode is active, show only the selected direction to avoid NB/SB duplicates
                                if od_mode and desired_dir is not None:
                                    analysis_df = analysis_df.loc[analysis_df["dir_norm"] == desired_dir].copy()
                                    st.caption(f"Filtered to O-D direction: **{desired_dir.upper()}**")

                                g = analysis_df.groupby(["segment_name", "dir_norm"]).agg(
                                    average_delay_mean=("average_delay", "mean"),
                                    average_delay_max=("average_delay", "max"),
                                    average_traveltime_mean=("average_traveltime", "mean"),
                                    average_traveltime_max=("average_traveltime", "max"),
                                    average_speed_mean=("average_speed", "mean"),
                                    average_speed_min=("average_speed", "min"),
                                    n=("average_delay", "count"),
                                ).reset_index()

                                # Display label with arrow so direction is obvious at a glance
                                arrow_map = {"nb": "↑ NB", "sb": "↓ SB", "unk": "• UNK"}
                                g["Segment (by Dir)"] = g.apply(
                                    lambda r: f"{r['segment_name']} ({arrow_map.get(r['dir_norm'], '• UNK')})", axis=1
                                )

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
                                        "Segment (by Dir)",
                                        "dir_norm",
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
                                        "dir_norm": "Dir",
                                        "average_delay_mean": "Avg Delay (min)",
                                        "average_delay_max": "Peak Delay (min)",
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
                                        ),
                                        "Dir": st.column_config.TextColumn("Dir"),
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
                              Vehicle Volume Analysis: {intersection}
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

                        # ---- Windowed raw hourly data for robust KPI math ----
                        raw = base_df[
                            (base_df["local_datetime"].dt.date >= date_range_vol[0])
                            & (base_df["local_datetime"].dt.date <= date_range_vol[1])
                        ].copy()
                        raw["total_volume"] = pd.to_numeric(raw.get("total_volume", np.nan), errors="coerce")
                        raw["local_datetime"] = pd.to_datetime(raw["local_datetime"])

                        st.subheader("🚦 Traffic Demand Performance Indicators")
                        if raw.empty or raw["total_volume"].dropna().empty:
                            st.info("No raw hourly volume in this window.")
                        else:
                            # ===============================
                            # Aggregation-aware KPI metrics
                            # ===============================
                            # Aggregate across ALL intersections to bucket totals for the selected granularity
                            bucket_all = _prep_bucket(raw, granularity_vol).groupby("local_datetime", as_index=False)["total_volume"].sum().sort_values("local_datetime")
                            if granularity_vol == "Monthly":
                                bucket_all["bucket_hours"] = pd.to_datetime(bucket_all["local_datetime"]).dt.days_in_month * 24
                            else:
                                bucket_all["bucket_hours"] = AGG_META[granularity_vol]["fixed_hours"]

                            # Capacity scaled for each bucket
                            bucket_all["cap"] = bucket_all["bucket_hours"] * THEORETICAL_LINK_CAPACITY_VPH
                            util_series = np.where(bucket_all["cap"] > 0, bucket_all["total_volume"] / bucket_all["cap"] * 100, np.nan)

                            # Peak / p95 / avg on the selected aggregation
                            peak_idx = int(bucket_all["total_volume"].idxmax())
                            peak_val = float(bucket_all.loc[peak_idx, "total_volume"])
                            peak_cap = float(bucket_all.loc[peak_idx, "cap"])
                            peak_util_pct = (peak_val / peak_cap * 100) if peak_cap > 0 else 0.0

                            p95_val = float(np.nanpercentile(bucket_all["total_volume"], 95)) if bucket_all["total_volume"].notna().any() else 0.0
                            avg_bucket_val = float(bucket_all["total_volume"].mean())
                            avg_util_pct = float(np.nanmean(util_series)) if np.isfinite(util_series).any() else 0.0

                            # Hourly CV for stability reference and Bucket CV for aggregation stability
                            hourly_avg = float(np.nanmean(raw["total_volume"])) if raw["total_volume"].notna().any() else 0.0
                            cv_hourly = (float(np.nanstd(raw["total_volume"])) / hourly_avg * 100) if hourly_avg > 0 else 0.0
                            cv_bucket = (float(np.nanstd(bucket_all["total_volume"])) / avg_bucket_val * 100) if avg_bucket_val > 0 else 0.0

                            # Exposure on hourly base (threshold in vph)
                            high_hours = int((raw["total_volume"] > HIGH_VOLUME_THRESHOLD_VPH).sum())
                            total_hours = int(raw["total_volume"].count())
                            risk_pct = (high_hours / total_hours * 100) if total_hours > 0 else 0.0

                            # Units/labels (ADT/AWT/AMT)
                            unit = AGG_META[granularity_vol]["unit"]
                            if granularity_vol == "Hourly":
                                avg_label = "Average Hourly Volume"
                                peak_label = "🔥 Peak Hourly Volume"
                                avg_suffix = "vph"
                            elif granularity_vol == "Daily":
                                avg_label = "Average Daily Traffic (ADT)"
                                peak_label = "🔥 Peak Daily Volume"
                                avg_suffix = "vpd"
                            elif granularity_vol == "Weekly":
                                avg_label = "Average Weekly Traffic (AWT)"
                                peak_label = "🔥 Peak Weekly Volume"
                                avg_suffix = "vpw"
                            else:
                                avg_label = "Average Monthly Traffic (AMT)"
                                peak_label = "🔥 Peak Monthly Volume"
                                avg_suffix = "vpm"

                            # Render KPI cards
                            col1, col2, col3, col4, col5 = st.columns(5)

                            with col1:
                                badge = (
                                    "badge-critical" if peak_util_pct > 90 else
                                    "badge-poor" if peak_util_pct > 75 else
                                    "badge-fair" if peak_util_pct > 60 else
                                    "badge-good"
                                )
                                st.metric(peak_label, f"{peak_val:,.0f} {unit}", delta=f"95th: {p95_val:,.0f} {unit}")
                                st.markdown(
                                    f'<span class="performance-badge {badge}">{peak_util_pct:.0f}% of Capacity</span>',
                                    unsafe_allow_html=True,
                                )

                            with col2:
                                st.metric(
                                    f"📊 {avg_label}",
                                    f"{avg_bucket_val:,.0f} {avg_suffix}",
                                    help=(
                                        "Average traffic on the selected aggregation.\n"
                                        "• ADT = daily average\n• AWT = weekly average\n• AMT = monthly average"
                                    ),
                                )
                                if granularity_vol == "Hourly":
                                    avg_util_pct_hourly = (hourly_avg / THEORETICAL_LINK_CAPACITY_VPH * 100) if THEORETICAL_LINK_CAPACITY_VPH else 0.0
                                    badge2 = "badge-good" if avg_util_pct_hourly <= 40 else ("badge-fair" if avg_util_pct_hourly <= 60 else "badge-poor")
                                    st.markdown(
                                        f'<span class="performance-badge {badge2}">{avg_util_pct_hourly:.0f}% Avg Util</span>',
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    badge2 = "badge-good" if avg_util_pct <= 40 else ("badge-fair" if avg_util_pct <= 60 else "badge-poor")
                                    st.markdown(
                                        f'<span class="performance-badge {badge2}">{avg_util_pct:.0f}% Avg Util</span>',
                                        unsafe_allow_html=True,
                                    )

                            with col3:
                                total_vehicles = float(np.nansum(raw["total_volume"]))
                                st.metric(
                                    "🚗 Total Vehicles (period)",
                                    f"{total_vehicles:,.0f}",
                                    help="Sum of vehicles across the selected time window (computed from hourly records).",
                                )
                                state_badge = (
                                    "badge-good" if total_vehicles < 0.4 * THEORETICAL_LINK_CAPACITY_VPH * 24
                                    else "badge-fair" if total_vehicles < 0.7 * THEORETICAL_LINK_CAPACITY_VPH * 24
                                    else "badge-poor"
                                )
                                st.markdown(
                                    f'<span class="performance-badge {state_badge}">Period Total</span>',
                                    unsafe_allow_html=True,
                                )

                            with col4:
                                st.metric(
                                    "🎯 Demand Consistency",
                                    f"{max(0, 100 - cv_bucket):.0f}%",
                                    delta=f"CV (bucket): {cv_bucket:.1f}%",
                                    help="Higher is steadier. CV calculated on bucket totals for the chosen aggregation."
                                )
                                label_cons = "Consistent" if cv_bucket < 30 else ("Variable" if cv_bucket < 50 else "Highly Variable")
                                badge_cons = "badge-good" if cv_bucket < 30 else ("badge-fair" if cv_bucket < 50 else "badge-poor")
                                st.markdown(
                                    f'<span class="performance-badge {badge_cons}">{label_cons}</span>',
                                    unsafe_allow_html=True,
                                )

                            with col5:
                                st.metric(
                                    "⚠️ High Volume Hours",
                                    f"{high_hours}",
                                    delta=f"{risk_pct:.1f}% of time",
                                    help=f"Hourly records with total_volume > {HIGH_VOLUME_THRESHOLD_VPH:,} vph (always computed on the hourly base).",
                                )
                                level_badge = (
                                    "badge-critical" if risk_pct > 25 else
                                    "badge-poor" if risk_pct > 15 else
                                    "badge-fair" if risk_pct > 5 else
                                    "badge-good"
                                )
                                level = (
                                    "Very High" if risk_pct > 25 else
                                    "High" if risk_pct > 15 else
                                    "Moderate" if risk_pct > 5 else
                                    "Low"
                                )
                                st.markdown(
                                    f'<span class="performance-badge {level_badge}">{level} Risk</span>',
                                    unsafe_allow_html=True,
                                )

                        # ---------------- Charts (optimized for aggregation) ----------------
                        st.subheader("📈 Volume Analysis Visualizations")
                        if len(filtered_volume_data) > 1:
                            try:
                                fig_trend, fig_box, fig_matrix = improved_volume_charts_for_tab2(
                                    raw_hourly_df=raw,
                                    granularity=granularity_vol,
                                    cap_vph=THEORETICAL_LINK_CAPACITY_VPH,
                                    high_vph=HIGH_VOLUME_THRESHOLD_VPH,
                                )
                                if fig_trend:
                                    st.plotly_chart(fig_trend, use_container_width=True)
                                colA, colB = st.columns(2)
                                with colA:
                                    if fig_box:
                                        st.plotly_chart(fig_box, use_container_width=True)
                                with colB:
                                    if fig_matrix:
                                        st.plotly_chart(fig_matrix, use_container_width=True)
                            except Exception as e:
                                st.error(f"❌ Error creating volume charts: {e}")

                        # ---------------- Insights (aggregation-aware) ----------------
                        if not raw.empty:
                            try:
                                # Aggregated per bucket across ALL intersections (sum), for the selected granularity
                                agg_all = _prep_bucket(raw, granularity_vol).groupby("local_datetime", as_index=False)["total_volume"].sum()
                                if agg_all.empty:
                                    raise ValueError("No data in selected window")

                                if granularity_vol == "Monthly":
                                    agg_all["bucket_hours"] = pd.to_datetime(agg_all["local_datetime"]).dt.days_in_month * 24
                                else:
                                    agg_all["bucket_hours"] = AGG_META[granularity_vol]["fixed_hours"]

                                # Capacity/threshold series (same length as agg_all)
                                agg_all["cap"] = agg_all["bucket_hours"] * THEORETICAL_LINK_CAPACITY_VPH
                                agg_all["thr"] = agg_all["bucket_hours"] * HIGH_VOLUME_THRESHOLD_VPH

                                # Peak/avg on the selected aggregation
                                peak_idx = int(agg_all["total_volume"].idxmax())
                                peak_val = float(agg_all.loc[peak_idx, "total_volume"])
                                peak_ts = pd.to_datetime(agg_all.loc[peak_idx, "local_datetime"])
                                avg_val = float(agg_all["total_volume"].mean())
                                p95_val = float(np.nanpercentile(agg_all["total_volume"], 95)) if agg_all["total_volume"].notna().any() else 0.0

                                # Utilizations on the bucket base
                                peak_cap = float(agg_all.loc[peak_idx, "cap"])
                                peak_util_pct = (peak_val / peak_cap * 100) if peak_cap > 0 else 0.0

                                util_series = np.where(agg_all["cap"] > 0, agg_all["total_volume"] / agg_all["cap"], np.nan)
                                p95_util_pct = float(np.nanpercentile(util_series * 100, 95)) if np.isfinite(util_series).any() else 0.0

                                # Shape & consistency (on the selected aggregation)
                                cv_bucket = (float(np.nanstd(agg_all["total_volume"])) / avg_val * 100) if avg_val > 0 else 0.0
                                peak_to_avg = (peak_val / avg_val) if avg_val > 0 else 0.0

                                # Exposure risk on hourly AND bucket bases
                                hourly_over_thr = int((raw["total_volume"] > HIGH_VOLUME_THRESHOLD_VPH).sum())
                                total_hours = int(raw["total_volume"].count())
                                hourly_risk_pct = (hourly_over_thr / total_hours * 100) if total_hours > 0 else 0.0

                                bucket_over_80_cap = int((agg_all["total_volume"] > 0.80 * agg_all["cap"]).sum())
                                bucket_risk_pct = (bucket_over_80_cap / len(agg_all) * 100) if len(agg_all) else 0.0

                                # Top contributors at the peak bucket
                                peak_bucket_all = _prep_bucket(raw, granularity_vol)
                                top_in_peak = (
                                    peak_bucket_all.loc[peak_bucket_all["local_datetime"] == peak_ts]
                                                   .groupby("intersection_name", as_index=False)["total_volume"].sum()
                                                   .sort_values("total_volume", ascending=False)
                                )
                                top3 = top_in_peak.head(3)
                                top3_list = " • ".join([f"{r['intersection_name']}: {int(r['total_volume']):,}" for _, r in top3.iterrows()]) if not top3.empty else "N/A"

                                # Units/labels
                                unit = AGG_META[granularity_vol]["unit"]
                                label = AGG_META[granularity_vol]["label"]
                                peak_when = _fmt_period(peak_ts, granularity_vol)

                                # Smart & simple recommendation
                                if peak_util_pct >= 95 or hourly_risk_pct >= 20:
                                    rec = ("Immediate capacity relief (short-term: retime signals, dynamic splits & queue management; "
                                           "mid-term: turn-lane/approach improvements where feasible; evaluate access control at peak contributors).")
                                    rec_badge = "badge-critical"
                                elif peak_util_pct >= 85 or hourly_risk_pct >= 10 or bucket_risk_pct >= 25:
                                    rec = ("Prioritize signal optimization (AM/PM plans + progression), adjust cycle lengths, and "
                                           "pilot demand management (driveway control, transit signal priority). Plan spot upgrades at top 2-3 intersections.")
                                    rec_badge = "badge-poor"
                                elif peak_util_pct >= 70 or hourly_risk_pct >= 5:
                                    rec = ("Retiming & coordination refresh, monitor weekly trends, and stage TSP/ITS enhancements.")
                                    rec_badge = "badge-fair"
                                else:
                                    rec = ("Monitor; current capacity is adequate with routine timing review.")
                                    rec_badge = "badge-good"

                                # Render
                                st.markdown(
                                    f"""
                                    <div class="insight-box">
                                        <h4>💡 Volume Analysis Insights</h4>
                                        <p><strong>📊 Capacity:</strong> Peak <b>{peak_val:,.0f} {unit}</b> on <b>{peak_when}</b>
                                           ({peak_util_pct:.0f}% of scaled capacity) • 95th percentile <b>{p95_val:,.0f} {unit}</b> ({p95_util_pct:.0f}% of capacity).</p>
                                        <p><strong>🚗 Typical {label.capitalize()} Volume:</strong> Average <b>{avg_val:,.0f} {unit}</b> •
                                           Peak/Avg ratio <b>{peak_to_avg:.1f}×</b> • Consistency <b>{max(0, 100 - cv_bucket):.0f}%</b> (lower CV is steadier).</p>
                                        <p><strong>🧮 Total Vehicles (window):</strong> <b>{float(np.nansum(raw['total_volume'])):,.0f}</b>.</p>
                                        <p><strong>⚠️ Exposure:</strong> Hourly > {HIGH_VOLUME_THRESHOLD_VPH:,} vph for <b>{hourly_over_thr}</b> hours
                                           (<b>{hourly_risk_pct:.1f}%</b> of hours) •
                                           {label.capitalize()}s above 80% of scaled capacity: <b>{bucket_over_80_cap}</b>
                                           (<b>{bucket_risk_pct:.1f}%</b> of {label}s).</p>
                                        <p><strong>📍 Peak Contributors:</strong> {top3_list}</p>
                                        <p><strong>🎯 Recommendation for CVAG:</strong> {rec}</p>
                                        <div style="margin-top:.4rem;">
                                            <span class="performance-badge {rec_badge}">Action Priority</span>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            except Exception as e:
                                st.error(f"❌ Error computing insights: {e}")

                        # ---------------- Risk table ----------------
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
                st.error(f"❌ Error processing traffic data: {e}")
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
      const m = bgColor.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
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
