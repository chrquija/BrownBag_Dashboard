# app.py
# ============================================
# Active Transportation & Operations Management Dashboard
# Enhanced for better user experience across all age groups
# ============================================

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
    date_range_preset_controls,
    compute_perf_kpis_interpretable,
    render_badge,
)

# Cycle length section (moved out)
from cycle_length_recommendations import render_cycle_length_section

# Map
from Map import build_corridor_map, build_intersection_map, build_intersections_overview

# =========================
# Page configuration - Mobile-first
# =========================
st.set_page_config(
    page_title="Washington Street Traffic Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",  # Start collapsed for cleaner look
)

# Plotly UI tweaks + default map height
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "toggleSpikelines", "zoom2d", "pan2d"],
    "displayModeBar": False  # Hide toolbar completely for simplicity
}
MAP_HEIGHT = 600  # Bigger maps

# =========================
# Constants / Config - Simplified names
# =========================
CAPACITY_LIMIT = 1800  # vehicles per hour
HIGH_TRAFFIC_THRESHOLD = 1200  # vehicles per hour
DELAY_THRESHOLD_SEC = 60  # seconds

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


# Helper functions - keeping original functionality but simplified
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


# =========================
# Enhanced CSS - Mobile-friendly with accessibility
# =========================
st.markdown("""
<style>
    /* Enhanced mobile-first styles with larger fonts */
    .main {
        font-size: 16px;  /* Larger base font size */
        line-height: 1.6;
    }

    .main-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 15px; padding: 2rem; margin: 1rem 0; color: white;
        box-shadow: 0 8px 32px rgba(30, 60, 114, 0.3);
        text-align: center;
    }
    .main-container h1 {
        font-size: 2.5rem;
        margin: 0 0 0.5rem;
        font-weight: 800;
    }
    .main-container p {
        font-size: 1.2rem;  /* Larger subtitle */
        margin: 0;
        opacity: 0.9;
    }

    /* Quick start guide */
    .quick-start {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(79, 172, 254, 0.3);
    }
    .quick-start h3 {
        margin: 0 0 1rem;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .quick-start ol {
        font-size: 1.1rem;
        margin: 0;
        padding-left: 1.5rem;
    }
    .quick-start li {
        margin: 0.5rem 0;
        line-height: 1.5;
    }

    /* What This Means boxes */
    .insight-simple {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.15) 0%, rgba(0, 242, 254, 0.15) 100%);
        border-left: 5px solid #4facfe;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.1);
    }
    .insight-simple h4 {
        color: #1e3c72;
        margin: 0 0 1rem;
        font-size: 1.4rem;
        font-weight: 600;
    }
    .insight-simple p {
        font-size: 1.1rem;
        line-height: 1.6;
        margin: 0.5rem 0;
    }

    /* Improved badges - larger and more accessible */
    .performance-badge {
        display: inline-block;
        padding: 0.6rem 1.2rem;  /* Bigger touch targets */
        border-radius: 25px;
        font-size: 1rem;  /* Larger text */
        font-weight: 600;
        margin: 0.4rem;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        min-height: 44px;  /* Accessibility guideline */
        line-height: 1.2;
    }
    .performance-badge:hover {
        transform: scale(1.05);
        border-color: rgba(255,255,255,0.25);
    }
    .badge-excellent { background: linear-gradient(45deg, #2ecc71, #27ae60); color: white; }
    .badge-good      { background: linear-gradient(45deg, #3498db, #2980b9); color: white; }
    .badge-fair      { background: linear-gradient(45deg, #f39c12, #e67e22); color: white; }
    .badge-poor      { background: linear-gradient(45deg, #e74c3c, #8e44ad); color: white; }
    .badge-critical  { background: linear-gradient(45deg, #e74c3c, #8e44ad); color: white; animation: pulse 2s infinite; }

    /* Better tabs */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 20px; 
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] { 
        height: 60px; 
        padding: 0 24px; 
        border-radius: 12px;
        background: rgba(79, 172, 254, 0.1); 
        border: 1px solid rgba(79, 172, 254, 0.2);
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* Improved metric cards */
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

    /* Help tooltips with better visibility */
    .help-tooltip {
        background: #e8f4f9;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 1rem;
        line-height: 1.5;
    }

    /* Sticky right rail improvements */
    :root { --cvag-rail-top: 5rem; }

    [data-testid="column"]:has(#od-map-anchor),
    [data-testid="column"]:has(#vol-map-anchor) {
        position: sticky;
        top: var(--cvag-rail-top);
        align-self: flex-start;
        z-index: 1;
    }

    .cvag-map-card {
        background: rgba(79,172,254,0.06);
        border: 1px solid rgba(79,172,254,0.18);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    }

    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-container h1 { font-size: 2rem; }
        .main-container p { font-size: 1.1rem; }
        .performance-badge { 
            padding: 0.5rem 1rem; 
            font-size: 0.9rem; 
            margin: 0.2rem;
        }
        [data-testid="column"]:has(#od-map-anchor),
        [data-testid="column"]:has(#vol-map-anchor) {
            position: static;
            top: auto;
        }
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Title / Intro - Simplified
# =========================
st.markdown("""
<div class="main-container">
    <h1>🚗 Washington Street Traffic Dashboard</h1>
    <p>Simple traffic insights for better commute planning</p>
</div>
""", unsafe_allow_html=True)

# Quick Start Guide (collapsible) - More prominent
with st.expander("👋 **New here? Quick Start Guide**", expanded=False):
    st.markdown("""
    <div class="quick-start">
        <h3>How to use this dashboard:</h3>
        <ol>
            <li><strong>Choose your route:</strong> Pick where you start and end your trip on Washington Street</li>
            <li><strong>Check your commute summary:</strong> See if your route is running smoothly today</li>
            <li><strong>Plan your trip:</strong> Use the "Time to Plan For" to arrive on time</li>
            <li><strong>Explore details:</strong> Use the tabs below for deeper insights about traffic patterns</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# =========================
# NEW TAB 2 HELPERS (keeping original functionality)
# =========================

AGG_META = {
    "Hourly": {"unit": "vehicles/hour", "bucket": "H", "label": "hour", "fixed_hours": 1},
    "Daily": {"unit": "vehicles/day", "bucket": "D", "label": "day", "fixed_hours": 24},
    "Weekly": {"unit": "vehicles/week", "bucket": "W", "label": "week", "fixed_hours": 24 * 7},
    "Monthly": {"unit": "vehicles/month", "bucket": "M", "label": "month", "fixed_hours": None},
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


def improved_volume_charts_for_tab2(
        raw_hourly_df: pd.DataFrame,
        granularity: str,
        cap_vph: float,
        high_vph: float,
        top_k: int = 8
):
    """
    Returns (fig_trend, fig_box, fig_matrix) - keeping original chart functionality
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
                hovertemplate=(
                    f"<b>%{{fullData.name}}</b><br>%{{x|{xfmt}}}<br>Volume: %{{y:,.0f}} {unit}<extra></extra>"
                ),
            )
        )

    # Capacity overlays (scaled by hours per bucket)
    xs = _cap_series_for_x(plot_df, cap_vph, high_vph)

    fig_trend.add_trace(
        go.Scatter(
            x=xs["local_datetime"],
            y=xs["capacity"],
            name=f"Road Capacity ({unit})",
            mode="lines",
            line=dict(dash="dash", color="red"),
            hovertemplate=(f"%{{x|{xfmt}}}<br>Capacity: %{{y:,.0f}} {unit}<extra></extra>"),
        )
    )

    fig_trend.add_trace(
        go.Scatter(
            x=xs["local_datetime"],
            y=xs["high"],
            name=f"High Traffic Level ({unit})",
            mode="lines",
            line=dict(dash="dot", color="orange"),
            hovertemplate=(f"%{{x|{xfmt}}}<br>Threshold: %{{y:,.0f}} {unit}<extra></extra>"),
        )
    )

    fig_trend.update_layout(
        xaxis_title="Date/Time",
        yaxis_title=f"Traffic Volume ({unit})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"Traffic Volume Trends - {granularity} View"
    )

    # ---------- Box distribution ----------
    cat_order = order[order.index.isin(keep)].index.tolist()
    fig_box = px.box(
        plot_df, x="intersection_name", y="total_volume",
        category_orders={"intersection_name": cat_order},
        points=False, title=f"Traffic Volume Distribution by Location — {granularity}"
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
        .rename(columns={"total_volume": f"Average {label.title()} Volume"})
    )
    mat["Rank"] = mat[f"Average {label.title()} Volume"].rank(ascending=False, method="dense").astype(int)
    mat = mat.sort_values("Rank")
    fig_matrix = px.bar(
        mat, y="intersection_name", x=f"Average {label.title()} Volume",
        orientation="h", text=f"Average {label.title()} Volume",
        title=f"Average {label.title()} Traffic Volume by Location"
    )
    fig_matrix.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
    fig_matrix.update_layout(
        xaxis_title=f"Average {label} volume ({unit})",
        yaxis_title="",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig_trend, fig_box, fig_matrix


# =========================
# Simplified Tabs - Better naming
# =========================
st.markdown("## 📊 Choose Your Analysis")
tab1, tab2 = st.tabs(["🚗 Route Performance Analysis", "📈 Traffic Volume Analysis"])

# -------------------------
# TAB 1: Performance / Travel Time - Enhanced UX
# -------------------------
with tab1:
    # Remove progress bars, add loading states instead
    with st.spinner('Loading your route data...'):
        corridor_df = get_corridor_df()

    if corridor_df.empty:
        st.error("⚠️ **Traffic data temporarily unavailable**")
        st.info("Please check your internet connection and refresh the page.")
    else:
        st.success("✅ Route data loaded successfully!")

        # Sidebar controls - Simplified language
        with st.sidebar:
            st.image("Logos/ACE-logo-HiRes.jpg", width=210)
            st.image("Logos/CV Sync__.jpg", width=205)

            with st.expander("⚙️ **Route Settings**", expanded=True):  # Start expanded for better UX
                st.markdown("##### 🗺️ Choose Your Route")
                st.caption("Select where your trip starts and ends")

                # O-D mode (origin → destination) - Better labeling
                od_mode = st.checkbox(
                    "**Analyze Specific Route**",
                    value=True,
                    help="Get detailed insights for your specific commute route from point A to point B",
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
                            origin = st.selectbox("🟢 **Starting From:**", node_list, index=0, key="od_origin")
                        with cB:
                            destination = st.selectbox("🔴 **Going To:**", node_list, index=len(node_list) - 1,
                                                       key="od_destination")
                    else:
                        st.info("Not enough route data available. Please try again later.")

                # Analysis Period - Smart defaults
                min_date = corridor_df["local_datetime"].dt.date.min()
                max_date = corridor_df["local_datetime"].dt.date.max()
                st.markdown("##### 📅 Time Period to Analyze")
                date_range = date_range_preset_controls(min_date, max_date, key_prefix="perf")

                # Advanced options collapsed by default
                with st.expander("🔧 **Advanced Options**", expanded=False):
                    granularity = st.selectbox(
                        "**Data Detail Level**",
                        ["Hourly", "Daily", "Weekly", "Monthly"],
                        index=0,
                        key="granularity_perf",
                        help="Hourly shows the most detail, Daily shows daily patterns, etc.",
                    )

                    time_filter, start_hour, end_hour = None, None, None
                    if granularity == "Hourly":
                        time_filter = st.selectbox(
                            "**Focus on Specific Times**",
                            [
                                "All Times",
                                "Rush Hours (7–9 AM, 4–6 PM)",
                                "Morning Rush (7–9 AM)",
                                "Evening Rush (4–6 PM)",
                                "Non-Rush Hours",
                                "Custom Time Range",
                            ],
                            key="time_period_focus_perf",
                        )
                        if time_filter == "Custom Time Range":
                            c1, c2 = st.columns(2)
                            with c1:
                                start_hour = st.number_input("Start Hour (0–23)", 0, 23, 7, step=1,
                                                             key="start_hour_perf")
                            with c2:
                                end_hour = st.number_input("End Hour (1–24)", 1, 24, 18, step=1, key="end_hour_perf")

        # Main analysis content
        if len(date_range) == 2:
            try:
                base_df = corridor_df.copy()
                if base_df.empty:
                    st.warning("⚠️ No data available for your selected route.")
                else:
                    # Process the route data (keeping original logic)
                    working_df = base_df.copy()
                    route_label = "All Segments"

                    # ensure numeric types early
                    for c in ["average_traveltime", "average_delay", "average_speed"]:
                        if c in working_df.columns:
                            working_df[c] = pd.to_numeric(working_df[c], errors="coerce")

                    desired_dir: str | None = None
                    path_segments: list[str] = []

                    if od_mode and origin and destination:
                        # Use canonical order (keeping original logic)
                        canonical = _canonical_order_in_data(base_df)
                        if len(canonical) < 2:
                            canonical = _build_node_order(base_df)

                        if origin in canonical and destination in canonical:
                            i0, i1 = canonical.index(origin), canonical.index(destination)

                            if i0 < i1:
                                desired_dir = "nb"
                            elif i0 > i1:
                                desired_dir = "sb"
                            else:
                                desired_dir = None

                            imin, imax = (i0, i1) if i0 < i1 else (i1, i0)
                            candidate_segments = [f"{canonical[j]} → {canonical[j + 1]}" for j in range(imin, imax)]

                            seg_names_in_data = set(base_df["segment_name"].dropna().unique().tolist())
                            path_segments = [s for s in candidate_segments if s in seg_names_in_data]

                            if path_segments:
                                seg_df = base_df[base_df["segment_name"].isin(path_segments)].copy()

                                if "direction" in seg_df.columns and desired_dir is not None:
                                    dnorm = normalize_dir(seg_df["direction"])
                                    seg_df = seg_df.loc[dnorm == desired_dir].copy()

                                working_df = seg_df.copy()
                                route_label = f"{origin} → {destination}"
                            else:
                                st.info("No data found for the selected route. Try different locations.")

                    # Layout with larger map
                    main_col_t1, right_col_t1 = st.columns([6, 4], gap="large")  # Give more space to map

                    # Right rail (sticky map) - Enhanced
                    with right_col_t1:
                        st.markdown('<div id="od-map-anchor"></div>', unsafe_allow_html=True)
                        st.markdown("##### 🗺️ Your Route Map")

                        fig_od = None
                        if od_mode and origin and destination and origin != destination:
                            try:
                                fig_od = build_corridor_map(origin, destination)
                                # Update map title to be more user-friendly
                                if fig_od:
                                    fig_od.update_layout(title=f"Corridor: Washington Street")
                            except Exception:
                                fig_od = None

                        if fig_od:
                            try:
                                fig_od.update_layout(height=MAP_HEIGHT, margin=dict(l=0, r=0, t=32, b=0))
                            except Exception:
                                pass
                            st.markdown('<div class="cvag-map-card">', unsafe_allow_html=True)
                            st.plotly_chart(fig_od, use_container_width=True, config=PLOTLY_CONFIG)
                            st.caption(f"📍 **Your Route:** {origin} → {destination}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="cvag-map-card">', unsafe_allow_html=True)
                            st.info("**Select a route** to see the map of your commute.")
                            st.markdown('</div>', unsafe_allow_html=True)

                    # Left/main content - Enhanced presentation
                    with main_col_t1:
                        filtered_data = process_traffic_data(
                            working_df,
                            date_range,
                            granularity,
                            time_filter if granularity == "Hourly" else None,
                            start_hour,
                            end_hour,
                        )

                        if filtered_data.empty:
                            st.warning("⚠️ No traffic data available for your selected filters.")
                            st.info("💡 **Try:** Selecting different dates or a different route")
                        else:
                            total_records = len(filtered_data)
                            data_span = (date_range[1] - date_range[0]).days + 1
                            time_context = f" • {time_filter}" if (granularity == "Hourly" and time_filter) else ""

                            # Enhanced header
                            st.markdown(
                                f"""
                            <div style="
                                background: linear-gradient(135deg, #2b77e5 0%, #19c3e6 100%);
                                border-radius:16px; padding:20px 24px; color:#fff; margin:8px 0 20px;
                                box-shadow:0 10px 26px rgba(25,115,210,.25); text-align:left;">
                              <div style="display:flex; align-items:center; gap:12px;">
                                <div style="width:40px;height:40px;border-radius:10px;background:rgba(255,255,255,.18);
                                            display:flex;align-items:center;justify-content:center;
                                            box-shadow:inset 0 0 0 1px rgba(255,255,255,.15);">🚗</div>
                                <div style="font-size:2rem;font-weight:800;letter-spacing:.2px;">
                                  Your Commute Summary: {route_label}
                                </div>
                              </div>
                              <div style="margin-top:12px;font-size:1.1rem;opacity:.9;">
                                📅 {date_range[0].strftime('%b %d, %Y')} to {date_range[1].strftime('%b %d, %Y')} ({data_span} days){time_context}<br>
                                ✅ Analyzing {total_records:,} traffic records from this period
                              </div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            # Build per-hour O-D series (keeping original logic)
                            od_hourly = process_traffic_data(
                                working_df,
                                date_range,
                                "Hourly",  # force hourly
                                time_filter,
                                start_hour,
                                end_hour,
                            )

                            if not od_hourly.empty:
                                if "direction" in od_hourly.columns and desired_dir is not None:
                                    dnorm2 = normalize_dir(od_hourly["direction"])
                                    od_hourly = od_hourly.loc[dnorm2 == desired_dir].copy()

                                for c in ["average_traveltime", "average_delay"]:
                                    if c in od_hourly.columns:
                                        od_hourly[c] = pd.to_numeric(od_hourly[c], errors="coerce")

                                if "segment_name" in od_hourly.columns and "local_datetime" in od_hourly.columns:
                                    od_hourly = (
                                        od_hourly.groupby(["local_datetime", "segment_name"], as_index=False)
                                        .agg({"average_traveltime": "mean", "average_delay": "mean"})
                                    )

                                od_series = (
                                    od_hourly.groupby("local_datetime", as_index=False)
                                    .agg({"average_traveltime": "sum", "average_delay": "sum"})
                                )
                                raw_data = od_series.copy()
                            else:
                                od_series = pd.DataFrame()
                                raw_data = filtered_data.copy()

                            # Ensure numeric types
                            if not raw_data.empty:
                                for col in ["average_delay", "average_traveltime", "average_speed"]:
                                    if col in raw_data.columns:
                                        raw_data[col] = pd.to_numeric(raw_data[col], errors="coerce")

                            if raw_data.empty:
                                st.info("No traffic data available for this specific route and time period.")
                            else:
                                st.subheader("🚦 Your Commute Health Check")
                                k = compute_perf_kpis_interpretable(raw_data, DELAY_THRESHOLD_SEC)

                                # Enhanced metrics with better explanations
                                buffer_minutes = max(0.0, k["planning_time"]["value"] - k["avg_tt"]["value"])

                                c1, c2, c3, c4, c5 = st.columns(5)
                                with c1:
                                    st.metric(
                                        "🎯 **Route Reliability**",
                                        f"{k['reliability']['value']:.0f}%",
                                        help="How predictable your commute times are. Higher = more consistent.",
                                    )
                                    st.markdown(render_badge(k['reliability']['score']), unsafe_allow_html=True)
                                with c2:
                                    st.metric(
                                        "⚠️ **Delay Frequency**",
                                        f"{k['congestion_freq']['value']:.1f}%",
                                        help="What percentage of time you'll hit significant delays (over 1 minute).",
                                    )
                                    st.caption(k['congestion_freq'].get('extra', ''))
                                    st.markdown(render_badge(k['congestion_freq']['score']), unsafe_allow_html=True)
                                with c3:
                                    st.metric(
                                        "⏱️ **Typical Trip Time**",
                                        f"{k['avg_tt']['value']:.1f} min",
                                        help="Your average commute time on this route.",
                                    )
                                    st.markdown(render_badge(k['avg_tt']['score']), unsafe_allow_html=True)
                                with c4:
                                    st.metric(
                                        "📈 **Time to Plan For**",
                                        f"{k['planning_time']['value']:.1f} min",
                                        help="Leave this much time to arrive on-time 95% of trips.",
                                    )
                                    st.markdown(render_badge(k['planning_time']['score']), unsafe_allow_html=True)
                                with c5:
                                    st.metric(
                                        "🧭 **Extra Buffer Time**",
                                        f"{buffer_minutes:.1f} min",
                                        help="Leave this many extra minutes for important appointments.",
                                    )
                                    st.markdown(render_badge(k['buffer_index']['score']), unsafe_allow_html=True)

                                # What This Means section
                                reliability_text = "very reliable" if k['reliability'][
                                                                          'value'] >= 85 else "fairly reliable" if \
                                k['reliability']['value'] >= 70 else "unreliable"
                                delay_text = "rarely delayed" if k['congestion_freq'][
                                                                     'value'] <= 5 else "sometimes delayed" if \
                                k['congestion_freq']['value'] <= 15 else "frequently delayed"

                                st.markdown(f"""
                                <div class="insight-simple">
                                    <h4>💡 What This Means for Your Commute</h4>
                                    <p><strong>Your {route_label} route is {reliability_text} and {delay_text}.</strong></p>
                                    <p><strong>For everyday trips:</strong> Plan for {k['avg_tt']['value']:.1f} minutes travel time.</p>
                                    <p><strong>For important appointments:</strong> Leave {k['planning_time']['value']:.1f} minutes total 
                                    (that's an extra {buffer_minutes:.1f} minutes buffer) to arrive on time 95% of trips.</p>
                                    {f"<p><strong>⚠️ Heads up:</strong> This route experiences delays {k['congestion_freq']['value']:.1f}% of the time. Consider leaving earlier during rush hours.</p>" if k['congestion_freq']['value'] > 10 else ""}
                                </div>
                                """, unsafe_allow_html=True)

                            # Charts section - keeping all original functionality
                            if len(filtered_data) > 1:
                                st.subheader("📈 Traffic Patterns Over Time")
                                v1, v2 = st.columns(2)

                                trends_df = od_series if 'od_series' in locals() and not od_series.empty else filtered_data

                                # Handle aggregation for different granularities
                                if 'od_series' in locals() and not od_series.empty and granularity in ("Daily",
                                                                                                       "Weekly",
                                                                                                       "Monthly"):
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
                                        dc.update_layout(title="Delay Patterns - When Do You Hit Traffic?")
                                        st.plotly_chart(dc, use_container_width=True, config=PLOTLY_CONFIG)
                                with v2:
                                    tc = performance_chart(trends_df, "travel")
                                    if tc:
                                        tc.update_layout(title="Travel Time Patterns - How Long Does It Take?")
                                        st.plotly_chart(tc, use_container_width=True, config=PLOTLY_CONFIG)

                                # Data table
                                if 'od_series' in locals() and not od_series.empty:
                                    with st.expander("📋 **Detailed Data Table**", expanded=False):
                                        st.subheader("When Are the Worst Times to Travel?")
                                        display_data = od_series.sort_values('average_traveltime',
                                                                             ascending=False).rename(
                                            columns={
                                                "local_datetime": "Date & Time",
                                                "average_traveltime": "Total Trip Time (minutes)",
                                                "average_delay": "Total Delay (minutes)",
                                            }
                                        )
                                        st.dataframe(display_data, use_container_width=True)

                                        # Download button
                                        csv_data = display_data.to_csv(index=False)
                                        st.download_button(
                                            "⬇️ **Download Your Route Data**",
                                            csv_data,
                                            f"route_data_{origin}_to_{destination}.csv",
                                            "text/csv",
                                            help="Download this data to analyze in Excel or other tools"
                                        )

                            # Bottleneck Analysis - keeping original functionality but better presentation
                            with st.expander("🚨 **Detailed Problem Spot Analysis**", expanded=False):
                                if 'raw_data' in locals() and not raw_data.empty and "segment_name" in working_df.columns:
                                    try:
                                        analysis_df = working_df[
                                            (working_df["local_datetime"].dt.date >= date_range[0])
                                            & (working_df["local_datetime"].dt.date <= date_range[1])
                                            ].copy()

                                        if "direction" in analysis_df.columns:
                                            analysis_df["dir_norm"] = normalize_dir(analysis_df["direction"])
                                        else:
                                            analysis_df["dir_norm"] = "unk"

                                        if od_mode and desired_dir is not None:
                                            analysis_df = analysis_df.loc[analysis_df["dir_norm"] == desired_dir].copy()
                                            st.caption(f"Showing data for: **{desired_dir.upper()} direction**")

                                        g = analysis_df.groupby(["segment_name", "dir_norm"]).agg(
                                            average_delay_mean=("average_delay", "mean"),
                                            average_delay_max=("average_delay", "max"),
                                            average_traveltime_mean=("average_traveltime", "mean"),
                                            average_traveltime_max=("average_traveltime", "max"),
                                            average_speed_mean=("average_speed", "mean"),
                                            average_speed_min=("average_speed", "min"),
                                            n=("average_delay", "count"),
                                        ).reset_index()

                                        # Better direction labels
                                        arrow_map = {"nb": "↑ Northbound", "sb": "↓ Southbound",
                                                     "unk": "• Unknown Direction"}
                                        g["Road Segment"] = g.apply(
                                            lambda
                                                r: f"{r['segment_name']} ({arrow_map.get(r['dir_norm'], '• Unknown Direction')})",
                                            axis=1
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
                                        g["Problem_Score"] = score.round(1)

                                        bins = [-0.1, 20, 40, 60, 80, 200]
                                        labels = ["🟢 Flows Well", "🔵 Minor Issues", "🟡 Some Problems", "🟠 Problem Area",
                                                  "🔴 Major Problem"]
                                        g["🚦 Traffic Rating"] = pd.cut(g["Problem_Score"], bins=bins, labels=labels)

                                        final = g[
                                            [
                                                "Road Segment",
                                                "🚦 Traffic Rating",
                                                "Problem_Score",
                                                "average_delay_mean",
                                                "average_delay_max",
                                                "average_traveltime_mean",
                                                "average_traveltime_max",
                                                "average_speed_mean",
                                                "n",
                                            ]
                                        ].rename(
                                            columns={
                                                "Problem_Score": "🚨 Problem Score (0-100)",
                                                "average_delay_mean": "Avg Delay (min)",
                                                "average_delay_max": "Worst Delay (min)",
                                                "average_traveltime_mean": "Avg Time (min)",
                                                "average_traveltime_max": "Longest Time (min)",
                                                "average_speed_mean": "Avg Speed (mph)",
                                                "n": "Data Points",
                                            }
                                        ).sort_values("🚨 Problem Score (0-100)", ascending=False)

                                        st.markdown("**Which parts of your route cause the most problems?**")
                                        st.dataframe(
                                            final.head(10),
                                            use_container_width=True,
                                            column_config={
                                                "🚨 Problem Score (0-100)": st.column_config.NumberColumn(
                                                    "🚨 Problem Score",
                                                    help="Higher score = bigger problem area (combines delay and travel time issues)",
                                                    format="%.1f",
                                                ),
                                            },
                                        )

                                        # Download buttons
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.download_button(
                                                "⬇️ **Download Problem Analysis**",
                                                data=final.to_csv(index=False).encode("utf-8"),
                                                file_name="route_problem_analysis.csv",
                                                mime="text/csv",
                                            )
                                        with col2:
                                            st.download_button(
                                                "⬇️ **Download All Route Data**",
                                                data=filtered_data.to_csv(index=False).encode("utf-8"),
                                                file_name="complete_route_data.csv",
                                                mime="text/csv",
                                            )
                                    except Exception as e:
                                        st.error(f"❌ Error in detailed analysis: {e}")
                                        st.info("Try selecting a different time period or route.")

            except Exception as e:
                st.error(f"❌ Error analyzing your route: {e}")
                st.info("💡 **Try:** Refreshing the page or selecting different options")
        else:
            st.warning("⚠️ Please select both start and end dates to analyze your route.")

# -------------------------
# TAB 2: Volume / Capacity - Enhanced UX
# -------------------------
with tab2:
    with st.spinner('Loading traffic volume data...'):
        volume_df = get_volume_df()

    if volume_df.empty:
        st.error("⚠️ **Traffic volume data temporarily unavailable**")
        st.info("Please check your internet connection and refresh the page.")
    else:
        st.success("✅ Traffic volume data loaded successfully!")

        with st.sidebar:
            with st.expander("⚙️ **Volume Analysis Settings**", expanded=True):
                st.markdown("##### 🚦 Choose Location to Analyze")
                st.caption("Select a specific intersection or view all locations")

                intersections = ["All Locations"] + sorted(
                    volume_df["intersection_name"].dropna().unique().tolist()
                )
                intersection = st.selectbox(
                    "Location:",
                    intersections,
                    key="intersection_vol",
                    help="Choose a specific intersection to focus your analysis"
                )

                min_date = volume_df["local_datetime"].dt.date.min()
                max_date = volume_df["local_datetime"].dt.date.max()

                st.markdown("##### 📅 Time Period to Analyze")
                date_range_vol = date_range_preset_controls(min_date, max_date, key_prefix="vol")

                # Advanced options collapsed
                with st.expander("🔧 **Advanced Options**", expanded=False):
                    granularity_vol = st.selectbox(
                        "**Data Detail Level**",
                        ["Hourly", "Daily", "Weekly", "Monthly"],
                        index=0,
                        key="granularity_vol",
                        help="Hourly shows the most detail, Daily shows daily patterns, etc."
                    )

                    direction_options = ["All Directions"] + sorted(volume_df["direction"].dropna().unique().tolist())
                    direction_filter = st.selectbox(
                        "**Traffic Direction**",
                        direction_options,
                        key="direction_filter_vol",
                        help="Filter to northbound, southbound, or see all traffic"
                    )

        if len(date_range_vol) == 2:
            try:
                base_df = volume_df.copy()
                if intersection != "All Locations":
                    base_df = base_df[base_df["intersection_name"] == intersection]
                if direction_filter != "All Directions":
                    base_df = base_df[base_df["direction"] == direction_filter]

                # Two-column layout with bigger map
                content_col, right_col = st.columns([6, 4], gap="large")

                # Right rail (sticky overview map) - Enhanced
                with right_col:
                    st.markdown('<div id="vol-map-anchor"></div>', unsafe_allow_html=True)
                    st.markdown("##### 🗺️ Corridor Overview")

                    try:
                        fig_over = build_intersections_overview(
                            selected_label=None if intersection == "All Locations" else intersection
                        )
                        if fig_over:
                            fig_over.update_layout(title="Corridor: Washington Street")
                    except Exception:
                        fig_over = None

                    if fig_over:
                        try:
                            fig_over.update_layout(height=MAP_HEIGHT, margin=dict(l=0, r=0, t=32, b=0))
                        except Exception:
                            pass
                        st.markdown('<div class="cvag-map-card">', unsafe_allow_html=True)
                        st.plotly_chart(fig_over, use_container_width=True, config=PLOTLY_CONFIG)
                        if intersection != "All Locations":
                            st.caption(f"📍 **Analyzing:** {intersection}")
                        else:
                            st.caption("📍 **Analyzing:** All corridor intersections")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="cvag-map-card">', unsafe_allow_html=True)
                        st.caption("Map temporarily unavailable.")
                        st.markdown('</div>', unsafe_allow_html=True)

                # Main analysis content - Enhanced
                with content_col:
                    if base_df.empty:
                        st.warning("⚠️ No traffic volume data for your selected filters.")
                        st.info("💡 **Try:** Selecting different dates or location")
                    else:
                        filtered_volume_data = process_traffic_data(base_df, date_range_vol, granularity_vol)

                        if filtered_volume_data.empty:
                            st.warning("⚠️ No volume data available for the selected time period.")
                        else:
                            span = (date_range_vol[1] - date_range_vol[0]).days + 1
                            total_obs = len(filtered_volume_data)

                            # Enhanced header
                            st.markdown(
                                f"""
                            <div style="
                                background: linear-gradient(135deg, #2b77e5 0%, #19c3e6 100%);
                                border-radius:16px; padding:20px 24px; color:#fff; margin:8px 0 20px;
                                box-shadow:0 10px 26px rgba(25,115,210,.25); text-align:left;">
                              <div style="display:flex; align-items:center; gap:12px;">
                                <div style="width:40px;height:40px;border-radius:10px;background:rgba(255,255,255,.18);
                                            display:flex;align-items:center;justify-content:center;
                                            box-shadow:inset 0 0 0 1px rgba(255,255,255,.15);">📊</div>
                                <div style="font-size:2rem;font-weight:800;letter-spacing:.2px;">
                                  Traffic Volume Analysis: {intersection}
                                </div>
                              </div>
                              <div style="margin-top:12px;font-size:1.1rem;opacity:.9;">
                                📅 {date_range_vol[0].strftime('%b %d, %Y')} to {date_range_vol[1].strftime('%b %d, %Y')} ({span} days)<br>
                                ✅ {total_obs:,} traffic measurements • Direction: {direction_filter}
                              </div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            # Raw data processing (keeping original logic)
                            raw = base_df[
                                (base_df["local_datetime"].dt.date >= date_range_vol[0])
                                & (base_df["local_datetime"].dt.date <= date_range_vol[1])
                                ].copy()
                            raw["total_volume"] = pd.to_numeric(raw.get("total_volume", np.nan), errors="coerce")
                            raw["local_datetime"] = pd.to_datetime(raw["local_datetime"])

                            st.subheader("🚦 Traffic Volume Health Check")
                            if raw.empty or raw["total_volume"].dropna().empty:
                                st.info("No traffic volume data available for this period.")
                            else:
                                # Calculate metrics (keeping original logic)
                                bucket_all = \
                                _prep_bucket(raw, granularity_vol).groupby("local_datetime", as_index=False)[
                                    "total_volume"].sum().sort_values("local_datetime")
                                if granularity_vol == "Monthly":
                                    bucket_all["bucket_hours"] = pd.to_datetime(
                                        bucket_all["local_datetime"]).dt.days_in_month * 24
                                else:
                                    bucket_all["bucket_hours"] = AGG_META[granularity_vol]["fixed_hours"]

                                bucket_all["cap"] = bucket_all["bucket_hours"] * CAPACITY_LIMIT
                                util_series = np.where(bucket_all["cap"] > 0,
                                                       bucket_all["total_volume"] / bucket_all["cap"] * 100, np.nan)

                                peak_idx = int(bucket_all["total_volume"].idxmax())
                                peak_val = float(bucket_all.loc[peak_idx, "total_volume"])
                                peak_cap = float(bucket_all.loc[peak_idx, "cap"])
                                peak_util_pct = (peak_val / peak_cap * 100) if peak_cap > 0 else 0.0

                                p95_val = float(np.nanpercentile(bucket_all["total_volume"], 95)) if bucket_all[
                                    "total_volume"].notna().any() else 0.0
                                avg_bucket_val = float(bucket_all["total_volume"].mean())

                                hourly_avg = float(np.nanmean(raw["total_volume"])) if raw[
                                    "total_volume"].notna().any() else 0.0
                                cv_bucket = (float(np.nanstd(
                                    bucket_all["total_volume"])) / avg_bucket_val * 100) if avg_bucket_val > 0 else 0.0

                                high_hours = int((raw["total_volume"] > HIGH_TRAFFIC_THRESHOLD).sum())
                                total_hours = int(raw["total_volume"].count())
                                risk_pct = (high_hours / total_hours * 100) if total_hours > 0 else 0.0

                                unit = AGG_META[granularity_vol]["unit"]

                                # Simplified labels
                                if granularity_vol == "Hourly":
                                    avg_label = "Average Hourly Traffic"
                                    peak_label = "🔥 Peak Hour Traffic"
                                elif granularity_vol == "Daily":
                                    avg_label = "Average Daily Traffic"
                                    peak_label = "🔥 Busiest Day Traffic"
                                elif granularity_vol == "Weekly":
                                    avg_label = "Average Weekly Traffic"
                                    peak_label = "🔥 Busiest Week Traffic"
                                else:
                                    avg_label = "Average Monthly Traffic"
                                    peak_label = "🔥 Busiest Month Traffic"

                                # Enhanced metrics display
                                col1, col2, col3, col4, col5 = st.columns(5)

                                with col1:
                                    # Traffic light color system
                                    if peak_util_pct > 90:
                                        badge_color = "badge-critical"
                                        status_text = "🔴 Very Busy"
                                    elif peak_util_pct > 75:
                                        badge_color = "badge-poor"
                                        status_text = "🟠 Busy"
                                    elif peak_util_pct > 60:
                                        badge_color = "badge-fair"
                                        status_text = "🟡 Moderate"
                                    else:
                                        badge_color = "badge-good"
                                        status_text = "🟢 Light Traffic"

                                    st.metric(peak_label, f"{peak_val:,.0f}", delta=f"95th: {p95_val:,.0f}")
                                    st.markdown(
                                        f'<span class="performance-badge {badge_color}">{status_text}</span>',
                                        unsafe_allow_html=True,
                                    )

                                with col2:
                                    st.metric(
                                        f"📊 **{avg_label}**",
                                        f"{avg_bucket_val:,.0f}",
                                        help=f"Average number of vehicles per {granularity_vol.lower()} period",
                                    )

                                    # Capacity utilization
                                    if granularity_vol == "Hourly":
                                        avg_util_pct = (hourly_avg / CAPACITY_LIMIT * 100) if CAPACITY_LIMIT else 0.0
                                    else:
                                        avg_util_pct = float(np.nanmean(util_series)) if np.isfinite(
                                            util_series).any() else 0.0

                                    if avg_util_pct <= 40:
                                        badge2 = "badge-good"
                                        util_text = "🟢 Light"
                                    elif avg_util_pct <= 60:
                                        badge2 = "badge-fair"
                                        util_text = "🟡 Moderate"
                                    else:
                                        badge2 = "badge-poor"
                                        util_text = "🟠 Heavy"

                                    st.markdown(
                                        f'<span class="performance-badge {badge2}">{util_text}</span>',
                                        unsafe_allow_html=True,
                                    )

                                with col3:
                                    total_vehicles = float(np.nansum(raw["total_volume"]))
                                    st.metric(
                                        "🚗 **Total Vehicles**",
                                        f"{total_vehicles:,.0f}",
                                        help=f"Total vehicles counted during this {span}-day period",
                                    )

                                    # Period assessment
                                    daily_avg = total_vehicles / span if span > 0 else 0
                                    if daily_avg < 8000:
                                        period_badge = "badge-good"
                                        period_text = "🟢 Quiet Period"
                                    elif daily_avg < 15000:
                                        period_badge = "badge-fair"
                                        period_text = "🟡 Normal Period"
                                    else:
                                        period_badge = "badge-poor"
                                        period_text = "🟠 Busy Period"

                                    st.markdown(
                                        f'<span class="performance-badge {period_badge}">{period_text}</span>',
                                        unsafe_allow_html=True,
                                    )

                                with col4:
                                    consistency = max(0, 100 - cv_bucket)
                                    st.metric(
                                        "🎯 **Traffic Consistency**",
                                        f"{consistency:.0f}%",
                                        delta=f"Variation: {cv_bucket:.1f}%",
                                        help="Higher = more predictable traffic patterns"
                                    )

                                    if cv_bucket < 30:
                                        cons_badge = "badge-good"
                                        cons_text = "🟢 Very Consistent"
                                    elif cv_bucket < 50:
                                        cons_badge = "badge-fair"
                                        cons_text = "🟡 Somewhat Variable"
                                    else:
                                        cons_badge = "badge-poor"
                                        cons_text = "🟠 Highly Variable"

                                    st.markdown(
                                        f'<span class="performance-badge {cons_badge}">{cons_text}</span>',
                                        unsafe_allow_html=True,
                                    )

                                with col5:
                                    st.metric(
                                        "⚠️ **Heavy Traffic Hours**",
                                        f"{high_hours}",
                                        delta=f"{risk_pct:.1f}% of time",
                                        help=f"Hours with over {HIGH_TRAFFIC_THRESHOLD:,} vehicles/hour",
                                    )

                                    if risk_pct > 25:
                                        risk_badge = "badge-critical"
                                        risk_text = "🔴 Very High"
                                    elif risk_pct > 15:
                                        risk_badge = "badge-poor"
                                        risk_text = "🟠 High"
                                    elif risk_pct > 5:
                                        risk_badge = "badge-fair"
                                        risk_text = "🟡 Moderate"
                                    else:
                                        risk_badge = "badge-good"
                                        risk_text = "🟢 Low"

                                    st.markdown(
                                        f'<span class="performance-badge {risk_badge}">{risk_text} Risk</span>',
                                        unsafe_allow_html=True,
                                    )

                            # Charts section (keeping original functionality)
                            st.subheader("📈 Traffic Volume Patterns")
                            if len(filtered_volume_data) > 1:
                                try:
                                    fig_trend, fig_box, fig_matrix = improved_volume_charts_for_tab2(
                                        raw_hourly_df=raw,
                                        granularity=granularity_vol,
                                        cap_vph=CAPACITY_LIMIT,
                                        high_vph=HIGH_TRAFFIC_THRESHOLD,
                                    )
                                    if fig_trend:
                                        st.plotly_chart(fig_trend, use_container_width=True, config=PLOTLY_CONFIG)
                                    colA, colB = st.columns(2)
                                    with colA:
                                        if fig_box:
                                            st.plotly_chart(fig_box, use_container_width=True, config=PLOTLY_CONFIG)
                                    with colB:
                                        if fig_matrix:
                                            st.plotly_chart(fig_matrix, use_container_width=True, config=PLOTLY_CONFIG)
                                except Exception as e:
                                    st.error(f"❌ Error creating charts: {e}")

                            # What This Means section for Volume
                            if not raw.empty and avg_bucket_val > 0:
                                # Generate insights based on data
                                if peak_util_pct >= 85:
                                    capacity_insight = "⚠️ **High capacity usage detected.** This location experiences very heavy traffic during peak times."
                                    recommendation = "🎯 **Recommendation:** Monitor this location closely and consider traffic management improvements."
                                elif peak_util_pct >= 60:
                                    capacity_insight = "🟡 **Moderate capacity usage.** Traffic levels are manageable but could become problematic during events."
                                    recommendation = "🎯 **Recommendation:** Continue monitoring and have contingency plans ready."
                                else:
                                    capacity_insight = "✅ **Good capacity levels.** Traffic flows smoothly with room for growth."
                                    recommendation = "🎯 **Recommendation:** Current infrastructure appears adequate for traffic demands."

                                period_label = AGG_META[granularity_vol]["label"]

                                st.markdown(f"""
                                <div class="insight-simple">
                                    <h4>💡 What This Means for {intersection}</h4>
                                    <p><strong>Traffic Level:</strong> This location sees an average of {avg_bucket_val:,.0f} vehicles per {period_label}, 
                                    with peak periods reaching {peak_val:,.0f} vehicles.</p>
                                    <p>{capacity_insight}</p>
                                    <p><strong>Consistency:</strong> Traffic patterns are {
                                "very consistent" if cv_bucket < 30 else
                                "moderately consistent" if cv_bucket < 50 else
                                "highly variable"
                                } (variation of {cv_bucket:.1f}%).</p>
                                    <p>{recommendation}</p>
                                </div>
                                """, unsafe_allow_html=True)

                            # Advanced analysis in expander
                            with st.expander("🚨 **Detailed Capacity Risk Analysis**", expanded=False):
                                try:
                                    g = raw.groupby(["intersection_name", "direction"]).agg(
                                        total_volume_mean=("total_volume", "mean"),
                                        total_volume_max=("total_volume", "max"),
                                        total_volume_std=("total_volume", "std"),
                                        total_volume_count=("total_volume", "count"),
                                    ).reset_index()

                                    g["Peak_Capacity_Util"] = (
                                            g["total_volume_max"] / CAPACITY_LIMIT * 100
                                    ).round(1)
                                    g["Avg_Capacity_Util"] = (
                                            g["total_volume_mean"] / CAPACITY_LIMIT * 100
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

                                    # Better risk level labels
                                    g["⚠️ Traffic Risk Level"] = pd.cut(
                                        g["🚨 Risk Score"],
                                        bins=[0, 40, 60, 80, 90, 999],
                                        labels=["🟢 Low Risk", "🟡 Watch Closely", "🟠 Problem Area", "🔴 High Risk",
                                                "🚨 Critical"],
                                        include_lowest=True,
                                    )
                                    g["🎯 Recommended Action"] = pd.cut(
                                        g["Peak_Capacity_Util"],
                                        bins=[0, 60, 75, 90, 999],
                                        labels=["🟢 Monitor", "🟡 Optimize Timing", "🟠 Infrastructure Upgrade",
                                                "🔴 Urgent Action"],
                                        include_lowest=True,
                                    )

                                    final = g[
                                        [
                                            "intersection_name",
                                            "direction",
                                            "⚠️ Traffic Risk Level",
                                            "🎯 Recommended Action",
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
                                            "intersection_name": "Location",
                                            "direction": "Direction",
                                            "Peak_Capacity_Util": "📊 Peak Capacity Usage %",
                                            "Avg_Capacity_Util": "📊 Average Capacity Usage %",
                                            "total_volume_mean": "Average Traffic (vehicles/hour)",
                                            "total_volume_max": "Peak Traffic (vehicles/hour)",
                                            "total_volume_count": "Data Points Available",
                                        }
                                    ).sort_values("🚨 Risk Score", ascending=False)

                                    st.markdown("**Which locations need the most attention?**")
                                    st.dataframe(
                                        final.head(15),
                                        use_container_width=True,
                                        column_config={
                                            "🚨 Risk Score": st.column_config.NumberColumn(
                                                "🚨 Risk Score",
                                                help="Combined score based on peak usage, average usage, and traffic variability",
                                                format="%.1f",
                                                min_value=0,
                                                max_value=120,
                                            ),
                                            "📊 Peak Capacity Usage %": st.column_config.NumberColumn("📊 Peak Usage %",
                                                                                                     format="%.1f%%"),
                                            "📊 Average Capacity Usage %": st.column_config.NumberColumn(
                                                "📊 Average Usage %", format="%.1f%%"),
                                        },
                                    )

                                    # Download buttons
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.download_button(
                                            "⬇️ **Download Risk Analysis**",
                                            data=final.to_csv(index=False).encode("utf-8"),
                                            file_name="traffic_capacity_risk_analysis.csv",
                                            mime="text/csv",
                                        )
                                    with col2:
                                        st.download_button(
                                            "⬇️ **Download Volume Data**",
                                            data=filtered_volume_data.to_csv(index=False).encode("utf-8"),
                                            file_name="traffic_volume_data.csv",
                                            mime="text/csv",
                                        )
                                except Exception as e:
                                    st.error(f"❌ Error in risk analysis: {e}")

                            # Cycle Length Recommendations (keeping original functionality)
                            if not raw.empty:
                                render_cycle_length_section(raw)

            except Exception as e:
                st.error(f"❌ Error processing traffic volume data: {e}")
                st.info("💡 **Try:** Refreshing the page or selecting different options")
        else:
            st.warning("⚠️ Please select both start and end dates to proceed with the analysis.")

# =========================
# FOOTER - Simplified and more accessible
# =========================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 1rem; color: #666; font-size: 1rem; line-height: 1.6;">
    <h4 style="color: #2980b9; margin: 0 0 1rem; font-size: 1.4rem;">🚗 Washington Street Traffic Dashboard</h4>
    <p style="margin: 0.5rem 0; font-size: 1.1rem; font-weight: 500;">
        Real-time traffic intelligence • Better commute planning • Data-driven insights
    </p>
    <div style="display: flex; justify-content: center; align-items: center; gap: 2rem; margin: 1.5rem 0; flex-wrap: wrap;">
        <a href="https://www.instagram.com/advantec98/" target="_blank" style="text-decoration: none; color: #2980b9; font-weight: 600;">
            📸 Instagram
        </a>
        <a href="https://www.linkedin.com/company/advantec-consulting-engineers-inc/" target="_blank" style="text-decoration: none; color: #2980b9; font-weight: 600;">
            💼 LinkedIn
        </a>
        <a href="https://www.facebook.com/advantecconsultingUSA" target="_blank" style="text-decoration: none; color: #2980b9; font-weight: 600;">
            📘 Facebook
        </a>
        <a href="https://advantec-usa.com/" target="_blank" style="text-decoration: none; color: #2980b9; font-weight: 600; padding: 0.5rem 1rem; border: 2px solid #2980b9; border-radius: 25px;">
            🌐 Website
        </a>
    </div>
    <p style="margin: 1rem 0 0; font-size: 0.95rem;">
        © 2025 ADVANTEC Consulting Engineers, Inc. — "Because We Care"
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# Mobile Responsiveness Enhancements
# =========================
st.markdown("""
<script>
function enhanceMobileExperience() {
    // Detect mobile devices
    const isMobile = window.innerWidth < 768;

    if (isMobile) {
        // Enhance touch targets
        const buttons = document.querySelectorAll('button');
        buttons.forEach(btn => {
            btn.style.minHeight = '44px';
            btn.style.padding = '12px 16px';
        });

        // Improve text readability
        const metrics = document.querySelectorAll('[data-testid="metric-container"]');
        metrics.forEach(metric => {
            metric.style.padding = '1rem';
            metric.style.fontSize = '1.1rem';
        });

        // Optimize sidebar for mobile
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.style.fontSize = '1rem';
        }
    }
}

// Run on load and resize
window.addEventListener('load', enhanceMobileExperience);
window.addEventListener('resize', enhanceMobileExperience);

// Add loading states for better UX
document.addEventListener('DOMContentLoaded', function() {
    const charts = document.querySelectorAll('.js-plotly-plot');
    charts.forEach(chart => {
        chart.style.transition = 'opacity 0.3s ease';
    });
});
</script>
""", unsafe_allow_html=True)