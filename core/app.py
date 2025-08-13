# Python
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json, difflib
from urllib.request import urlopen

# Plotly (figures are created in helpers; keeping imports is harmless)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Mapping
import pydeck as pdk

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
HIGH_VOLUME_THRESHOLD_VPH    = 1200
CRITICAL_DELAY_SEC           = 120
HIGH_DELAY_SEC               = 60

# Cost/Carbon defaults (all tweakable in UI)
DEFAULT_MINUTES_SAVED = 1.0           # requested “1 minute” default
DEFAULT_VOT_PER_HR    = 22.0          # $/person-hour (Value of Time)
DEFAULT_OCCUPANCY     = 1.25          # persons / vehicle
DEFAULT_SHARE_AFFECT  = 0.35          # fraction of trips affected by the improvement
DEFAULT_VEH_PER_DAY   = 18000         # vehicles per segment-day (if no counts)
DEFAULT_DELAY_BURN_GPH= 0.35          # gal/hour during delay/stop-go
CO2_PER_GALLON_KG     = 8.887         # kg CO2 per gallon gasoline

# Your raw GitHub GeoJSONs (one per segment)
DEFAULT_GEOJSON_URLS = [
    "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/Avenue52_CalleTampico.geojson",
    "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/CalleTampico_VillageShoppingctr.geojson",
    "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/villageshoppingctr_ave50.geojson",
    "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/Avenue50_sagebrushave.geojson",
    "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/sagebrushave_eisenhowerdr.geojson",
    "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/eisenhowerdr_avenue48.geojson",
    "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/avenue48_avenue47.geojson",
    "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/avenue47_pointhappysimon.geojson",
    "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/pointhappysimon_hwy111.geojson",
]
DEFAULT_GEOJSON_URLS_TEXT = "\n".join(DEFAULT_GEOJSON_URLS)


# =========================
# Small helpers
# =========================
def _norm(s: str) -> str:
    """Normalize strings for fuzzy matching."""
    if s is None:
        return ""
    s = str(s).lower()
    ok = []
    for ch in s:
        ok.append(ch if ch.isalnum() else " ")
    return " ".join("".join(ok).split())

def find_coord_cols(df: pd.DataFrame):
    lat_candidates = [c for c in df.columns if c.lower() in ("lat", "latitude", "y")]
    lon_candidates = [c for c in df.columns if c.lower() in ("lon","lng","longitude","x")]
    return (lat_candidates[0] if lat_candidates else None,
            lon_candidates[0] if lon_candidates else None)

def _score_color(score: float) -> list[int]:
    if score is None: return [140, 140, 140]
    if score >= 80:   return [220, 30, 30]   # Critical
    if score >= 60:   return [255, 120, 40]  # Poor
    if score >= 40:   return [240, 200, 40]  # Fair
    if score >= 20:   return [60, 180, 90]   # Good
    return [40, 160, 220]                    # Excellent

@st.cache_data(show_spinner=False)
def load_geojson_from_urls(urls: list[str]) -> dict | None:
    """Merge many GeoJSON files into one FeatureCollection."""
    features = []
    for u in urls:
        try:
            with urlopen(u.strip()) as f:
                gj = json.load(f)
            if gj.get("type") == "FeatureCollection":
                features.extend(gj.get("features", []))
            elif gj.get("type") == "Feature":
                features.append(gj)
        except Exception as e:
            st.warning(f"Couldn’t load: {u} — {e}")
    if not features:
        return None
    # add filename hint for fuzzy joins
    for u, feat in zip(urls, features):
        name = u.split("/")[-1].replace(".geojson","").replace("_"," ")
        feat.setdefault("properties", {})["file_hint"] = name
    return {"type": "FeatureCollection", "features": features}

def center_from_geojson(gj: dict) -> tuple[float, float]:
    lats, lons = [], []
    for f in gj.get("features", []):
        geom = f.get("geometry", {}) or {}
        coords = geom.get("coordinates", [])
        t = geom.get("type")
        if t == "LineString":
            for x, y in coords: lons.append(x); lats.append(y)
        elif t == "MultiLineString":
            for line in coords:
                for x, y in line: lons.append(x); lats.append(y)
        elif t == "Point" and len(coords) == 2:
            x, y = coords
            lons.append(x); lats.append(y)
    if not lats:
        return 33.72, -116.36  # Coachella fallback
    return float(np.mean(lats)), float(np.mean(lons))

def add_scores_to_geojson(gj: dict, table: pd.DataFrame, prop_key: str) -> dict:
    """
    Attach score/label to features by properties[prop_key] if present,
    else try other property names and fuzzy match against table['Segment'].
    """
    if gj is None or table is None or table.empty:
        return gj

    seg_names = table["Segment"].astype(str).tolist() if "Segment" in table.columns else table.iloc[:,0].astype(str).tolist()
    seg_norm  = {_norm(s): s for s in seg_names}
    score_map = {}
    if "Segment" in table.columns:
        key_col = "Segment"
    else:
        key_col = table.columns[0]

    for _, r in table[[key_col, "Bottleneck_Score", "🎯 Performance Rating"]].dropna().iterrows():
        score_map[_norm(r[key_col])] = (float(r["Bottleneck_Score"]), str(r["🎯 Performance Rating"]), str(r[key_col]))

    fallback_keys = [prop_key, "segment_name", "Segment", "name", "segment", "id", "title", "from_to", "file_hint"]

    for f in gj.get("features", []):
        props = f.setdefault("properties", {})
        matched = None
        for k in fallback_keys:
            if k in props and props[k] not in (None, ""):
                cand = _norm(props[k])
                if cand in score_map:
                    matched = cand
                    break

        # If not matched, fuzzy match with available props / file_hint
        if matched is None:
            cands = []
            for k in fallback_keys:
                val = props.get(k)
                if val:
                    cands.append(_norm(val))
            best = None
            for c in cands:
                m = difflib.get_close_matches(c, list(score_map.keys()), n=1, cutoff=0.55)
                if m: best = m[0]; break
            matched = best

        if matched and matched in score_map:
            sc, label, display = score_map[matched]
            props["Bottleneck_Score"] = sc
            props["Rating"] = label
            props["MatchedSegment"] = display
            props["lineColor"] = _score_color(sc)
            props["lineWidth"] = 2 + int(min(8, sc/12))
        else:
            props["Bottleneck_Score"] = None
            props["Rating"] = None
            props["MatchedSegment"] = props.get(prop_key) or props.get("file_hint")
            props["lineColor"] = [140,140,140]
            props["lineWidth"] = 2
    return gj

def build_pydeck_map_from_geojson(gj: dict, join_key: str = "segment_name"):
    lat_c, lon_c = center_from_geojson(gj)
    return pdk.Deck(
        layers=[
            pdk.Layer(
                "GeoJsonLayer",
                gj,
                stroked=True,
                filled=False,
                get_line_color="properties.lineColor",
                get_line_width="properties.lineWidth",
                lineWidthUnits="pixels",
                pickable=True,
                auto_highlight=True,
            )
        ],
        initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=12, pitch=35),
        tooltip={"text": "{properties.MatchedSegment}\nScore: {properties.Bottleneck_Score}\n{properties.Rating}"}
    )

def build_pydeck_map_points(df_pts: pd.DataFrame, lat_col: str, lon_col: str, name_col: str, score_col: str | None):
    if df_pts is None or df_pts.empty: return None
    if score_col and score_col in df_pts.columns:
        df_pts["_color"] = df_pts[score_col].apply(_score_color)
        get_color = "_color"
        radius = 70
    else:
        df_pts["_color"] = [[60,180,90]] * len(df_pts)
        get_color = "_color"
        radius = 60
    return pdk.Deck(
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=df_pts,
                get_position=[lon_col, lat_col],
                get_fill_color=get_color,
                get_radius=radius,
                pickable=True,
                auto_highlight=True,
            )
        ],
        initial_view_state=pdk.ViewState(
            latitude=float(df_pts[lat_col].mean()),
            longitude=float(df_pts[lon_col].mean()),
            zoom=12,
            pitch=30,
        ),
        tooltip={"text": f"{{{name_col}}}\nScore: {{{score_col}}}" if score_col in df_pts.columns else f"{{{name_col}}}"}
    )


def compute_cost_carbon_table(segments: pd.DataFrame,
                              minutes_saved: float,
                              veh_per_day: int,
                              share_affected: float,
                              occupancy: float,
                              vot_per_hr: float,
                              delay_burn_gph: float) -> pd.DataFrame:
    """
    segments: DataFrame with ['Segment','Bottleneck_Score','🎯 Performance Rating'] (optional)
    Returns a table of daily savings per segment for a minutes_saved improvement.
    """
    if segments is None or segments.empty:
        return pd.DataFrame()

    df = segments.copy()
    if "Segment" not in df.columns:
        # Create a fallback column
        df["Segment"] = df.iloc[:,0].astype(str)

    # Vehicles affected per segment per day
    df["Vehicles/day (affected)"] = int(veh_per_day * share_affected)

    # Total vehicle-minutes saved per day
    df["Veh-min saved/day"] = df["Vehicles/day (affected)"] * float(minutes_saved)

    # Person-hours saved and value ($)
    df["Person-hours saved/day"] = (df["Veh-min saved/day"] * occupancy) / 60.0
    df["$ saved/day (time)"] = df["Person-hours saved/day"] * float(vot_per_hr)

    # Fuel + CO2
    gal_per_min = float(delay_burn_gph) / 60.0
    df["Gallons saved/day"] = df["Veh-min saved/day"] * gal_per_min
    df["CO2 saved/day (kg)"] = df["Gallons saved/day"] * CO2_PER_GALLON_KG

    # Labels (impact tiers by CO2 per day)
    def _tier(x):
        if x >= 200: return "🔴 Very High"
        if x >= 80:  return "🟠 High"
        if x >= 30:  return "🟡 Medium"
        if x >= 10:  return "🔵 Low"
        return "🟢 Minimal"
    df["🌱 Carbon Impact Tier"] = df["CO2 saved/day (kg)"].apply(_tier)

    cols = ["Segment", "🎯 Performance Rating", "Bottleneck_Score",
            "Vehicles/day (affected)", "Veh-min saved/day",
            "Person-hours saved/day", "$ saved/day (time)",
            "Gallons saved/day", "CO2 saved/day (kg)", "🌱 Carbon Impact Tier"]
    keep = [c for c in cols if c in df.columns]
    df = df[keep].sort_values("Bottleneck_Score" if "Bottleneck_Score" in df else keep[-1], ascending=False)
    # Pretty formats for Streamlit display will be done via column_config
    return df


# =========================
# CSS (thinner ADVANTEC card; adaptive theme footer handled later)
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
        padding: 1.1rem; /* thinner */
        border-radius: 14px; margin: 0.75rem 0 1.25rem; color: white; text-align: center;
        box-shadow: 0 6px 22px rgba(79, 172, 254, 0.25); backdrop-filter: blur(10px);
    }
    .context-header h2 { margin: 0; font-size: 1.85rem; font-weight: 700; text-shadow: 0 1px 3px rgba(0,0,0,0.25); }
    .context-header p  { margin: .4rem 0 0; font-size: 1.0rem; opacity: 0.92; font-weight: 400; }
    @media (prefers-color-scheme: dark) {
        .context-header { background: linear-gradient(135deg, #2980b9 0%, #3498db 100%); }
    }

    .insight-box {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.15) 0%, rgba(0, 242, 254, 0.15) 100%);
        border-left: 5px solid #4facfe; border-radius: 12px; padding: 1.1rem 1.3rem; margin: 1.1rem 0;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.1);
    }

    .performance-badge { display:inline-block; padding:.35rem .9rem; border-radius:25px; font-size:.85rem;
        font-weight:600; margin:.2rem; border:2px solid transparent; transition: all .3s ease; }
    .badge-excellent { background: linear-gradient(45deg,#2ecc71,#27ae60); color:#fff; }
    .badge-good      { background: linear-gradient(45deg,#3498db,#2980b9); color:#fff; }
    .badge-fair      { background: linear-gradient(45deg,#f39c12,#e67e22); color:#fff; }
    .badge-poor      { background: linear-gradient(45deg,#e74c3c,#c0392b); color:#fff; }
    .badge-critical  { background: linear-gradient(45deg,#e74c3c,#8e44ad); color:#fff; animation: pulse 2s infinite; }
    @keyframes pulse { 0% {opacity:1} 50% {opacity:.7} 100% {opacity:1} }

    .chart-container { background: rgba(79, 172, 254, 0.05); border-radius: 14px; padding: 1rem; margin: 1rem 0;
        border: 1px solid rgba(79, 172, 254, 0.12); }
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
    font-size: 1.02rem; font-weight: 400; color: var(--text-color);
    background: linear-gradient(135deg, rgba(79, 172, 254, 0.1), rgba(0, 242, 254, 0.05));
    padding: .9rem 1.1rem; border-radius: 14px; box-shadow: 0 8px 32px rgba(79,172,254,0.08);
    margin: 1rem 0; line-height: 1.65; border: 1px solid rgba(79,172,254,0.18); backdrop-filter: blur(8px);
">
    <div style="text-align:center; margin-bottom: .25rem;">
        <strong style="font-size: 1.15rem; color: #2980b9;">🚀 The ADVANTEC Platform</strong>
    </div>
    <p>Leverages <strong>millions of data points</strong> trained on advanced Machine Learning algorithms to optimize traffic flow, reduce travel time, minimize fuel consumption, and decrease greenhouse gas emissions across the Coachella Valley transportation network.</p>
    <p><strong>Key Capabilities:</strong> Real-time anomaly detection • Intelligent cycle length optimization • Predictive traffic modeling • Performance analytics</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 1.0rem; border-radius: 14px;
    margin: .9rem 0; text-align: center; box-shadow: 0 6px 20px rgba(52, 152, 219, 0.25);">
    <h3 style="margin:0; font-weight:600;">🔍 Research Question</h3>
    <p style="margin: .35rem 0 0; font-size: 1.0rem;">What are the main bottlenecks (slowest intersections) on Washington St that are most prone to causing increased travel times?</p>
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
        time.sleep(0.4)
        progress_bar.empty()
        status_text.empty()

        with st.sidebar:
            with st.expander("🚧 Performance Analysis Controls", expanded=False):
                seg_options = ["All Segments"] + sorted(corridor_df["segment_name"].dropna().unique().tolist())
                corridor = st.selectbox("🛣️ Select Corridor Segment", seg_options, help="Choose a specific segment or analyze all segments")

                min_date = corridor_df["local_datetime"].dt.date.min()
                max_date = corridor_df["local_datetime"].dt.date.max()

                st.markdown("#### 📅 Analysis Period")
                date_range = date_range_preset_controls(min_date, max_date, key_prefix="perf")

                st.markdown("#### ⏰ Analysis Settings")
                granularity = st.selectbox("Data Aggregation", ["Hourly", "Daily", "Weekly", "Monthly"], index=0)

                time_filter, start_hour, end_hour = None, None, None
                if granularity == "Hourly":
                    time_filter = st.selectbox(
                        "Time Period Focus",
                        ["All Hours", "Peak Hours (7–9 AM, 4–6 PM)", "AM Peak (7–9 AM)", "PM Peak (4–6 PM)", "Off-Peak", "Custom Range"],
                    )
                    if time_filter == "Custom Range":
                        c1, c2 = st.columns(2)
                        with c1: start_hour = st.number_input("Start Hour (0–23)", 0, 23, 7, step=1)
                        with c2: end_hour   = st.number_input("End Hour (1–24)", 1, 24, 18, step=1)

            with st.expander("🗺️ Map data (GeoJSON)", expanded=False):
                geojson_urls_text = st.text_area(
                    "Raw GitHub GeoJSON URLs (one per line)",
                    value=DEFAULT_GEOJSON_URLS_TEXT,
                    height=140,
                    help="Paste additional raw GitHub links here if needed."
                )
                geojson_join_key = st.text_input(
                    "GeoJSON property to join on",
                    value="segment_name",
                    help="We’ll also try common keys & fuzzy match if this doesn’t exist."
                )

            with st.expander("💸 Cost / 🌱 Carbon Assumptions", expanded=False):
                minutes_saved = st.number_input("Minutes saved per trip", 0.1, 15.0, DEFAULT_MINUTES_SAVED, step=0.1)
                vot_per_hr    = st.number_input("Value of time ($/person-hour)", 5.0, 80.0, DEFAULT_VOT_PER_HR, step=1.0)
                occupancy     = st.number_input("Average occupancy (persons/veh)", 1.0, 3.0, DEFAULT_OCCUPANCY, step=0.05)
                share_affected= st.slider("Share of trips affected", 0.0, 1.0, DEFAULT_SHARE_AFFECT, step=0.05)
                veh_per_day   = st.number_input("Vehicles per segment-day (if no counts)", 1000, 60000, DEFAULT_VEH_PER_DAY, step=500)
                delay_burn_gph= st.number_input("Fuel burn during delay (gal/hr)", 0.1, 1.0, DEFAULT_DELAY_BURN_GPH, step=0.05)

        if len(date_range) == 2:
            try:
                base_df = corridor_df.copy()
                if corridor != "All Segments":
                    base_df = base_df[base_df["segment_name"] == corridor]

                if base_df.empty:
                    st.warning("⚠️ No data for the selected segment.")
                else:
                    filtered_data = process_traffic_data(
                        base_df, date_range, granularity,
                        time_filter if granularity == "Hourly" else None,
                        start_hour, end_hour,
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
                                <p>📅 {date_range[0].strftime('%b %d, %Y')} to {date_range[1].strftime('%b %d, %Y')} ({data_span} days) • {granularity} Aggregation{time_context}</p>
                                <p>📈 Analyzing {total_records:,} data points across the selected period</p>
                            </div>
                            """, unsafe_allow_html=True,
                        )

                        # ----- Raw hourly slice for KPIs -----
                        raw_data = base_df[
                            (base_df["local_datetime"].dt.date >= date_range[0]) &
                            (base_df["local_datetime"].dt.date <= date_range[1])
                        ].copy()

                        # =================
                        # KPIs tuned for route choice
                        # =================
                        if raw_data.empty:
                            st.info("No raw hourly data in this window.")
                        else:
                            for col in ["average_delay", "average_traveltime", "average_speed"]:
                                if col in raw_data:
                                    raw_data[col] = pd.to_numeric(raw_data[col], errors="coerce")

                            col1, col2, col3, col4, col5 = st.columns(5)

                            # 1) Reliability Index (already in your app)
                            with col1:
                                if "average_traveltime" in raw_data and raw_data["average_traveltime"].notna().any():
                                    avg_tt = float(np.nanmean(raw_data["average_traveltime"]))
                                    cv_tt  = float(np.nanstd(raw_data["average_traveltime"])) / avg_tt * 100 if avg_tt > 0 else 0
                                    reliability = max(0, 100 - cv_tt)
                                else:
                                    reliability, cv_tt = 0.0, 0.0
                                rel_rating, badge = get_performance_rating(reliability)
                                st.metric("🎯 Reliability Index", f"{reliability:.0f}%", delta=f"CV: {cv_tt:.1f}%")
                                st.markdown(f'<span class="performance-badge {badge}">{rel_rating}</span>', unsafe_allow_html=True)

                            # 2) Congestion Frequency (already in your app)
                            with col2:
                                if "average_delay" in raw_data and raw_data["average_delay"].notna().any():
                                    high_delay_pct = (raw_data["average_delay"] > HIGH_DELAY_SEC).mean() * 100
                                    hours_high = int((raw_data["average_delay"] > HIGH_DELAY_SEC).sum())
                                else:
                                    high_delay_pct, hours_high = 0.0, 0
                                freq_rating, badge = get_performance_rating(100 - high_delay_pct)
                                st.metric("⚠️ Congestion Frequency", f"{high_delay_pct:.1f}%", delta=f"{hours_high} hours")
                                st.markdown(f'<span class="performance-badge {badge}">{freq_rating}</span>', unsafe_allow_html=True)

                            # 3) Typical Travel Time (median)
                            with col3:
                                med_tt = float(np.nanmedian(raw_data["average_traveltime"])) if raw_data["average_traveltime"].notna().any() else 0.0
                                st.metric("🚗 Typical Travel Time", f"{med_tt:.1f} min", help="Median of observed travel time")

                            # 4) 95th Percentile Travel Time
                            with col4:
                                p95_tt = float(np.nanpercentile(raw_data["average_traveltime"].dropna(), 95)) if raw_data["average_traveltime"].notna().any() else 0.0
                                st.metric("⏱️ 95th % Travel Time", f"{p95_tt:.1f} min", help="Upper bound traveler should plan for")

                            # 5) Max Bottleneck Impact Score in window
                            with col5:
                                # We'll compute after we form 'final'; temporarily show placeholder
                                st.metric("🚨 Max Impact Score", "–")

                        # =================
                        # Trend charts
                        # =================
                        if len(filtered_data) > 1:
                            st.subheader("📈 Performance Trends")
                            v1, v2 = st.columns(2)
                            with v1:
                                dc = performance_chart(filtered_data, "delay")
                                if dc: st.plotly_chart(dc, use_container_width=True)
                            with v2:
                                tc = performance_chart(filtered_data, "travel")
                                if tc: st.plotly_chart(tc, use_container_width=True)

                        # =================
                        # Bottleneck table (used also for map & economics)
                        # =================
                        final = pd.DataFrame()
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

                                # Normalize & score
                                def _normcol(s):
                                    s = s.astype(float)
                                    mn, mx = np.nanmin(s), np.nanmax(s)
                                    if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
                                        return (s - mn) / (mx - mn)
                                    return pd.Series(np.zeros(len(s)), index=s.index)

                                score = (0.45 * _normcol(g["average_delay_max"])
                                       + 0.35 * _normcol(g["average_delay_mean"])
                                       + 0.20 * _normcol(g["average_traveltime_max"])) * 100
                                g["Bottleneck_Score"] = score.round(1)

                                bins = [-0.1, 20, 40, 60, 80, 200]
                                labels = ["🟢 Excellent", "🔵 Good", "🟡 Fair", "🟠 Poor", "🔴 Critical"]
                                g["🎯 Performance Rating"] = pd.cut(g["Bottleneck_Score"], bins=bins, labels=labels)

                                final = g[[
                                    "segment_name","direction","🎯 Performance Rating","Bottleneck_Score",
                                    "average_delay_mean","average_delay_max",
                                    "average_traveltime_mean","average_traveltime_max",
                                    "average_speed_mean","average_speed_min","n",
                                ]].rename(columns={
                                    "segment_name":"Segment","direction":"Dir",
                                    "average_delay_mean":"Avg Delay (s)",
                                    "average_delay_max":"Peak Delay (s)",
                                    "average_traveltime_mean":"Avg Time (min)",
                                    "average_traveltime_max":"Peak Time (min)",
                                    "average_speed_mean":"Avg Speed (mph)",
                                    "average_speed_min":"Min Speed (mph)",
                                    "n":"Obs",
                                }).sort_values("Bottleneck_Score", ascending=False)

                                st.subheader("🚨 Comprehensive Bottleneck Analysis")
                                st.dataframe(
                                    final.head(15),
                                    use_container_width=True,
                                    column_config={
                                        "Bottleneck_Score": st.column_config.NumberColumn("🚨 Impact Score", format="%.1f"),
                                        "Avg Delay (s)":    st.column_config.NumberColumn(format="%.1f"),
                                        "Peak Delay (s)":   st.column_config.NumberColumn(format="%.1f"),
                                        "Avg Time (min)":   st.column_config.NumberColumn(format="%.1f"),
                                        "Peak Time (min)":  st.column_config.NumberColumn(format="%.1f"),
                                        "Avg Speed (mph)":  st.column_config.NumberColumn(format="%.1f"),
                                        "Min Speed (mph)":  st.column_config.NumberColumn(format="%.1f"),
                                    },
                                )
                                st.download_button(
                                    "⬇️ Download Bottleneck Table (CSV)",
                                    data=final.to_csv(index=False).encode("utf-8"),
                                    file_name="bottlenecks.csv",
                                    mime="text/csv",
                                )

                                # back-fill KPI #5
                                if not final.empty:
                                    max_score = float(final["Bottleneck_Score"].max())
                                    st.session_state["max_bottleneck_score"] = max_score
                                else:
                                    st.session_state["max_bottleneck_score"] = None

                            except Exception as e:
                                st.error(f"❌ Error in performance analysis: {e}")

                        # Update KPI #5 value if available
                        if "max_bottleneck_score" in st.session_state and st.session_state["max_bottleneck_score"] is not None:
                            st.metric("🚨 Max Impact Score", f"{st.session_state['max_bottleneck_score']:.1f}")

                        # =================
                        # Map & Economics (side-by-side)
                        # =================
                        st.subheader("🗺️ Map & 💸 Cost / 🌱 Carbon Impact (per minute saved)")
                        c1, c2 = st.columns([1.35, 1.0], gap="large")

                        with c1:
                            urls = [u.strip() for u in geojson_urls_text.splitlines() if u.strip()]
                            if urls:
                                gj = load_geojson_from_urls(urls)
                                if gj and not final.empty:
                                    gj = add_scores_to_geojson(gj, final, prop_key=geojson_join_key)
                                if gj:
                                    deck = build_pydeck_map_from_geojson(gj, join_key=geojson_join_key)
                                    st.pydeck_chart(deck, use_container_width=True)
                                    st.caption(f"Lines are color-coded by 🚨 Impact Score and sized by severity. (Join key tried: `{geojson_join_key}` + fallbacks + fuzzy match)")
                                else:
                                    st.warning("No valid features found in the provided GeoJSON URLs.")
                            else:
                                # Fallback: points if user has lat/lon in their dataset
                                lat_col, lon_col = find_coord_cols(base_df)
                                if lat_col and lon_col:
                                    pts = base_df.groupby("segment_name").agg(
                                        Latitude=(lat_col, "mean"),
                                        Longitude=(lon_col, "mean"),
                                        avg_speed=("average_speed", "mean")
                                    ).reset_index().rename(columns={"segment_name":"Segment"})
                                    if not final.empty:
                                        pts = pts.merge(final[["Segment","Bottleneck_Score"]], on="Segment", how="left")
                                    deck = build_pydeck_map_points(pts, "Latitude", "Longitude", "Segment", "Bottleneck_Score")
                                    if deck: st.pydeck_chart(deck, use_container_width=True)
                                else:
                                    st.info("Provide GeoJSON URLs above, or include coordinate columns (lat/lon) to show a map.")

                        with c2:
                            econ_table = compute_cost_carbon_table(
                                segments=final[["Segment","🎯 Performance Rating","Bottleneck_Score"]].copy() if not final.empty else pd.DataFrame(),
                                minutes_saved=minutes_saved,
                                veh_per_day=veh_per_day,
                                share_affected=share_affected,
                                occupancy=occupancy,
                                vot_per_hr=vot_per_hr,
                                delay_burn_gph=delay_burn_gph
                            )
                            if econ_table.empty:
                                st.info("Run analysis first to see cost/carbon benefits per segment.")
                            else:
                                st.dataframe(
                                    econ_table.head(12),
                                    use_container_width=True,
                                    column_config={
                                        "Bottleneck_Score":        st.column_config.NumberColumn("🚨 Impact", format="%.1f"),
                                        "Vehicles/day (affected)": st.column_config.NumberColumn(format="%,d"),
                                        "Veh-min saved/day":       st.column_config.NumberColumn(format="%,.0f"),
                                        "Person-hours saved/day":  st.column_config.NumberColumn(format="%.1f"),
                                        "$ saved/day (time)":      st.column_config.NumberColumn(format="$%,.0f"),
                                        "Gallons saved/day":       st.column_config.NumberColumn(format="%,.1f"),
                                        "CO2 saved/day (kg)":      st.column_config.NumberColumn(format="%,.1f"),
                                    },
                                )
                                st.download_button(
                                    "⬇️ Download Cost/Carbon (CSV)",
                                    data=econ_table.to_csv(index=False).encode("utf-8"),
                                    file_name="cost_carbon_savings.csv",
                                    mime="text/csv",
                                )

                        # =================
                        # Insight card remains
                        # =================
                        if not raw_data.empty:
                            worst_delay = float(np.nanmax(raw_data["average_delay"])) if raw_data["average_delay"].notna().any() else 0.0
                            avg_tt = float(np.nanmean(raw_data["average_traveltime"])) if raw_data["average_traveltime"].notna().any() else 0.0
                            worst_tt = float(np.nanmax(raw_data["average_traveltime"])) if raw_data["average_traveltime"].notna().any() else 0.0
                            tt_delta = ((worst_tt - avg_tt) / avg_tt * 100) if avg_tt > 0 else 0
                            cv_tt = (float(np.nanstd(raw_data["average_traveltime"])) / avg_tt * 100) if avg_tt > 0 else 0
                            reliability = max(0, 100 - cv_tt)
                            high_delay_pct = (raw_data["average_delay"] > HIGH_DELAY_SEC).mean() * 100 if raw_data["average_delay"].notna().any() else 0.0

                            st.markdown(f"""
                            <div class="insight-box">
                                <h4>💡 Advanced Performance Insights</h4>
                                <p><strong>📊 Data Overview:</strong> {len(filtered_data):,} {granularity.lower()} observations across {(date_range[1] - date_range[0]).days + 1} days.</p>
                                <p><strong>🚨 Peaks:</strong> Delay up to {worst_delay:.0f}s ({worst_delay / 60:.1f} min) • Travel time up to {worst_tt:.1f} min (+{tt_delta:.0f}% vs avg).</p>
                                <p><strong>🎯 Reliability:</strong> {reliability:.0f}% travel time reliability • Delays &gt; {HIGH_DELAY_SEC}s occur {high_delay_pct:.1f}% of hours.</p>
                                <p><strong>📌 Action:</strong> {"Critical intervention needed" if worst_delay > CRITICAL_DELAY_SEC else "Optimization recommended" if worst_delay > HIGH_DELAY_SEC else "Monitor trends"}.</p>
                            </div>
                            """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error processing traffic data: {e}")
        else:
            st.warning("⚠️ Please select both start and end dates to proceed.")


# -------------------------
# TAB 2: Volume / Capacity (KPIs unchanged per your request)
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
        time.sleep(0.4)
        progress_bar.empty()
        status_text.empty()

        with st.sidebar:
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
                        st.markdown(
                            f"""
                            <div class="context-header">
                                <h2>📊 Volume Analysis: {intersection}</h2>
                                <p>📅 {date_range_vol[0].strftime('%b %d, %Y')} to {date_range_vol[1].strftime('%b %d, %Y')}
                                ({span} days) • {granularity_vol} Aggregation</p>
                                <p>📈 {len(filtered_volume_data):,} observations • Direction: {direction_filter}</p>
                            </div>
                            """, unsafe_allow_html=True,
                        )

                        raw = base_df[
                            (base_df["local_datetime"].dt.date >= date_range_vol[0]) &
                            (base_df["local_datetime"].dt.date <= date_range_vol[1])
                        ].copy()

                        st.subheader("🚦 Traffic Demand Performance Indicators")
                        if raw.empty:
                            st.info("No raw hourly volume in this window.")
                        else:
                            raw["total_volume"] = pd.to_numeric(raw["total_volume"], errors="coerce")
                            col1, col2, col3, col4, col5 = st.columns(5)

                            with col1:
                                peak = float(np.nanmax(raw["total_volume"])) if raw["total_volume"].notna().any() else 0
                                p95  = float(np.nanpercentile(raw["total_volume"].dropna(), 95)) if raw["total_volume"].notna().any() else 0
                                util = (peak / THEORETICAL_LINK_CAPACITY_VPH) * 100 if THEORETICAL_LINK_CAPACITY_VPH else 0
                                if util > 90: badge = "badge-critical"
                                elif util > 75: badge = "badge-poor"
                                elif util > 60: badge = "badge-fair"
                                else: badge = "badge-good"
                                st.metric("🔥 Peak Demand", f"{peak:,.0f} vph", delta=f"95th: {p95:,.0f}")
                                st.markdown(f'<span class="performance-badge {badge}">{util:.0f}% Capacity</span>', unsafe_allow_html=True)

                            with col2:
                                avg = float(np.nanmean(raw["total_volume"])) if raw["total_volume"].notna().any() else 0
                                med = float(np.nanmedian(raw["total_volume"])) if raw["total_volume"].notna().any() else 0
                                st.metric("📊 Average Demand", f"{avg:,.0f} vph", delta=f"Median: {med:,.0f}")
                                avg_util = (avg / THEORETICAL_LINK_CAPACITY_VPH) * 100 if THEORETICAL_LINK_CAPACITY_VPH else 0
                                badge = "badge-good" if avg_util <= 40 else ("badge-fair" if avg_util <= 60 else "badge-poor")
                                st.markdown(f'<span class="performance-badge {badge}">{avg_util:.0f}% Avg Util</span>', unsafe_allow_html=True)

                            with col3:
                                ratio = (peak / avg) if avg > 0 else 0
                                st.metric("📈 Peak/Average Ratio", f"{ratio:.1f}x", help="Higher ⇒ more peaked demand")
                                badge = "badge-good" if ratio <= 2 else ("badge-fair" if ratio <= 3 else "badge-poor")
                                state = "Low" if ratio <= 2 else ("Moderate" if ratio <= 3 else "High")
                                st.markdown(f'<span class="performance-badge {badge}">{state} Peaking</span>', unsafe_allow_html=True)

                            with col4:
                                cv = (float(np.nanstd(raw["total_volume"])) / avg * 100) if avg > 0 else 0
                                st.metric("🎯 Demand Consistency", f"{max(0, 100 - cv):.0f}%", delta=f"CV: {cv:.1f}%")
                                badge = "badge-good" if cv < 30 else ("badge-fair" if cv < 50 else "badge-poor")
                                label = "Consistent" if cv < 30 else ("Variable" if cv < 50 else "Highly Variable")
                                st.markdown(f'<span class="performance-badge {badge}">{label}</span>', unsafe_allow_html=True)

                            with col5:
                                high_hours = int((raw["total_volume"] > HIGH_VOLUME_THRESHOLD_VPH).sum())
                                total_hours = int(raw["total_volume"].count())
                                risk_pct = (high_hours / total_hours * 100) if total_hours > 0 else 0
                                st.metric("⚠️ High Volume Hours", f"{high_hours}", delta=f"{risk_pct:.1f}% of time")
                                if risk_pct > 25: badge = "badge-critical"
                                elif risk_pct > 15: badge = "badge-poor"
                                elif risk_pct > 5: badge = "badge-fair"
                                else: badge = "badge-good"
                                level = "Very High" if risk_pct > 25 else ("High" if risk_pct > 15 else ("Moderate" if risk_pct > 5 else "Low"))
                                st.markdown(f'<span class="performance-badge {badge}">{level} Risk</span>', unsafe_allow_html=True)

                        st.subheader("📈 Volume Analysis Visualizations")
                        if len(filtered_volume_data) > 1:
                            chart1, chart2, chart3 = volume_charts(filtered_volume_data, THEORETICAL_LINK_CAPACITY_VPH, HIGH_VOLUME_THRESHOLD_VPH)
                            if chart1: st.plotly_chart(chart1, use_container_width=True)
                            colA, colB = st.columns(2)
                            with colA:
                                if chart3: st.plotly_chart(chart3, use_container_width=True)
                            with colB:
                                if chart2: st.plotly_chart(chart2, use_container_width=True)

                        if not raw.empty:
                            peak = float(np.nanmax(raw["total_volume"])) if raw["total_volume"].notna().any() else 0
                            avg = float(np.nanmean(raw["total_volume"])) if raw["total_volume"].notna().any() else 0
                            ratio = (peak / avg) if avg > 0 else 0
                            cv = (float(np.nanstd(raw["total_volume"])) / avg * 100) if avg > 0 else 0
                            util = (peak / THEORETICAL_LINK_CAPACITY_VPH) * 100 if THEORETICAL_LINK_CAPACITY_VPH else 0
                            high_hours = int((raw["total_volume"] > HIGH_VOLUME_THRESHOLD_VPH).sum())
                            total_hours = int(raw["total_volume"].count())
                            risk_pct = (high_hours / total_hours * 100) if total_hours > 0 else 0

                            action = ("Immediate capacity expansion needed" if util > 90
                                      else "Consider signal optimization" if util > 75
                                      else "Monitor trends & optimize timing" if util > 60
                                      else "Current capacity appears adequate")

                            st.markdown(f"""
                            <div class="insight-box">
                                <h4>💡 Advanced Volume Analysis Insights</h4>
                                <p><strong>📊 Capacity:</strong> Peak {peak:,.0f} vph ({util:.0f}% of {THEORETICAL_LINK_CAPACITY_VPH:,} vph) • Avg {avg:,.0f} vph.</p>
                                <p><strong>📈 Demand Shape:</strong> {ratio:.1f}× peak-to-average • Consistency {max(0, 100 - cv):.0f}%.</p>
                                <p><strong>⚠️ Risk:</strong> >{HIGH_VOLUME_THRESHOLD_VPH:,} vph occurs {high_hours} hours ({risk_pct:.1f}% of period).</p>
                                <p><strong>🎯 Recommendation:</strong> {action}.</p>
                            </div>
                            """, unsafe_allow_html=True)

                        st.subheader("🚨 Intersection Volume & Capacity Risk Analysis")
                        try:
                            g = raw.groupby(["intersection_name", "direction"]).agg(
                                total_volume_mean=("total_volume", "mean"),
                                total_volume_max=("total_volume", "max"),
                                total_volume_std=("total_volume", "std"),
                                total_volume_count=("total_volume", "count"),
                            ).reset_index()

                            g["Peak_Capacity_Util"] = (g["total_volume_max"] / THEORETICAL_LINK_CAPACITY_VPH * 100).round(1)
                            g["Avg_Capacity_Util"]  = (g["total_volume_mean"] / THEORETICAL_LINK_CAPACITY_VPH * 100).round(1)
                            g["Volume_Variability"] = (g["total_volume_std"] / g["total_volume_mean"] * 100).replace([np.inf, -np.inf], np.nan).fillna(0).round(1)
                            g["Peak_Avg_Ratio"]     = (g["total_volume_max"] / g["total_volume_mean"]).replace([np.inf, -np.inf], 0).fillna(0).round(1)

                            g["🚨 Risk Score"] = (0.5 * g["Peak_Capacity_Util"] + 0.3 * g["Avg_Capacity_Util"] + 0.2 * (g["Peak_Avg_Ratio"] * 10)).round(1)

                            g["⚠️ Risk Level"] = pd.cut(
                                g["🚨 Risk Score"], bins=[0, 40, 60, 80, 90, 999],
                                labels=["🟢 Low Risk", "🟡 Moderate Risk", "🟠 High Risk", "🔴 Critical Risk", "🚨 Severe Risk"],
                                include_lowest=True,
                            )
                            g["🎯 Action Priority"] = pd.cut(
                                g["Peak_Capacity_Util"], bins=[0, 60, 75, 90, 999],
                                labels=["🟢 Monitor", "🟡 Optimize", "🟠 Upgrade", "🔴 Urgent"], include_lowest=True,
                            )

                            final_v = g[[
                                "intersection_name","direction","⚠️ Risk Level","🎯 Action Priority","🚨 Risk Score",
                                "Peak_Capacity_Util","Avg_Capacity_Util","total_volume_mean","total_volume_max",
                                "Peak_Avg_Ratio","total_volume_count"
                            ]].rename(columns={
                                "intersection_name":"Intersection","direction":"Dir",
                                "Peak_Capacity_Util":"📊 Peak Capacity %","Avg_Capacity_Util":"📊 Avg Capacity %",
                                "total_volume_mean":"Avg Volume (vph)","total_volume_max":"Peak Volume (vph)",
                                "total_volume_count":"Data Points",
                            }).sort_values("🚨 Risk Score", ascending=False)

                            st.dataframe(
                                final_v.head(15),
                                use_container_width=True,
                                column_config={
                                    "🚨 Risk Score": st.column_config.NumberColumn("🚨 Capacity Risk Score", format="%.1f", min_value=0, max_value=120),
                                    "📊 Peak Capacity %": st.column_config.NumberColumn(format="%.1f%%"),
                                    "📊 Avg Capacity %":  st.column_config.NumberColumn(format="%.1f%%"),
                                },
                            )

                            st.download_button(
                                "⬇️ Download Capacity Risk Table (CSV)",
                                data=final_v.to_csv(index=False).encode("utf-8"),
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

                        render_cycle_length_section(raw)

            except Exception as e:
                st.error(f"❌ Error processing volume data: {e}")
                st.info("Please check your data sources and try again.")
        else:
            st.warning("⚠️ Please select both start and end dates to proceed with the volume analysis.")


# =========================
# FOOTER (adaptive to dark mode)
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

  /* force subtitle/copyright white in dark; black-ish in light */
  html.dark .footer-sub, html[data-theme="dark"] .footer-sub, html[data-base-theme="dark"] .footer-sub, body[data-theme="dark"] .footer-sub { color:#ffffff !important; opacity:.95 !important; }
  html.dark .footer-copy, html[data-theme="dark"] .footer-copy, html[data-base-theme="dark"] .footer-copy, body[data-theme="dark"] .footer-copy { color:#ffffff !important; opacity:.95 !important; }
  html.dark .footer-title, html[data-theme="dark"] .footer-title, html[data-base-theme="dark"] .footer-title { color:#7ec3ff !important; }
  html.dark .footer-card, html[data-theme="dark"] .footer-card, html[data-base-theme="dark"] .footer-card { border-color: rgba(79,172,254,0.35) !important; }
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
/* Track Streamlit theme switches and toggle .dark to update colors */
(function(){
  const el = document.documentElement;
  const setDark = () => {
    const isDark =
      el.getAttribute('data-theme') === 'dark' ||
      el.getAttribute('data-base-theme') === 'dark' ||
      document.body.getAttribute('data-theme') === 'dark';
    el.classList.toggle('dark', !!isDark);
  };
  new MutationObserver(setDark).observe(el, { attributes: true, attributeFilter: ['data-theme','data-base-theme'] });
  setDark();
})();
</script>
"""
st.markdown(FOOTER, unsafe_allow_html=True)
