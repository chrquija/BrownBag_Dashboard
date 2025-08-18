# Python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import textwrap


# -------------------------
# Time-period filter (keeps your AM/MD/PM thresholds)
# -------------------------
@st.cache_data
def filter_by_period(df: pd.DataFrame, time_col: str, period: str) -> pd.DataFrame:
    """Filter dataframe by time period (AM 05–10, MD 11–15, PM 16–20, ALL)."""
    if time_col not in df.columns:
        return df

    df_copy = df.copy()
    df_copy[time_col] = pd.to_datetime(df_copy[time_col], errors="coerce")

    if period == "AM":
        return df_copy[(df_copy[time_col].dt.hour >= 5) & (df_copy[time_col].dt.hour <= 10)]
    elif period == "MD":
        return df_copy[(df_copy[time_col].dt.hour >= 11) & (df_copy[time_col].dt.hour <= 15)]
    elif period == "PM":
        return df_copy[(df_copy[time_col].dt.hour >= 16) & (df_copy[time_col].dt.hour <= 20)]
    else:
        return df_copy


# -------------------------
# Cycle length thresholds (keeps your cycle thresholds)
# -------------------------
@st.cache_data
def get_hourly_cycle_length(volume):
    """
    Return recommended cycle length:
    ≥2400 → 140 sec, ≥1500 → 130 sec, ≥600 → 120 sec, ≥300 → 110 sec, else Free mode
    """
    if pd.isna(volume) or volume <= 0:
        return "Free mode"
    elif volume >= 2400:
        return "140 sec"
    elif volume >= 1500:
        return "130 sec"
    elif volume >= 600:
        return "120 sec"
    elif volume >= 300:
        return "110 sec"
    else:
        return "Free mode"


def _get_status(recommended: str, current: str) -> str:
    """Compare recommended cycle vs current and return status label."""
    if recommended == current:
        return "🟢 OPTIMAL"
    if recommended == "Free mode" and current != "Free mode":
        return "🔽 REDUCE"
    if recommended != "Free mode" and current == "Free mode":
        return "⬆️ INCREASE"

    rec_val = int(recommended.split()[0]) if recommended != "Free mode" else 0
    cur_val = int(current.split()[0]) if current != "Free mode" else 0
    if rec_val > cur_val:
        return "⬆️ INCREASE"
    if rec_val < cur_val:
        return "🔽 REDUCE"
    return "🟢 OPTIMAL"


# -------------------------
# Visual helpers
# -------------------------
CYCLE_ORDER = ["Free mode", "110 sec", "120 sec", "130 sec", "140 sec"]
CYCLE_COLORS = {
    "Free mode": "#7f8c8d",
    "110 sec": "#27ae60",
    "120 sec": "#3498db",
    "130 sec": "#f39c12",
    "140 sec": "#e74c3c",
}
STATUS_COLORS = {"🟢 OPTIMAL": "#2ecc71", "⬆️ INCREASE": "#e67e22", "🔽 REDUCE": "#8e44ad"}


def _legend_html() -> str:
    """HTML legend for cycle length thresholds."""
    chips = []
    items = [
        ("140 sec", "≥ 2400 vph", CYCLE_COLORS["140 sec"]),
        ("130 sec", "≥ 1500 vph", CYCLE_COLORS["130 sec"]),
        ("120 sec", "≥ 600 vph", CYCLE_COLORS["120 sec"]),
        ("110 sec", "≥ 300 vph", CYCLE_COLORS["110 sec"]),
        ("Free mode", "< 300 vph", CYCLE_COLORS["Free mode"]),
    ]
    for label, cond, color in items:
        chips.append(
            f'<span style="display:inline-flex;align-items:center;margin:.25rem .5rem;padding:.3rem .6rem;'
            f'border-radius:999px;background:{color};color:#fff;font-weight:600;font-size:.85rem;">'
            f'{label}</span><span style="margin-right:1rem;opacity:.85;font-size:.9rem">{cond}</span>'
        )
    return (
        '<div style="border:1px solid rgba(79,172,254,.25);padding:.6rem 1rem;border-radius:12px;'
        'background:linear-gradient(135deg, rgba(79,172,254,.08), rgba(0,242,254,.06));'
        'box-shadow:0 8px 24px rgba(79,172,254,.08);margin-top:.25rem;">'
        '<div style="font-weight:700;margin-bottom:.35rem;color:#1e3c72;">Cycle Length Thresholds</div>'
        + "".join(chips)
        + "</div>"
    )


def _sec_value(label: str) -> int:
    """Map label to numeric seconds for sorting/plotting."""
    return int(label.split()[0]) if label != "Free mode" else 0


# -------------------------
# Main renderer
# -------------------------
def render_cycle_length_section(raw: pd.DataFrame, key_prefix: str = "cycle") -> None:
    """Render the enhanced Cycle Length Recommendations section."""

    if raw.empty:
        st.info("No hourly volume data available for cycle length recommendations.")
        return

    # ---- Context values for header ----
    raw = raw.copy()
    raw["local_datetime"] = pd.to_datetime(raw["local_datetime"], errors="coerce")

    start_dt = raw["local_datetime"].min()
    end_dt = raw["local_datetime"].max()
    start_label = start_dt.strftime("%A, %b %d, %Y") if pd.notnull(start_dt) else "N/A"
    end_label = end_dt.strftime("%A, %b %d, %Y") if pd.notnull(end_dt) else "N/A"

    intersections = sorted(raw["intersection_name"].dropna().unique().tolist()) if "intersection_name" in raw else []
    if len(intersections) == 1:
        intersection_label = intersections[0]
    elif len(intersections) > 1:
        intersection_label = f"{len(intersections)} Intersections"
    else:
        intersection_label = "N/A"

    directions = sorted(raw["direction"].dropna().unique().tolist()) if "direction" in raw else []
    if len(directions) == 1:
        direction_label = directions[0]
    elif len(directions) > 1:
        direction_label = "All Directions"
    else:
        direction_label = "N/A"

    # ---- Header (dedented HTML to avoid markdown code block rendering) ----
    header_html = textwrap.dedent(f"""
    <div style="
        background: linear-gradient(135deg, #2b77e5 0%, #19c3e6 100%);
        border-radius: 16px;
        padding: 22px 24px 20px;
        color: #fff;
        box-shadow: 0 10px 26px rgba(25, 115, 210, .25);
        margin: 8px 0 16px;
        text-align: left;
    ">
      <div style="display:flex; align-items:center; gap:12px;">
        <div style="
            width:40px; height:40px; border-radius:10px;
            background: rgba(255,255,255,.18);
            display:flex; align-items:center; justify-content:center;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.15);
        ">
          <span style="font-size:20px;">🔁</span>
        </div>
        <div style="font-size:2.1rem; font-weight:800; letter-spacing:.2px; line-height:1.1;">
          Cycle Length Recommendations
        </div>
      </div>

      <div style="display:flex; flex-wrap:wrap; gap:8px 10px; margin:12px 0 6px;">
        <span style="
            display:inline-flex; align-items:center; gap:6px;
            padding:6px 10px; border-radius:999px;
            background: rgba(255,255,255,.16);
            font-weight:700; font-size:0.95rem;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.18);
        ">
          <span style="opacity:.9;">Intersection:</span>
          <span style="opacity:1;">{intersection_label}</span>
        </span>

        <span style="
            display:inline-flex; align-items:center; gap:6px;
            padding:6px 10px; border-radius:999px;
            background: rgba(255,255,255,.16);
            font-weight:700; font-size:0.95rem;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.18);
        ">
          <span style="opacity:.9;">Direction:</span>
          <span style="opacity:1;">{direction_label}</span>
        </span>

        <span style="
            display:inline-flex; align-items:center; gap:6px;
            padding:6px 10px; border-radius:999px;
            background: rgba(255,255,255,.16);
            font-weight:700; font-size:0.95rem;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.18);
        ">
          <span style="opacity:.9;">Study Type:</span>
          <span style="opacity:1;">Hourly Analysis</span>
        </span>
      </div>

      <div style="display:flex; align-items:center; gap:8px; margin-top:2px;">
        <span style="
            width:24px; height:24px; border-radius:8px;
            background: rgba(255,255,255,.18);
            display:inline-flex; align-items:center; justify-content:center;
            font-size:13px; box-shadow: inset 0 0 0 1px rgba(255,255,255,.16);
        ">📅</span>
        <span style="font-size:1.05rem; font-weight:600; opacity:.95;">
          {start_label} — {end_label}
        </span>
      </div>
    </div>
    """)
    st.markdown(header_html, unsafe_allow_html=True)

    # Controls
    c1, c2 = st.columns([2, 1.6])
    with c1:
        time_period = st.selectbox(
            "🕐 Time Period",
            ["AM (05:00-10:00)", "MD (11:00-15:00)", "PM (16:00-20:00)", "All Day"],
            index=0,
            help="Analyze AM, Midday, PM, or All Day periods",
            key=f"{key_prefix}_period",
        )
    with c2:
        current_cycle = st.selectbox(
            "⚙️ Current System Cycle",
            CYCLE_ORDER[::-1],  # 140, 130, 120, 110, Free
            index=0,
            help="Cycle used currently; compared against recommendations",
            key=f"{key_prefix}_current",
        )

    # Legend
    st.markdown(_legend_html(), unsafe_allow_html=True)

    # Time period filtering
    period_map = {"AM (05:00-10:00)": "AM", "MD (11:00-15:00)": "MD", "PM (16:00-20:00)": "PM", "All Day": "ALL"}
    selected_period = period_map[time_period]
    period_data = raw if selected_period == "ALL" else filter_by_period(raw, "local_datetime", selected_period)
    if period_data.empty:
        st.warning("⚠️ No data available for the selected time period.")
        return

    # Hour window label for KPIs
    period_windows = {"AM": "05:00–10:00", "MD": "11:00–15:00", "PM": "16:00–20:00", "ALL": "00:00–23:00"}
    hours_window_str = period_windows.get(selected_period, "—")

    # Hourly aggregation
    period_data["hour"] = period_data["local_datetime"].dt.hour
    hourly = period_data.groupby("hour", as_index=False)["total_volume"].mean()
    hourly["Volume"] = hourly["total_volume"].round(0).astype(int)

    # Recommendations + Status
    hourly["CVAG Recommendation"] = hourly["Volume"].apply(get_hourly_cycle_length)
    hourly["Status"] = hourly["CVAG Recommendation"].apply(lambda rec: _get_status(rec, current_cycle))
    hourly["Hour"] = hourly["hour"].apply(lambda x: f"{x:02d}:00")
    hourly["Rec (sec)"] = hourly["CVAG Recommendation"].apply(_sec_value)

    # --- KPI calculations ---
    total_hours = len(hourly)
    optimal_hours = int((hourly["Status"] == "🟢 OPTIMAL").sum())
    changes_needed = total_hours - optimal_hours

    # Lists of hours needing changes (not displayed under KPI)
    inc_hours_list = hourly.loc[hourly["Status"] == "⬆️ INCREASE", "Hour"].tolist()
    red_hours_list = hourly.loc[hourly["Status"] == "🔽 REDUCE", "Hour"].tolist()

    # High-volume threshold KPI (based on raw rows in selected period)
    HIGH_VOLUME_THRESHOLD_VPH = 1200
    period_data["total_volume"] = pd.to_numeric(period_data["total_volume"], errors="coerce")
    total_rows = int(period_data["total_volume"].count())
    high_rows = period_data.loc[period_data["total_volume"] > HIGH_VOLUME_THRESHOLD_VPH]
    high_hours = int(len(high_rows)) if total_rows > 0 else 0
    high_share = (high_hours / total_rows * 100) if total_rows > 0 else 0.0

    # Unique hour-of-day labels that exceeded threshold
    exceed_hour_ids = sorted(high_rows["local_datetime"].dt.hour.unique().tolist()) if len(high_rows) else []
    exceed_hour_labels = [f"{h:02d}:00" for h in exceed_hour_ids]

    # Peak capacity utilization
    INTERSECTION_CAPACITY_VPH = 1800
    peak_volume_pd = float(period_data["total_volume"].max()) if total_rows > 0 else 0.0
    peak_capacity_util = (peak_volume_pd / INTERSECTION_CAPACITY_VPH * 100) if INTERSECTION_CAPACITY_VPH else 0.0

    # Helper to summarize lists
    def _hours_preview(lst, max_items=8):
        if not lst:
            return "None"
        tail = "" if len(lst) <= max_items else f" (+{len(lst)-max_items} more)"
        return ", ".join(lst[:max_items]) + tail

    # KPIs row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("📅 Hours Analyzed", total_hours, delta=hours_window_str)
    with k2:
        system_eff = (optimal_hours / total_hours * 100) if total_hours else 0
        st.metric("✅ Optimal Hours", optimal_hours, delta=f"{system_eff:.0f}% efficiency")
    with k3:
        # Keep counts only (no hour lists here)
        st.metric("🔧 Changes Needed", changes_needed, delta=f"↑ {len(inc_hours_list)} • ↓ {len(red_hours_list)}")
    with k4:
        st.metric("⚠️ Hours Above High-Volume Threshold", f"{high_hours}", delta=f"{high_share:.1f}% of time")
        st.caption(f"Threshold: > {HIGH_VOLUME_THRESHOLD_VPH:,} vph")
        st.caption(f"Hours: {_hours_preview(exceed_hour_labels)}")
    with k5:
        st.metric("🚦 Peak Capacity Utilization", f"{peak_capacity_util:.0f}%", delta=f"Peak {peak_volume_pd:,.0f} vph")
        st.caption(f"Intersection Capacity: {INTERSECTION_CAPACITY_VPH:,} vph")

    # Charts row
    ch1, ch2 = st.columns([2.2, 1.8])

    with ch1:
        # Volume by hour colored by recommended cycle
        fig = px.bar(
            hourly.sort_values("hour"),
            x="Hour",
            y="Volume",
            color="CVAG Recommendation",
            color_discrete_map=CYCLE_COLORS,
            category_orders={"CVAG Recommendation": CYCLE_ORDER, "Hour": [f"{h:02d}:00" for h in range(24)]},
            title="Hourly Volume with Recommended Cycle Length",
            labels={"Volume": "Avg Volume (vph)", "Hour": "Hour of Day"},
        )
        # Overlay markers for status
        fig.add_trace(
            go.Scatter(
                x=hourly["Hour"],
                y=hourly["Volume"],
                mode="markers",
                marker=dict(
                    size=10,
                    color=[STATUS_COLORS[s] for s in hourly["Status"]],
                    line=dict(width=1, color="white"),
                    symbol="diamond",
                ),
                name="Status",
                hovertemplate="Hour=%{x}<br>Volume=%{y:.0f}<extra></extra>",
            )
        )
        fig.update_layout(
            height=420,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        # Pie: Hours by Status
        status_counts = hourly["Status"].value_counts().reindex(["🟢 OPTIMAL", "⬆️ INCREASE", "🔽 REDUCE"], fill_value=0)
        pie = px.pie(
            names=status_counts.index,
            values=status_counts.values,
            title="Hours by Status",
            color=status_counts.index,
            color_discrete_map={
                "🟢 OPTIMAL": STATUS_COLORS["🟢 OPTIMAL"],
                "⬆️ INCREASE": STATUS_COLORS["⬆️ INCREASE"],
                "🔽 REDUCE": STATUS_COLORS["🔽 REDUCE"],
            },
            hole=0.35,
        )
        pie.update_traces(textposition="inside", textinfo="label+percent")
        pie.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(pie, use_container_width=True)

    # Stylized table
    hourly_display = hourly[["Hour", "Volume", "CVAG Recommendation", "Status"]].rename(
        columns={"Volume": "Avg Volume (vph)"}
    )
    st.dataframe(
        hourly_display,
        use_container_width=True,
        column_config={
            "Hour": st.column_config.TextColumn("Hour", width="small"),
            "Avg Volume (vph)": st.column_config.NumberColumn("Avg Volume (vph)", format="%d"),
            "CVAG Recommendation": st.column_config.TextColumn("Cycle Length Recommendation For CVAG", width="medium"),
            "Status": st.column_config.TextColumn("Cycle Length Status", width="medium"),
        },
    )

    # Insights + download
    inc_hours = int((hourly["Status"] == "⬆️ INCREASE").sum())
    red_hours = int((hourly["Status"] == "🔽 REDUCE").sum())
    peak_volume = int(hourly["Volume"].max())
    peak_hour = hourly.loc[hourly["Volume"].idxmax(), "Hour"]

    st.markdown(
        f"""
        <div class="insight-box" style="margin-top:.5rem;">
            <h4>💡 Cycle Length Optimization Insights</h4>
            <p><strong>📊 System Efficiency:</strong> {optimal_hours}/{total_hours} hours optimal ({(optimal_hours/total_hours*100 if total_hours else 0):.0f}%)</p>
            <p><strong>📈 Volume Profile:</strong> Peak {peak_volume:,} vph at {peak_hour} • Threshold exceedance: {high_hours} hours ({high_share:.1f}% of time)</p>
            <p><strong>🔧 Actions:</strong> ↑ {inc_hours} hours need longer cycles • ↓ {red_hours} hours need shorter cycles</p>
            <p><strong>🚦 Capacity:</strong> Peak utilization {peak_capacity_util:.0f}% of intersection capacity ({INTERSECTION_CAPACITY_VPH:,} vph)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.download_button(
        "⬇️ Download Cycle Length Analysis (CSV)",
        data=hourly_display.to_csv(index=False).encode("utf-8"),
        file_name=f"cycle_length_recommendations_{selected_period.lower()}.csv",
        mime="text/csv",
        key=f"{key_prefix}_download",
    )