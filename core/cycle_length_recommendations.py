# Python
import streamlit as st
import pandas as pd


@st.cache_data
def filter_by_period(df: pd.DataFrame, time_col: str, period: str) -> pd.DataFrame:
    """Filter dataframe by time period."""
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


@st.cache_data
def get_hourly_cycle_length(volume):
    """Get CVAG recommended cycle length based on volume."""
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


def render_cycle_length_section(raw: pd.DataFrame, key_prefix: str = "cycle") -> None:
    """Render the 'Cycle Length Recommendations — Hourly Analysis' section."""
    st.subheader("🔁 Cycle Length Recommendations — Hourly Analysis")

    if raw.empty:
        st.info("No hourly volume data available for cycle length recommendations.")
        return

    # Time period selection controls
    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        time_period = st.selectbox(
            "🕐 Time Period",
            ["AM (05:00-10:00)", "MD (11:00-15:00)", "PM (16:00-20:00)", "All Day"],
            index=0,
            help="Select time period for cycle length analysis",
            key=f"{key_prefix}_period",
        )

    with col2:
        current_cycle = st.selectbox(
            "⚙️ Current System Cycle",
            ["140 sec", "130 sec", "120 sec", "110 sec", "Free mode"],
            index=0,
            help="Current signal cycle length to compare against recommendations",
            key=f"{key_prefix}_current",
        )

    with col3:
        st.caption(
            "Using CVAG thresholds: ≥2400vph→140s, ≥1500vph→130s, ≥600vph→120s, ≥300vph→110s, <300vph→Free"
        )

    # Map time period selection to filter function parameter
    period_map = {
        "AM (05:00-10:00)": "AM",
        "MD (11:00-15:00)": "MD",
        "PM (16:00-20:00)": "PM",
        "All Day": "ALL",
    }
    selected_period = period_map[time_period]

    # Filter data by time period
    if selected_period == "ALL":
        period_data = raw.copy()
    else:
        period_data = filter_by_period(raw, "local_datetime", selected_period)

    if period_data.empty:
        st.warning("⚠️ No data available for the selected time period.")
        return

    # Calculate hourly averages
    period_data["hour"] = period_data["local_datetime"].dt.hour
    hourly_analysis = period_data.groupby("hour", as_index=False)["total_volume"].mean()
    hourly_analysis["Volume"] = hourly_analysis["total_volume"].round(0).astype(int)

    # Get cycle recommendations
    hourly_analysis["CVAG Recommendation"] = hourly_analysis["Volume"].apply(get_hourly_cycle_length)

    # Compare with current system
    hourly_analysis["Status"] = hourly_analysis["CVAG Recommendation"].apply(
        lambda rec: _get_status(rec, current_cycle)
    )

    # Summary metrics
    total_hours = len(hourly_analysis)
    optimal_hours = int((hourly_analysis["Status"] == "🟢 OPTIMAL").sum())
    changes_needed = total_hours - optimal_hours
    system_efficiency = (optimal_hours / total_hours * 100) if total_hours > 0 else 0
    avg_volume = int(hourly_analysis["Volume"].mean()) if total_hours > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📅 Hours Analyzed", total_hours)
    with c2:
        st.metric("✅ Optimal Hours", optimal_hours, delta=f"{system_efficiency:.0f}% efficiency")
    with c3:
        st.metric("⚠️ Hours Needing Changes", changes_needed)
    with c4:
        st.metric("📊 Average Volume", f"{avg_volume:,} vph")

    # Display table
    hourly_display = hourly_analysis.copy()
    hourly_display["Hour"] = hourly_display["hour"].apply(lambda x: f"{x:02d}:00")
    display_df = hourly_display[["Hour", "Volume", "CVAG Recommendation", "Status"]].rename(
        columns={"Volume": "Avg Volume (vph)"}
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "Hour": st.column_config.TextColumn("Hour", width="small"),
            "Avg Volume (vph)": st.column_config.NumberColumn("Total Vehicle Volume", format="%d"),
            "CVAG Recommendation": st.column_config.TextColumn("CVAG Recommendation", width="medium"),
            "Status": st.column_config.TextColumn("Status", width="medium"),
        },
    )

    increase_hours = int((hourly_analysis["Status"] == "⬆️ INCREASE").sum())
    reduce_hours = int((hourly_analysis["Status"] == "🔽 REDUCE").sum())
    peak_volume = int(hourly_analysis["Volume"].max())
    peak_hour = int(hourly_analysis.loc[hourly_analysis["Volume"].idxmax(), "hour"])

    st.markdown(
        f"""
        <div class="insight-box">
            <h4>💡 Cycle Length Optimization Insights</h4>
            <p><strong>📊 Current System Efficiency:</strong> {system_efficiency:.0f}% optimal ({optimal_hours}/{total_hours} hours)</p>
            <p><strong>📈 Volume Profile:</strong> Peak {peak_volume:,} vph at {peak_hour:02d}:00 • Average {avg_volume:,} vph during {time_period.lower()}</p>
            <p><strong>🔧 Recommended Actions:</strong> {increase_hours} hours need longer cycles • {reduce_hours} hours need shorter cycles</p>
            <p><strong>🎯 Priority:</strong> {"Focus on peak hour optimization" if system_efficiency < 70 else "Fine-tune existing timing" if system_efficiency < 90 else "System appears well-optimized"}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.download_button(
        "⬇️ Download Cycle Length Analysis (CSV)",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name=f"cycle_length_recommendations_{selected_period.lower()}.csv",
        mime="text/csv",
        key=f"{key_prefix}_download",
    )