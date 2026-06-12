import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from io import BytesIO
import datetime
from concurrent.futures import ThreadPoolExecutor

@st.cache_data
def load_single_day_metric(daily_config, date, metric_column, approach_filter=None):
    """
    Loads a single daily file and extracts a specific metric.
    Cached at the daily level for performance.
    Returns (value, successful_url)
    """
    base_url = daily_config["base_url"].strip()
    file_pattern = daily_config["file_pattern"].strip()
    date_format = daily_config["date_format"].strip()
    
    date_str = date.strftime(date_format)
    primary_filename = file_pattern.replace("{date}", date_str)
    
    # Try multiple common patterns if primary fails (e.g. 1_ prefix for daily exports)
    urls_to_try = [base_url + primary_filename]
    for i in range(1, 11): # Try 1_ to 10_
        urls_to_try.append(base_url + f"{i}_{primary_filename}")

    # Fallback metric columns
    metric_fallbacks = {
        "Average Delay (s)": ["Average Delay (s)", "Average Delay (sec)", "Average Delay", "Delay (s)", "Delay Range 1", "Avg Control Delay (seconds)"],
        "Arrivals on Green (%)": ["Arrivals on Green (%)", "Arrivals On Green (%)", "AOG (%)", "AOG", "Arrivals On Green Range 1"],
        "Split Failures (%)": ["Split Failures (%)", "Split Failure (%)", "SF (%)", "Split Failures Range 1"],
        "Vehicle Samples 1": ["Vehicle Samples 1", "Total Vehicles", "Volume", "Total Volume", "Turning Movement Range 1"]
    }
    
    target_columns = metric_fallbacks.get(metric_column, [metric_column])

    for url in urls_to_try:
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                xls = pd.ExcelFile(BytesIO(response.content))
                
                sheet_name = "By Approach" if (approach_filter and approach_filter != "All Approaches") else "Intersection"
                try:
                    df = pd.read_excel(xls, sheet_name)
                except ValueError:
                    # Sheet not found, try common aliases
                    alt_sheets = ["By Approach", "Approach", "Approaches"] if sheet_name == "By Approach" else ["Intersection", "Overall"]
                    for alt in alt_sheets:
                        try:
                            df = pd.read_excel(xls, alt)
                            break
                        except ValueError:
                            continue
                    else:
                        continue # No valid sheet found in this file

                if approach_filter and approach_filter != "All Approaches":
                    df["Approach"] = df["Approach"].astype(str).str.strip()
                    
                    # Try exact match first
                    match = df[df["Approach"] == approach_filter]
                    
                    # If no match, try common abbreviations
                    if match.empty:
                        abbr_map = {
                            "Northbound": ["N", "NB"],
                            "Southbound": ["S", "SB"],
                            "Eastbound": ["E", "EB"],
                            "Westbound": ["W", "WB"]
                        }
                        abbrs = abbr_map.get(approach_filter, [])
                        if abbrs:
                            match = df[df["Approach"].isin(abbrs)]
                    
                    if not match.empty:
                        # Find the first available metric column from fallbacks (case-insensitive)
                        col_map = {c.lower(): c for c in match.columns}
                        val = None
                        for col in target_columns:
                            if col.lower() in col_map:
                                val = match[col_map[col.lower()]].iloc[0]
                                break
                    else:
                        val = None
                else:
                    # Find the first available metric column from fallbacks (case-insensitive)
                    col_map = {c.lower(): c for c in df.columns}
                    val = None
                    for col in target_columns:
                        if col.lower() in col_map:
                            val = df[col_map[col.lower()]].iloc[0]
                            break
                
                return val, url
        except Exception:
            continue
            
    return None, None

@st.cache_data
def load_trend_data_preset(registry, trend_selection):
    """
    Loads daily metrics using preset windows and years.
    Returns a list of data points.
    """
    selected_intersections = trend_selection["intersections"]
    window_name = trend_selection["window"]
    year_selector = trend_selection["years"]
    metric_column = trend_selection["metric_column"]
    approach = trend_selection["approach"]
    
    years_to_load = []
    if year_selector == "Both years":
        years_to_load = ["2026", "2025"]
    else:
        years_to_load = [year_selector]
        
    tasks = []
    for int_label in selected_intersections:
        intersection = next(i for i in registry if i["label"] == int_label)
        daily_config = intersection["daily_data"]
        window_dates_by_year = daily_config["comparison_windows"].get(window_name, {})
        
        for year in years_to_load:
            dates_str = window_dates_by_year.get(year, [])
            for date_str in dates_str:
                d = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                tasks.append({
                    "int_label": int_label,
                    "year": year,
                    "date": d,
                    "config": daily_config,
                    "metric": metric_column,
                    "approach": approach,
                    "years_to_load": years_to_load,
                    "selected_intersections": selected_intersections
                })

    def fetch_task(t):
        val, successful_url = load_single_day_metric(t["config"], t["date"], t["metric"], t["approach"])
        
        day_name = t["date"].strftime("%A")
        day_mapping = {
            "Wednesday": "Day 1: Wednesday",
            "Thursday": "Day 2: Thursday",
            "Friday": "Day 3: Friday",
            "Saturday": "Day 4: Saturday",
            "Sunday": "Day 5: Sunday",
            "Monday": "Day 6: Monday",
            "Tuesday": "Day 7: Tuesday"
        }
        fest_day = day_mapping.get(day_name, day_name)
        
        # Determine legend label
        if len(t["years_to_load"]) > 1:
            legend_label = f"{t['int_label']} - {t['year']}"
        else:
            legend_label = t['int_label']

        if val is not None:
            return {
                "Festival Day": fest_day,
                "Date": t["date"],
                "Value": val,
                "Legend": legend_label,
                "Intersection": t["int_label"],
                "Year": t["year"],
                "Approach": t["approach"],
                "URL": successful_url,
                "Source Sheet": "By Approach" if (t["approach"] and t["approach"] != "All Approaches") else "Intersection",
                "Metric Column": t["metric"]
            }
        else:
            # Re-construct the primary URL for reporting
            base_url = t["config"]["base_url"].strip()
            file_pattern = t["config"]["file_pattern"].strip()
            date_format = t["config"]["date_format"].strip()
            date_str = t["date"].strftime(date_format)
            url = base_url + file_pattern.replace("{date}", date_str)
            return {"Missing": True, "Date": t["date"], "Year": t["year"], "Intersection": t["int_label"], "URL": url}

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_task, tasks))
    
    return results

def render_trend_comparison_section(registry, direction_map, direction_colors, trend_selection, show_debug=True, show_comparison_table=True, show_date_labels=True, show_data_labels=True, show_shading=False):
    st.write("### Multi-Intersection Trend Comparison")
    
    if not trend_selection:
        st.info("Please select 'Trend Analysis' in the sidebar to configure this view.")
        return

    if not trend_selection["intersections"]:
        st.warning("Please select at least one intersection in the sidebar.")
        return

    with st.spinner("Loading trend data..."):
        results = load_trend_data_preset(registry, trend_selection)
    
    data_points = [r for r in results if r and "Missing" not in r]
    missing_points = [r for r in results if r and "Missing" in r]
    
    for m in missing_points:
        st.warning(f"⚠️ **Missing data:** {m['Intersection']} ({m['Year']}-{m['Date'].strftime('%m-%d')}). Tried: {m['URL'].split('/')[-1]} (and prefixes). Please verify the file exists on GitHub.")

    if not data_points:
        st.error("No usable daily files found for the selected criteria.")
        return

    df_trend = pd.DataFrame(data_points)
    
    # Count unique lines for layout decisions
    num_lines = df_trend["Legend"].nunique()
    
    # Custom sort order for X-axis
    day_order = [
        "Day 1: Wednesday", "Day 2: Thursday", "Day 3: Friday", 
        "Day 4: Saturday", "Day 5: Sunday", "Day 6: Monday", "Day 7: Tuesday"
    ]
    df_trend["Festival Day"] = pd.Categorical(df_trend["Festival Day"], categories=day_order, ordered=True)
    df_trend = df_trend.sort_values(["Legend", "Festival Day"])

    # Normalize percentages
    selected_metric_label = trend_selection["metric_label"]
    is_pct = "%" in selected_metric_label
    if is_pct:
        if df_trend["Value"].max() > 1:
            df_trend["Value"] = df_trend["Value"] / 100.0

    # Prepare metadata for tooltips and labels
    df_trend["DateStr"] = df_trend["Date"].apply(lambda x: x.strftime('%Y-%m-%d'))
    df_trend["ApproachTooltip"] = df_trend["Approach"].apply(lambda x: "E, W, N, S" if x == "All Approaches" else x)
    
    approach_display = "E, W, N, S" if trend_selection["approach"] == "All Approaches" else trend_selection["approach"]

    # Calculate date range for the title
    min_date = df_trend["Date"].min()
    max_date = df_trend["Date"].max()
    if min_date == max_date:
        date_range_str = min_date.strftime("%m/%d/%Y")
    else:
        date_range_str = f"{min_date.strftime('%m/%d/%Y')} - {max_date.strftime('%m/%d/%Y')}"

    # Decide on chart type: Bar for 1-2 points, Line for 3+
    unique_days = df_trend["Festival Day"].nunique()
    use_bar = unique_days <= 2

    if use_bar:
        fig = px.bar(
            df_trend,
            x="Festival Day",
            y="Value",
            color="Legend",
            barmode="group",
            text_auto=(".0f" if not is_pct else ".1%") if show_data_labels else False,
            title=f"{approach_display} {selected_metric_label} per Day<br>{date_range_str} ({trend_selection['window']})",
            labels={"Value": selected_metric_label, "Festival Day": "Festival Relative Day"}
        )
        # Simulation of "3D" effect with borders and shadows (via marker properties)
        fig.update_traces(
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5,
            marker_opacity=0.9
        )
    else:
        # Prepare text labels for line chart
        if show_date_labels and show_data_labels:
            df_trend["ChartText"] = df_trend.apply(lambda r: f"{r['DateStr']}<br>{r['Value']:.1%}" if is_pct else f"{r['DateStr']}<br>{r['Value']:.0f}", axis=1)
        elif show_date_labels:
            df_trend["ChartText"] = df_trend["DateStr"]
        elif show_data_labels:
            df_trend["ChartText"] = df_trend["Value"].apply(lambda x: f"{x:.1%}" if is_pct else f"{x:.0f}")
        else:
            df_trend["ChartText"] = None

        # Smart overlap detection for labels
        df_trend["TextPos"] = "top center"
        y_max = df_trend["Value"].max()
        y_min = df_trend["Value"].min()
        y_range = y_max - y_min if y_max != y_min else 1.0
        # 5% of range as threshold for overlap
        threshold = y_range * 0.05
        
        for day in df_trend["Festival Day"].unique():
            day_indices = df_trend[df_trend["Festival Day"] == day].index.tolist()
            if len(day_indices) > 1:
                # Sort indices by value to find close neighbors
                day_indices_sorted = sorted(day_indices, key=lambda idx: df_trend.at[idx, "Value"])
                for i in range(len(day_indices_sorted) - 1):
                    idx_lower = day_indices_sorted[i]
                    idx_upper = day_indices_sorted[i+1]
                    if (df_trend.at[idx_upper, "Value"] - df_trend.at[idx_lower, "Value"]) < threshold:
                        # Push labels away from each other
                        df_trend.at[idx_lower, "TextPos"] = "bottom center"
                        # Upper one remains "top center" (already set)

        fig = px.line(
            df_trend, 
            x="Festival Day", 
            y="Value", 
            color="Legend",
            text="ChartText",
            markers=True,
            title=f"{approach_display} {selected_metric_label} per Day<br>{date_range_str} ({trend_selection['window']})",
            labels={"Value": selected_metric_label, "Festival Day": "Festival Relative Day"}
        )
        
        # Apply the computed text positions per trace
        for trace in fig.data:
            trace_df = df_trend[df_trend["Legend"] == trace.name].sort_values("Festival Day")
            trace.textposition = trace_df["TextPos"].tolist()
    
    # Dynamic legend and margin based on the number of lines
    if num_lines > 5:
        # Side legend for many lines to avoid "blob" at bottom
        legend_config = dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=12, color="black"),
            borderwidth=1,
            bordercolor="rgba(0,0,0,0.2)",
            title=dict(text="<b>Intersections</b>", font=dict(size=13)),
            itemsizing='constant'
        )
        chart_margins = dict(t=100, b=80, l=50, r=160)
    else:
        # Bottom legend for fewer lines
        legend_config = dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=14, color="black"),
            borderwidth=1,
            bordercolor="rgba(0,0,0,0.2)",
            title=None,
            itemsizing='constant'
        )
        chart_margins = dict(t=100, b=100, l=50, r=30)

    fig.update_layout(
        font=dict(family="Arial, sans-serif", color="black"),
        hoverlabel=dict(font=dict(size=18, color="black")),
        margin=chart_margins,
        legend=legend_config,
        xaxis=dict(
            tickfont=dict(size=14, color="black"), 
            title=dict(font=dict(color="black")), 
            title_text="Festival Relative Day",
            categoryorder="array",
            categoryarray=day_order
        ),
        yaxis=dict(
            tickfont=dict(size=14, color="black"),
            tickformat=".1%" if is_pct else None,
            title=dict(font=dict(color="black"))
        ),
        height=600,
        title_font_color="black"
    )
    
    if show_shading:
        # Day 1-2: Staging (Indices 0, 1)
        fig.add_vrect(
            x0=-0.5, x1=1.5, 
            fillcolor="rgba(128, 128, 128, 0.15)", 
            layer="below", line_width=0,
            annotation_text="Staging", 
            annotation_position="top left",
            annotation_font=dict(size=14, color="gray")
        )
        # Day 3-5: Operations (Indices 2, 3, 4)
        fig.add_vrect(
            x0=1.5, x1=4.5, 
            fillcolor="rgba(0, 255, 0, 0.1)", 
            layer="below", line_width=0,
            annotation_text="Operations", 
            annotation_position="top left",
            annotation_font=dict(size=14, color="green")
        )
        # Day 6-7: Demobilization (Indices 5, 6)
        fig.add_vrect(
            x0=4.5, x1=6.5, 
            fillcolor="rgba(255, 165, 0, 0.15)", 
            layer="below", line_width=0,
            annotation_text="Demobilization", 
            annotation_position="top left",
            annotation_font=dict(size=14, color="orange")
        )

    # Debug view
    if show_debug:
        with st.expander("📊 Data"):
            st.dataframe(df_trend)
    
    fig.update_traces(
        hovertemplate="<b>%{customdata[2]}</b> (%{customdata[1]})<br>" +
                      "Festival Day: %{x}<br>" +
                      "Date: %{customdata[0]}<br>" +
                      "Approach: %{customdata[4]}<br>" +
                      f"{selected_metric_label}: %{{y" + (".1%" if is_pct else "") + "}<extra></extra>",
    )
    
    # Apply customdata per trace to fix tooltip year issue
    for trace in fig.data:
        trace_df = df_trend[df_trend["Legend"] == trace.name]
        trace.customdata = trace_df[["DateStr", "Year", "Intersection", "Approach", "ApproachTooltip"]]

    st.plotly_chart(fig, use_container_width=True)

    # 4. Weekday Comparison Data
    if show_comparison_table:
        with st.expander("Weekday Comparison Data", expanded=True):
            # Pivot the data
            # Index: Intersection, Approach, Festival Day
            # Columns: Year
            # Values: Value, DateStr
            
            pivot_df = df_trend.copy()
            
            # Group by these to ensure we have unique rows for the pivot
            group_cols = ["Intersection", "Approach", "Festival Day"]
            
            # Create a pivot for values
            val_pivot = pivot_df.pivot_table(
                index=group_cols, 
                columns="Year", 
                values="Value", 
                aggfunc='first'
            ).reset_index()
            
            # Create a pivot for dates
            date_pivot = pivot_df.pivot_table(
                index=group_cols, 
                columns="Year", 
                values="DateStr", 
                aggfunc='first'
            ).reset_index()
            
            # Merge them or just construct manually if 2025 and 2026 are the only years
            available_years = sorted(pivot_df["Year"].unique())
            
            table_data = val_pivot[group_cols].copy()
            
            for yr in available_years:
                # Add Date column
                if yr in date_pivot.columns:
                    table_data[f"{yr} Date"] = date_pivot[yr]
                # Add Value column
                if yr in val_pivot.columns:
                    table_data[f"{yr} Value"] = val_pivot[yr]
            
            # If both 2025 and 2026 exist, calculate diff and % change
            if 2025 in available_years and 2026 in available_years:
                v2025 = table_data["2025 Value"]
                v2026 = table_data["2026 Value"]
                table_data["Difference"] = v2026 - v2025
                # Avoid division by zero
                table_data["% Change"] = (table_data["Difference"] / v2025).apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")
            
            # Formatting
            # Round numeric columns
            for yr in available_years:
                col_name = f"{yr} Value"
                if is_pct:
                    table_data[col_name] = table_data[col_name].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")
                else:
                    table_data[col_name] = table_data[col_name].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "N/A")
            
            if "Difference" in table_data.columns:
                if is_pct:
                    table_data["Difference"] = table_data["Difference"].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")
                else:
                    table_data["Difference"] = table_data["Difference"].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "N/A")

            # Final Column Ordering
            cols = ["Festival Day"]
            if len(trend_selection["intersections"]) > 1:
                cols.append("Intersection")
            if trend_selection.get("approach", "All Approaches") != "All Approaches":
                cols.append("Approach")
            
            for yr in available_years:
                cols.extend([f"{yr} Date", f"{yr} Value"])
            
            if "Difference" in table_data.columns:
                cols.extend(["Difference", "% Change"])
            
            # Sort by Festival Day
            table_data["Festival Day"] = pd.Categorical(table_data["Festival Day"], categories=day_order, ordered=True)
            table_data = table_data.sort_values(["Intersection", "Approach", "Festival Day"])
            
            st.dataframe(table_data[cols], use_container_width=True, hide_index=True)

    if num_lines > 6:
        st.info("💡 **Tip:** Selecting many intersections/years can make the chart cluttered. Consider focusing on fewer combinations for better clarity.")
