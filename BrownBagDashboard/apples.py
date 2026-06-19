import streamlit as st
import pandas as pd
import plotly.express as px


def _apply_comparison_layout(fig, y_axis_label, tickformat=None):
    """
    Applies a standard, clean layout for comparison charts in the apples-to-apples tab.
    Ensures that titles and legends have enough breathing room even at 100% zoom.
    """
    fig.update_layout(
        font=dict(family="Arial, sans-serif"),
        hoverlabel=dict(
            font_size=18,
        ),
        # Increase top margin for title and bottom margin for legend
        margin=dict(t=130, b=100, l=50, r=30),
        # Position title at the top of the container area
        title=dict(
            y=0.96,
            x=0.02,
            xanchor='left',
            yanchor='top',
            font=dict(size=26)
        ),
        # Anchor legend at the bottom of the plot area to prevent title collision
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            borderwidth=1,
            font=dict(size=20),
            itemsizing='constant'
        ),
        xaxis=dict(
            title=dict(text="<b>Approach</b>", font=dict(size=24)),
            tickfont=dict(size=18)
        ),
        yaxis=dict(
            title=dict(text=f"<b>{y_axis_label}</b>", font=dict(size=24)),
            tickformat=tickformat,
            tickfont=dict(size=18)
        ),
        height=550
    )


def render_apples_tab(registry, get_intersection_data_func, get_meta_value_func, direction_map, direction_colors=None,
                      manual_selection=None, load_daily_func=None):
    """
    Renders a tab for comparing two different time periods (datasets)
    for the same intersection (Apples-to-Apples Comparison).
    """
    import datetime
    if direction_colors is None:
        direction_colors = {}

    # 1. Selection logic
    # Find intersections that have multiple datasets for comparison OR daily data
    comparable_intersections = [i for i in registry if
                                len(i.get("datasets", [])) > 1 or i.get("daily_data", {}).get("available")]

    if not comparable_intersections:
        st.info("No intersections with multiple datasets or daily data found for comparison.")
        return

    # 2. Create container for results (this will appear first in the UI)
    results_container = st.container()

    # 3. Selection widgets
    if manual_selection:
        selected_label = manual_selection.get("label")
        selected_intersection = next((i for i in comparable_intersections if i["label"] == selected_label),
                                     comparable_intersections[0])
        p1_label = manual_selection.get("p1")
        p2_label = manual_selection.get("p2")
        use_daily = False  # Manual selection currently only supports predefined datasets
    else:
        st.markdown("---")
        st.write("### Comparison Selection")
        col_sel1, col_sel2, col_sel3 = st.columns([2, 1, 1])

        with col_sel1:
            selected_label = st.selectbox(
                "Select Intersection",
                options=[i["label"] for i in comparable_intersections],
                key="apples_intersection_selector"
            )
            selected_intersection = next(i for i in comparable_intersections if i["label"] == selected_label)

        daily_config = selected_intersection.get("daily_data", {})
        use_daily = daily_config.get("available", False)

        if use_daily:
            min_date = datetime.datetime.strptime(daily_config["date_range"]["start"], "%Y-%m-%d").date()
            max_date = datetime.datetime.strptime(daily_config["date_range"]["end"], "%Y-%m-%d").date()

            with col_sel2:
                st.write("**Baseline Period (P1)**")
                p1_start = st.date_input("Start Date (P1)", value=min_date, min_value=min_date, max_value=max_date,
                                         key="p1_start")
                p1_end = st.date_input("End Date (P1)", value=min_date + datetime.timedelta(days=6), min_value=min_date,
                                       max_value=max_date, key="p1_end")
                p1_label = f"{p1_start} to {p1_end}"
            with col_sel3:
                st.write("**Comparison Period (P2)**")
                p2_start = st.date_input("Start Date (P2)", value=max_date - datetime.timedelta(days=6),
                                         min_value=min_date, max_value=max_date, key="p2_start")
                p2_end = st.date_input("End Date (P2)", value=max_date, min_value=min_date, max_value=max_date,
                                       key="p2_end")
                p2_label = f"{p2_start} to {p2_end}"

            if p1_start > p1_end or p2_start > p2_end:
                st.error("Start date must be before or equal to end date.")
                return
        else:
            dataset_options = [d["date_label"] for d in selected_intersection["datasets"]]
            with col_sel2:
                p1_label = st.selectbox("Baseline Period (P1)", options=dataset_options, index=0)
            with col_sel3:
                p2_label = st.selectbox("Comparison Period (P2)", options=dataset_options,
                                        index=min(1, len(dataset_options) - 1))

    if p1_label == p2_label:
        st.warning("Please select two different time periods to generate a comparison.")
        return

    # 4. Fill results container
    with results_container:
        # Data Loading
        with st.spinner(f"Loading data for {p1_label} and {p2_label}..."):
            if use_daily and load_daily_func:
                data1 = load_daily_func(daily_config, p1_start, p1_end)
                data2 = load_daily_func(daily_config, p2_start, p2_end)
            else:
                d1 = next(d for d in selected_intersection["datasets"] if d["date_label"] == p1_label)
                d2 = next(d for d in selected_intersection["datasets"] if d["date_label"] == p2_label)
                data1 = get_intersection_data_func(selected_intersection, d1)
                data2 = get_intersection_data_func(selected_intersection, d2)

        if data1 is None or data2 is None:
            st.error("Could not load data for one or both time periods. Please check the data source.")
            return

        df_meta1, df_int1, df_app1, df_mov1 = data1
        df_meta2, df_int2, df_app2, df_mov2 = data2

        # Extract date ranges for display
        dr1 = f"{get_meta_value_func(df_meta1, 'Start Date')} to {get_meta_value_func(df_meta1, 'End Date')}"
        dr2 = f"{get_meta_value_func(df_meta2, 'Start Date')} to {get_meta_value_func(df_meta2, 'End Date')}"

        st.write("### High-Level KPI Comparison")

        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

        def get_delta_str(v1, v2, is_pct=False):
            if pd.isna(v1) or pd.isna(v2): return None
            diff = v2 - v1
            if is_pct:
                # If values are on 0-1 scale, diff * 100 is percentage points
                if v1 < 1 and v2 < 1:
                    return f"{diff * 100:+.1f}% pts"
                return f"{diff:+.1f}%"
            return f"{diff:+.1f}" if abs(diff) < 1000 else f"{int(diff):+,}"

        def format_val(v, is_pct=False, is_int=False):
            if pd.isna(v): return "N/A"
            if is_pct:
                if v >= 1: return f"{v:.1f}%"
                return f"{v:.1%}"
            if is_int: return f"{int(v):,}"
            return f"{v:.1f}"

        # Metric mapping
        metrics = [
            ("Delay Range 1", "Average Delay (s)", kpi_col1, False, False, "inverse"),
            ("Arrivals On Green Range 1", "Arrivals On Green", kpi_col2, True, False, "normal"),
            ("Split Failures Range 1", "Split Failures", kpi_col3, True, False, "inverse"),
            ("Vehicle Samples 1", "Total Vehicles", kpi_col4, False, True, "normal")
        ]

        for col_name, label, column, is_pct, is_int, d_color in metrics:
            v1 = df_int1[col_name].iloc[0]
            v2 = df_int2[col_name].iloc[0]

            with column:
                # Calculate delta for styling
                diff = v2 - v1 if not (pd.isna(v1) or pd.isna(v2)) else 0
                delta_str = get_delta_str(v1, v2, is_pct)

                # Determine color logic consistent with st.metric
                if delta_str is None or diff == 0:
                    d_style = "color: gray;"
                else:
                    is_positive = diff > 0
                    if d_color == "normal":
                        color = "#10b981" if is_positive else "#ef4444"  # green vs red
                    else:  # inverse
                        color = "#ef4444" if is_positive else "#10b981"  # red vs green
                    d_style = f"color: {color}; font-weight: 700;"

                st.markdown(f"""
<div style="background: var(--secondary-background-color); padding: 15px 5px; border-radius: 12px; border: 2px solid #1f4582; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.05); min-height: 150px; display: flex; flex-direction: column; justify-content: center;">
    <div style="font-size: 0.8rem; color: #1f4582; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px; margin-bottom: 8px; line-height: 1.2;">{label}<br><span style="font-size: 0.7rem; opacity: 0.8;">({p2_label})</span></div>
    <div style="font-size: 1.8rem; font-weight: 900; color: var(--text-color); line-height: 1;">{format_val(v2, is_pct, is_int)}</div>
    {f'<div style="font-size: 0.9rem; {d_style} margin-top: 5px;">{delta_str}</div>' if delta_str else ''}
    <div style="font-size: 0.75rem; opacity: 0.6; margin-top: 8px;">{p1_label}: {format_val(v1, is_pct, is_int)}</div>
</div>
""", unsafe_allow_html=True)

        # Comparative Visualizations
        st.markdown("---")
        st.write("### Performance by Approach Comparison")

        # Helper to prep and concat data
        def prep_df(df, label):
            df_new = df.copy()
            df_new["Approach"] = df_new["Approach"].astype(str).str.strip()
            df_new["Approach Full"] = df_new["Approach"].map(direction_map).fillna(df_new["Approach"])
            df_new["Period"] = label
            return df_new

        df_comp_app = pd.concat([prep_df(df_app1, p1_label), prep_df(df_app2, p2_label)], ignore_index=True)

        # Normalize pct columns if needed
        for col in ["Arrivals On Green Range 1", "Split Failures Range 1"]:
            if df_comp_app[col].max() > 1:
                df_comp_app[col] = df_comp_app[col] / 100.0

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_delay = px.bar(
                df_comp_app,
                x="Approach Full",
                y="Delay Range 1",
                color="Period",
                barmode="group",
                title=f"<b>{selected_label}</b><br><sup>Comparison: Average Delay (s) | {p1_label} vs {p2_label}</sup>",
                labels={"Delay Range 1": "Average Delay (s)", "Approach Full": "Approach"},
                color_discrete_map={p1_label: "#94a3b8", p2_label: "#1f4582"},
                text_auto='.1f'
            )
            _apply_comparison_layout(fig_delay, "Control Delay (seconds)")
            fig_delay.update_traces(
                textfont_size=18,
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>Approach:</b> %{x}<br><b>%{fullData.name}:</b> %{y:.1f} seconds<extra></extra>"
            )
            st.plotly_chart(fig_delay, use_container_width=True)

        with chart_col2:
            fig_aog = px.bar(
                df_comp_app,
                x="Approach Full",
                y="Arrivals On Green Range 1",
                color="Period",
                barmode="group",
                title=f"<b>{selected_label}</b><br><sup>Comparison: Arrivals On Green | {p1_label} vs {p2_label}</sup>",
                labels={"Arrivals On Green Range 1": "Arrivals On Green (%)", "Approach Full": "Approach"},
                color_discrete_map={p1_label: "#94a3b8", p2_label: "#1f4582"},
                text_auto='.1%'
            )
            _apply_comparison_layout(fig_aog, "Arrivals On Green (%)", tickformat=".0%")
            fig_aog.update_traces(
                textfont_size=18,
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>Approach:</b> %{x}<br><b>%{fullData.name}:</b> %{y:.1%}<extra></extra>"
            )
            st.plotly_chart(fig_aog, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)
        with chart_col3:
            fig_sf = px.bar(
                df_comp_app,
                x="Approach Full",
                y="Split Failures Range 1",
                color="Period",
                barmode="group",
                title=f"<b>{selected_label}</b><br><sup>Comparison: Split Failures | {p1_label} vs {p2_label}</sup>",
                labels={"Split Failures Range 1": "Split Failures (%)", "Approach Full": "Approach"},
                color_discrete_map={p1_label: "#94a3b8", p2_label: "#1f4582"},
                text_auto='.1%'
            )
            _apply_comparison_layout(fig_sf, "Split Failures (%)", tickformat=".0%")
            fig_sf.update_traces(
                textfont_size=18,
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>Approach:</b> %{x}<br><b>%{fullData.name}:</b> %{y:.1%}<extra></extra>"
            )
            st.plotly_chart(fig_sf, use_container_width=True)

        with chart_col4:
            fig_vol = px.bar(
                df_comp_app,
                x="Approach Full",
                y="Vehicle Samples 1",
                color="Period",
                barmode="group",
                title=f"<b>{selected_label}</b><br><sup>Comparison: Total Volume | {p1_label} vs {p2_label}</sup>",
                labels={"Vehicle Samples 1": "Total Volume (Vehicles)", "Approach Full": "Approach"},
                color_discrete_map={p1_label: "#94a3b8", p2_label: "#1f4582"},
                text_auto=',.0f'
            )
            _apply_comparison_layout(fig_vol, "Total Vehicles")
            fig_vol.update_traces(
                textfont_size=18,
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>Approach:</b> %{x}<br><b>%{fullData.name}:</b> %{y:,.0f} vehicles<extra></extra>"
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        # --- Movement-Level Comparison ---
        st.markdown("---")
        st.write("### Movement-Level Comparison")
        st.caption("Detailed period-over-period comparison by individual traffic movement.")

        # Map for human-readable metrics (consistent with brownbag.py)
        mov_metric_map = {
            "Avg Control Delay (seconds)": "Delay Range 1",
            "Arrivals On Green %": "Arrivals On Green Range 1",
            "Split Failure %": "Split Failures Range 1",
            "Vehicles": "Vehicle Samples 1"
        }

        # Check if 'Vehicle Samples 1' exists in df_mov1, if not use Turning Movement (consistent with brownbag.py)
        if "Vehicle Samples 1" not in df_mov1.columns:
            mov_metric_map["Vehicles"] = "Turning Movement Range 1"

        col_mov_f1, col_mov_f2 = st.columns([1, 1])

        # Prep df_mov data for filtering and mapping
        def prep_mov_df(df, label):
            df_new = df.copy()
            df_new["Approach"] = df_new["Approach"].astype(str).str.strip()
            df_new["Approach Full"] = df_new["Approach"].map(direction_map).fillna(df_new["Approach"])
            df_new["Period"] = label
            # Clean movement names to standard set (Left, Thru, Right)
            df_new["Movement"] = df_new["Movement"].astype(str).str.strip()
            # Normalize common variants
            mov_norm = {
                "Thru": "Thru", "Through": "Thru", "T": "Thru",
                "Left": "Left", "Left Turn": "Left", "L": "Left",
                "Right": "Right", "Right Turn": "Right", "R": "Right",
                "U-Turn": "Left"  # Group U-turns with Lefts for simplicity if they exist
            }
            df_new["Movement"] = df_new["Movement"].map(mov_norm).fillna(df_new["Movement"])
            return df_new

        df_comp_mov = pd.concat([prep_mov_df(df_mov1, p1_label), prep_mov_df(df_mov2, p2_label)], ignore_index=True)

        with col_mov_f1:
            # We filter for the 4 primary approaches by default
            primary_approaches = ["Northbound", "Southbound", "Eastbound", "Westbound"]
            available_apps = sorted(list(df_comp_mov["Approach Full"].unique()))
            # Ensure primary approaches are prioritized if they exist
            default_apps = [a for a in primary_approaches if a in available_apps]
            if not default_apps:
                default_apps = available_apps[:4]

            selected_mov_apps = st.multiselect(
                "Filter Approaches",
                options=available_apps,
                default=default_apps,
                key="apples_mov_app_selector"
            )

        with col_mov_f2:
            selected_mov_metric_label = st.selectbox(
                "Metric to Compare",
                list(mov_metric_map.keys()),
                key="apples_mov_metric_selector"
            )
            selected_mov_metric_col = mov_metric_map[selected_mov_metric_label]

        # 1. Filter for the selected approaches and only Thru, Right, Left
        standard_movements = ["Thru", "Right", "Left"]
        filtered_mov_df = df_comp_mov[
            (df_comp_mov["Approach Full"].isin(selected_mov_apps)) &
            (df_comp_mov["Movement"].isin(standard_movements))
            ].copy()

        # Handle Percentages (Divide by 100 if needed)
        is_percentage = "Split Failure" in selected_mov_metric_label or "Arrivals On Green" in selected_mov_metric_label
        if is_percentage:
            if filtered_mov_df[selected_mov_metric_col].max() > 1:
                filtered_mov_df[selected_mov_metric_col] = filtered_mov_df[selected_mov_metric_col] / 100.0

        # Determine formatting based on metric
        mov_text_fmt = '.1f'
        mov_hover_fmt = ":.1f"
        mov_y_label = selected_mov_metric_label
        mov_tick_fmt = None
        mov_unit = ""

        if "Delay" in selected_mov_metric_label:
            mov_unit = " seconds"
        elif "Volume" in selected_mov_metric_label or "Vehicles" in selected_mov_metric_label:
            mov_text_fmt = ',.0f'
            mov_hover_fmt = ":,d"
            mov_unit = " vehicles"
        else:
            mov_text_fmt = '.1%'
            mov_hover_fmt = ":.1%"
            mov_tick_fmt = ".0%"

        # Sort movements for consistent display (Left, Thru, Right)
        mov_order = {"Left": 0, "Thru": 1, "Right": 2}
        filtered_mov_df["mov_sort"] = filtered_mov_df["Movement"].map(mov_order)
        filtered_mov_df = filtered_mov_df.sort_values(["Approach Full", "mov_sort"])

        # Calculate Deltas for text labels on chart
        df_p1 = filtered_mov_df[filtered_mov_df["Period"] == p1_label].copy()
        df_p2 = filtered_mov_df[filtered_mov_df["Period"] == p2_label].copy()
        df_merge = pd.merge(df_p1, df_p2, on=["Approach Full", "Movement"], suffixes=("_p1", "_p2"))

        # Map deltas and raw values back to the P2 rows in the original filtered_mov_df
        labels_map = {}
        for _, row in df_merge.iterrows():
            v1 = row[selected_mov_metric_col + "_p1"]
            v2 = row[selected_mov_metric_col + "_p2"]

            # Formatting for the count value
            if "Vehicles" in selected_mov_metric_label:
                v2_str = f"{v2:,.0f}"
            elif is_percentage:
                v2_str = f"{v2:.1%}"
            else:
                v2_str = f"{v2:.1f}"

            # Delta logic with color coding
            if v1 != 0 and not pd.isna(v1) and not pd.isna(v2):
                delta_pct = (v2 - v1) / v1
                delta_str = f"{delta_pct:+.0%}"

                # Determine if increase is good or bad (inverse metrics: Delay, Split Failure)
                is_inverse = "Delay" in selected_mov_metric_label or "Split Failure" in selected_mov_metric_label

                # Determine delta color
                if delta_pct >= 0.005:
                    delta_color = "red" if is_inverse else "green"
                elif delta_pct <= -0.005:
                    delta_color = "green" if is_inverse else "red"
                else:
                    delta_color = "inherit"

                # Combine the raw value and the % change
                combined_label = f"{v2_str}<br><span style='color:{delta_color}'>({delta_str})</span>"
            else:
                combined_label = v2_str

            labels_map[(row["Approach Full"], row["Movement"])] = combined_label

        filtered_mov_df["Chart Label"] = filtered_mov_df.apply(
            lambda x: labels_map.get((x["Approach Full"], x["Movement"]), "") if x["Period"] == p2_label else "",
            axis=1
        )

        # Create the faceted small multiples chart
        fig_mov_comp = px.bar(
            filtered_mov_df,
            x="Movement",
            y=selected_mov_metric_col,
            color="Period",
            facet_col="Approach Full",
            facet_col_wrap=2,
            facet_row_spacing=0.15,  # Increase vertical gap between rows
            barmode="group",
            title=f"<b>{selected_label}</b><br><sup>Movement Comparison: {selected_mov_metric_label} | {p1_label} vs {p2_label}</sup>",
            labels={
                selected_mov_metric_col: selected_mov_metric_label,
                "Movement": "Movement",
                "Approach Full": "Approach"
            },
            color_discrete_map={p1_label: "#94a3b8", p2_label: "#1f4582"},  # Consistent Period Colors (Light/Dark)
            category_orders={"Movement": standard_movements},
            text="Chart Label"  # Show both raw count and % change on the P2 bar
        )

        # Clean up facet titles and styling
        # Move facet titles up to avoid intersecting with borders
        fig_mov_comp.for_each_annotation(lambda a: a.update(
            text=f"<b>{a.text.split('=')[-1]}</b>",
            font=dict(size=18),
            y=a.y + 0.02  # Bump titles up slightly (reduced from 0.04 to give more headroom)
        ))

        # Ensure y-axis labels and formatting are correct
        _apply_comparison_layout(fig_mov_comp, mov_y_label, tickformat=mov_tick_fmt)

        # Add headroom for labels on top of bars
        max_val = filtered_mov_df[selected_mov_metric_col].max()
        if pd.notna(max_val) and max_val > 0:
            # Increase range by 25% to accommodate multi-line labels like "14,730 (+10%)"
            fig_mov_comp.update_yaxes(range=[0, max_val * 1.25])

        # Layout refinement (Overrides generic layout for faceted movement chart)
        fig_mov_comp.update_layout(
            margin=dict(t=130, b=100, l=60, r=40),
            title=dict(
                y=0.96,
                x=0.02,
                xanchor='left',
                yanchor='top'
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,  # Moved to bottom to prevent collision
                xanchor="center",
                x=0.5,
                title=None,
                font=dict(size=20),
                itemsizing='constant'
            ),
            hovermode="x unified",
            uniformtext_minsize=7,
            uniformtext_mode='show',
            height=700  # Keep height stable regardless of approach count to prevent jumping UI
        )

        # Remove "Movement" title from each facet x-axis to reduce clutter
        # and ENSURE labels show on all subplots (matches all x-axes)
        fig_mov_comp.update_xaxes(
            title=None,
            showline=True,
            linewidth=1,
            mirror=True,
            showticklabels=True,  # Force labels on all subplots
            tickfont=dict(size=14)
        )
        fig_mov_comp.update_yaxes(
            showline=True,
            linewidth=1,
            mirror=True,
            tickfont=dict(size=14),
            title_font=dict(size=18)
        )

        fig_mov_comp.update_traces(
            textfont_size=14,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{fullData.name}:</b> %{y" + mov_hover_fmt + "}" + mov_unit + "<extra></extra>"
        )

        # --- OPTIONAL: Add Delta labels above each group ---
        # We can do this by adding a invisible trace or annotations.
        # However, Plotly's faceted grouping makes it slightly tricky.
        # A simpler way is to just use the text_auto which we already have.
        # But for % change, we might need a separate expander or table.
        # Let's add an expander for the actual delta data for better "analytical" depth.

        st.plotly_chart(fig_mov_comp, use_container_width=True)

        with st.expander("View Movement Data & % Changes"):
            # Calculate Deltas for the selected metric
            df_p1 = filtered_mov_df[filtered_mov_df["Period"] == p1_label].copy()
            df_p2 = filtered_mov_df[filtered_mov_df["Period"] == p2_label].copy()

            # Merge to compare
            df_delta = pd.merge(
                df_p1[["Approach Full", "Movement", selected_mov_metric_col]],
                df_p2[["Approach Full", "Movement", selected_mov_metric_col]],
                on=["Approach Full", "Movement"],
                suffixes=("_p1", "_p2")
            )

            # Use period labels and metric name for column headers as requested (e.g. "2025 Volume")
            metric_display_name = "Volume" if selected_mov_metric_label == "Vehicles" else selected_mov_metric_label
            col_p1 = f"{p1_label} {metric_display_name}"
            col_p2 = f"{p2_label} {metric_display_name}"

            # Rename for display
            df_delta = df_delta.rename(columns={
                "Approach Full": "Approach",
                selected_mov_metric_col + "_p1": col_p1,
                selected_mov_metric_col + "_p2": col_p2
            })

            df_delta["Change"] = df_delta[col_p2] - df_delta[col_p1]
            if is_percentage:
                df_delta["% Change"] = df_delta["Change"]  # It's already percentage points
            else:
                # Avoid division by zero
                df_delta["% Change"] = df_delta.apply(
                    lambda row: (row["Change"] / row[col_p1]) if row[col_p1] != 0 else 0,
                    axis=1
                )

            # Display formatted table
            styler = df_delta.style.format({
                col_p1: mov_text_fmt.replace('.1%', '{:.1%}').replace(',.0f', '{:,.0f}').replace('.1f', '{:.1f}'),
                col_p2: mov_text_fmt.replace('.1%', '{:.1%}').replace(',.0f', '{:,.0f}').replace('.1f', '{:.1f}'),
                "Change": mov_text_fmt.replace('.1%', '{:+.1%}').replace(',.0f', '{:+,.0f}').replace('.1f', '{:+.1f}'),
                "% Change": "{:+.1%}"
            })

            # --- Handle background gradient coloring (requires matplotlib) ---
            # NOTE: We MUST check for matplotlib explicitly BEFORE calling background_gradient,
            # because Pandas may defer the actual import until rendering time (st.dataframe),
            # which makes a try-except block around background_gradient alone insufficient.
            try:
                import matplotlib
                matplotlib_available = True
            except (ImportError, ModuleNotFoundError):
                matplotlib_available = False

            if matplotlib_available:
                try:
                    cmap_name = "RdYlGn_r" if "Delay" in selected_mov_metric_label or "Split Failure" in selected_mov_metric_label else "RdYlGn"
                    styler = styler.background_gradient(subset=["Change"], cmap=cmap_name)
                except Exception:
                    matplotlib_available = False

            if not matplotlib_available:
                # Fallback to simple color highlights if matplotlib is not available
                def color_delta(val):
                    if pd.isna(val) or val == 0: return ''
                    is_inv = "Delay" in selected_mov_metric_label or "Split Failure" in selected_mov_metric_label
                    # Red highlight for "bad" change, Green for "good" change
                    if is_inv:
                        bg_color = "#fecaca" if val > 0 else "#bbf7d0"  # Light Red/Green
                    else:
                        bg_color = "#bbf7d0" if val > 0 else "#fecaca"  # Light Green/Red
                    return f"background-color: {bg_color}"

                # Check pandas version for styling method (applymap vs map)
                if hasattr(styler, 'map'):
                    styler = styler.map(color_delta, subset=["Change"])
                else:
                    styler = styler.applymap(color_delta, subset=["Change"])

            st.dataframe(styler, use_container_width=True)

        st.markdown("---")
        st.info(f"**Baseline Period (P1):** {p1_label} ({dr1})  \n"
                f"**Comparison Period (P2):** {p2_label} ({dr2})")