import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# (no local file access or mock generation needed)
from MapMap import render_map
# (no local file access or mock generation needed)

# Set page configuration
st.set_page_config(
    page_title="Intersection Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
# Use the live dataset hosted on GitHub (raw URL)
DATA_URL = "https://raw.githubusercontent.com/chrquija/BrownBag_Dashboard/refs/heads/main/data/W_SAN_RAFAEL_RD_and_TRAMWAY_RD.xlsx"
INTERSECTION_NAME = "N PALM CANYON DR & W SAN RAFAEL RD & TRAMWAY RD"


# --- Data Loading Functions ---


@st.cache_data
def load_data(source: str):
    """Load data from a local path or URL. Returns tuple of DataFrames or None on failure.

    Expected sheets:
      - Metadata
      - Intersection
      - By Approach
      - By Movement
    """
    def _read_all(src: str):
        xls = pd.ExcelFile(src)
        available = set([s.strip() for s in xls.sheet_names])

        required = {
            "Metadata": None,
            "Intersection": None,
            "By Approach": None,
            "By Movement": None,
        }

        missing = [name for name in required.keys() if name not in available]
        if missing:
            st.error(
                "Missing required sheet(s) in workbook: "
                + ", ".join(missing)
                + f". Found: {', '.join(available)}"
            )
            return None

        df_meta = pd.read_excel(xls, sheet_name="Metadata")
        df_int = pd.read_excel(xls, sheet_name="Intersection")
        df_app = pd.read_excel(xls, sheet_name="By Approach")
        df_mov = pd.read_excel(xls, sheet_name="By Movement")
        return df_meta, df_int, df_app, df_mov

    try:
        # Primary attempt
        data = _read_all(source)
        if data is not None:
            return data
        return None
    except Exception as e:
        # If user provided a GitHub raw URL with refs/heads, try the simplified branch form
        if isinstance(source, str) and "refs/heads/" in source:
            alt = source.replace("refs/heads/", "")
            try:
                data = _read_all(alt)
                if data is not None:
                    st.info("Loaded data from alternate URL format of the provided GitHub raw link.")
                    return data
            except Exception:
                pass
        st.error(f"Error reading Excel file: {e}")
        return None


# --- Main Application ---

def main():
    # Custom header + sidebar metadata
    corridor = "North Palm Canyon Drive"
    intersection = "N PALM CANYON DR & W SAN RAFAEL RD & TRAMWAY RD"
    primary_street = "N PALM CANYON"
    secondary_street = "W SAN RAFAEL RD"
    tertiary_street = "TRAMWAY RD"
    city = "City of Palm Springs"
    date_range = "10-01 -> 11-30 2025"
    coordinates = "33.85832, -116.55739"
    # Sidebar: move static metadata out of the main header to reduce crowding
    st.sidebar.markdown("## Location Info")
    st.sidebar.markdown(
        f"""
- **Corridor:** {corridor}
- **Intersection:** {intersection}
- **Primary:** {primary_street}
- **Secondary:** {secondary_street}
- **Tertiary:** {tertiary_street}
- **City:** {city}
- **Date Range:** {date_range}
- **Coordinates:** {coordinates}
        """
    )

    # Optional toggles for showing details in the main area
    show_details_in_header = st.sidebar.checkbox("Show metadata in banner", value=False)
    show_details_expander = st.sidebar.checkbox("Show details expander under banner", value=True)

    # Compact header (title + key subtitle only)
    header_html = f"""
    <style>
        .bbg-header {{
            background: #1f4582;
            color: #ffffff;
            padding: 14px 20px; /* compact */
            border-radius: 14px;
            margin-bottom: 14px;
        }}
        .bbg-header h1 {{
            margin: 0 0 2px 0;
            font-size: 24px;
            font-weight: 800;
        }}
        .bbg-header .subtitle {{
            opacity: 0.95;
            font-size: 14px;
            margin: 0;
        }}
        .bbg-meta {{
            display: grid;
            grid-template-columns: repeat(2, minmax(200px, 1fr));
            gap: 4px 16px;
            font-size: 13px;
            margin-top: 8px;
        }}
        .bbg-meta div span {{ opacity: 0.9; }}
        @media (max-width: 800px) {{ .bbg-meta {{ grid-template-columns: 1fr; }} }}
    </style>
    <div class="bbg-header">
        <h1>Intersection Performance Dashboard</h1>
        <p class="subtitle">{intersection}</p>
        {f'<div class="bbg-meta">\n'
           f'  <div><strong>Corridor:</strong> <span>{corridor}</span></div>\n'
           f'  <div><strong>City:</strong> <span>{city}</span></div>\n'
           f'  <div><strong>Date Range:</strong> <span>{date_range}</span></div>\n'
           f'  <div><strong>Coordinates:</strong> <span>{coordinates}</span></div>\n'
           f'</div>' if show_details_in_header else ''}
    </div>
    """

    st.markdown(header_html, unsafe_allow_html=True)

    # Optional expander below the banner for quick access without opening sidebar
    if show_details_expander:
        with st.expander("Location details"):
            st.markdown(
                f"""
**Corridor:** {corridor}  
**Intersection:** {intersection}  
**Primary Street:** {primary_street}  
**Secondary Street:** {secondary_street}  
**Tertiary Street:** {tertiary_street}  
**City:** {city}  
**Date Range:** {date_range}  
**Coordinates:** {coordinates}
                """
            )

    # Right rail layout: create persistent two-column canvas
    st.markdown("---")
    
    # Make the right rail sticky and a bit thinner via CSS
    st.markdown(
        """
        <style>
        /* Sticky right rail: target the 2nd column's inner block */
        div[data-testid="column"]:nth-child(2) > div {
            position: sticky;
            top: 96px; /* leave space for header */
        }
        /* Disable stickiness on small screens to avoid layout issues */
        @media (max-width: 1100px) {
            div[data-testid="column"]:nth-child(2) > div { position: static; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Make the map column thinner: 8/4 split
    left_col, right_col = st.columns([8, 4], gap="large")

    # Map stays pinned in the right rail
    with right_col:
        render_map(
            latitude=33.85832,
            longitude=-116.55739,
            height=900,  # longer map
            zoom=13,
            label=intersection,
        )

    # All analytics content lives inside the left column
    with left_col:
        # Attempt to load data (from live URL). If it fails, stop execution.
        data = load_data(DATA_URL)

        if data is None:
            st.stop()
        else:
            df_meta, df_int, df_app, df_mov = data

        st.caption(f"Data source: {DATA_URL}")

        # 1. High-level KPIs (Intersection Sheet)
        st.markdown("---")
        st.subheader("Advantec AI Overview")

        # Helper to format percentages safely (handles 0.78 vs 78)
        def format_percent(val):
            if pd.isna(val):
                return "N/A"
            # Changed > to >= so that a value of exactly 1 is treated as 1%, not 100%
            if val >= 1:
                return f"{val:.1f}%"  # Assumes 0-100 scale
            return f"{val:.1%}"       # Assumes 0-1 scale

        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

        with kpi_col1:
            val = df_int["Delay Range 1"].iloc[0]
            st.metric("Average Delay (s)", f"{val:.1f} seconds")

        with kpi_col2:
            val = df_int["Arrivals On Green Range 1"].iloc[0]
            st.metric("Arrivals On Green", format_percent(val))

        with kpi_col3:
            val = df_int["Split Failures Range 1"].iloc[0]
            st.metric("Split Failures", format_percent(val))

        with kpi_col4:
            val = df_int["Vehicle Samples 1"].iloc[0]
            st.metric("Total Volume", f"{int(val):,} vehicles")

        # 2. Approach Analysis (By Approach Sheet)
        st.markdown("---")
        st.subheader("Performance by Approach")

        # Map abbreviated directions to full names
        direction_map = {
            "NB": "Northbound", "SB": "Southbound", "EB": "Eastbound", "WB": "Westbound",
            "NE": "Northeast", "NW": "Northwest", "SE": "Southeast", "SW": "Southwest"
        }
        # Create a copy to avoid SettingWithCopy warning on the cached dataframe
        df_app_plot = df_app.copy()
        df_app_plot["Approach Full"] = df_app_plot["Approach"].map(direction_map).fillna(df_app_plot["Approach"])

        col_chart_1, col_chart_2 = st.columns(2)

        with col_chart_1:
            fig_delay = px.bar(
                df_app_plot,
                x="Approach Full",
                y="Delay Range 1",
                title="Average Delay by Approach",
                text_auto='.1f',
                color="Delay Range 1",
                color_continuous_scale="RdYlGn_r",  # High delay = Red
                labels={"Delay Range 1": "Control Delay (seconds)", "Approach Full": "Approach"}
            )
            fig_delay.update_layout(
                showlegend=False,
                yaxis_title="Control Delay (seconds)"
            )
            # Increase text size inside bars
            fig_delay.update_traces(textfont_size=16)

            st.plotly_chart(fig_delay, use_container_width=True)

        with col_chart_2:
            # Comparison of Volume vs Split Failures
            fig_combo = go.Figure()

            # Bar for Volume (Use Vehicle Samples 1)
            # Check if column exists, otherwise fallback
            vol_col = "Vehicle Samples 1" if "Vehicle Samples 1" in df_app_plot.columns else "Turning Movement Range 1"

            fig_combo.add_trace(go.Bar(
                x=df_app_plot["Approach Full"],
                y=df_app_plot[vol_col],
                name="Volume",
                marker_color='rgb(55, 83, 109)'
            ))

            # Line for Split Failures
            # DIVIDE BY 100 to convert integer 4 to 0.04 (4%)
            split_fail_values = df_app_plot["Split Failures Range 1"] / 100.0

            fig_combo.add_trace(go.Scatter(
                x=df_app_plot["Approach Full"],
                y=split_fail_values,
                name="Split Failure %",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color='rgb(219, 64, 82)', width=3)
            ))

            fig_combo.update_layout(
                title="Volume vs. Split Failures",
                xaxis_title="Approach",
                yaxis=dict(title="Volume"),
                yaxis2=dict(title="Split Failures %", overlaying="y", side="right", tickformat=".1%"),
                legend=dict(x=0, y=1.2, orientation="h")
            )
            st.plotly_chart(fig_combo, use_container_width=True)

        # 3. Detailed Movement Analysis (By Movement Sheet)
        st.markdown("---")
        st.subheader("Movement Details")

        # Map for human-readable metrics
        metric_map = {
            "Avg Control Delay (seconds)": "Delay Range 1",
            "Arrivals On Green %": "Arrivals On Green Range 1",
            "Split Failure %": "Split Failures Range 1",
            "Volume": "Vehicle Samples 1"  # Assuming we want sample count here too
        }

        # Check if 'Vehicle Samples 1' exists in df_mov, if not use Turning Movement
        if "Vehicle Samples 1" not in df_mov.columns:
            metric_map["Volume"] = "Turning Movement Range 1"

        # Filter controls
        col_filter, col_display = st.columns([1, 3])

        with col_filter:
            st.write("**Filter Data**")

            # Use full names for filter
            # Get unique approaches from data
            unique_apps = sorted(df_mov["Approach"].unique())
            # Create display labels
            app_labels = [direction_map.get(a, a) for a in unique_apps]

            selected_labels = st.multiselect(
                "Select Approach",
                options=app_labels,
                default=app_labels
            )

            # Convert selected labels back to codes (e.g., "Northeast" -> "NE")
            reverse_map = {v: k for k, v in direction_map.items()}
            selected_codes = [reverse_map.get(l, l) for l in selected_labels]

            selected_metric_label = st.selectbox(
                "Select Metric to Visualize",
                list(metric_map.keys())
            )
            selected_metric_col = metric_map[selected_metric_label]

        with col_display:
            filtered_df = df_mov[df_mov["Approach"].isin(selected_codes)].copy()

            # Add full name column for plotting
            filtered_df["Approach Full"] = filtered_df["Approach"].map(direction_map).fillna(filtered_df["Approach"])

            # Handle Percentages (Divide by 100 if needed)
            # Apply to BOTH Split Failure AND Arrivals On Green
            if "Split Failure" in selected_metric_label or "Arrivals On Green" in selected_metric_label:
                # Check if values are integers > 1 (like 32 for 32%)
                if filtered_df[selected_metric_col].max() > 1:
                    filtered_df[selected_metric_col] = filtered_df[selected_metric_col] / 100.0

            # Determine text format
            if "Delay" in selected_metric_label:
                text_fmt = '.1f'
            elif "Volume" in selected_metric_label:
                text_fmt = '.0f'
            else:
                text_fmt = '.1%'

            fig_mov = px.bar(
                filtered_df,
                x="Approach Full",
                y=selected_metric_col,
                color="Movement",
                barmode="group",
                title=f"{selected_metric_label} by Movement",
                text_auto=text_fmt,
                labels={selected_metric_col: selected_metric_label, "Approach Full": "Approach"}
            )

            # INCREASE TEXT SIZE HERE
            fig_mov.update_traces(textfont_size=16)

            st.plotly_chart(fig_mov, use_container_width=True)

        # Raw Data Expander
        with st.expander("View Raw Data"):
            st.write("Intersection Data", df_int)
            st.write("Approach Data", df_app)
            st.write("Movement Data", df_mov)


if __name__ == "__main__":
    main()