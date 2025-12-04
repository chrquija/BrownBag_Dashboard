import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Intersection Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
FILE_PATH = "data/W_SAN_RAFAEL_RD_and_TRAMWAY_RD.xlsx"
INTERSECTION_NAME = "N PALM CANYON DR & W SAN RAFAEL RD & TRAMWAY RD"


# --- Data Loading Functions ---

def generate_mock_data():
    """Generates mock data matching the described structure for demo purposes."""
    # Intersection Sheet
    df_int = pd.DataFrame({
        "Delay Range 1": [32.5],
        "Arrivals On Green Range 1": [0.65],
        "Split Failures Range 1": [0.12],
        "Turning Movement Range 1": [12500]
    })

    # By Approach Sheet
    approaches = ["NB", "SB", "EB", "WB"]
    df_app = pd.DataFrame({
        "Approach": approaches,
        "Delay Range 1": [25.0, 45.2, 15.5, 22.1],
        "Arrivals On Green Range 1": [0.70, 0.45, 0.85, 0.75],
        "Split Failures Range 1": [0.05, 0.25, 0.02, 0.08],
        "Turning Movement Range 1": [4000, 3500, 2000, 3000]
    })

    # By Movement Sheet
    movements = []
    for app in approaches:
        for move in ["L", "T", "R"]:
            movements.append({
                "Approach": app,
                "Movement": move,
                "Delay Range 1": np.random.uniform(10, 60),
                "Arrivals On Green Range 1": np.random.uniform(0.3, 0.9),
                "Split Failures Range 1": np.random.uniform(0, 0.3),
                "Turning Movement Range 1": np.random.randint(100, 1500)
            })
    df_mov = pd.DataFrame(movements)

    return None, df_int, df_app, df_mov


@st.cache_data
def load_data(filepath):
    if not os.path.exists(filepath):
        return None

    try:
        df_meta = pd.read_excel(filepath, sheet_name="Metadata")
        df_int = pd.read_excel(filepath, sheet_name="Intersection")
        df_app = pd.read_excel(filepath, sheet_name="By Approach")
        df_mov = pd.read_excel(filepath, sheet_name="By Movement")
        return df_meta, df_int, df_app, df_mov
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None


# --- Main Application ---

def main():
    st.title(INTERSECTION_NAME)
    st.markdown("### Intersection Performance Dashboard")

    # Attempt to load data
    data = load_data(FILE_PATH)

    if data is None:
        st.info(f"File not found at `{FILE_PATH}`. Showing demonstration data based on the specified structure.")
        df_meta, df_int, df_app, df_mov = generate_mock_data()
    else:
        df_meta, df_int, df_app, df_mov = data

    # 1. High-level KPIs (Intersection Sheet)
    st.markdown("---")
    st.subheader("Overview")

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        val = df_int["Delay Range 1"].iloc[0]
        st.metric("Avg Delay (s)", f"{val:.1f}")

    with kpi_col2:
        val = df_int["Arrivals On Green Range 1"].iloc[0]
        st.metric("Arrivals On Green", f"{val:.1%}")

    with kpi_col3:
        val = df_int["Split Failures Range 1"].iloc[0]
        st.metric("Split Failures", f"{val:.1%}")

    with kpi_col4:
        val = df_int["Turning Movement Range 1"].iloc[0]
        st.metric("Total Volume", f"{int(val):,}")

    # 2. Approach Analysis (By Approach Sheet)
    st.markdown("---")
    st.subheader("Performance by Approach")

    col_chart_1, col_chart_2 = st.columns(2)

    with col_chart_1:
        fig_delay = px.bar(
            df_app,
            x="Approach",
            y="Delay Range 1",
            title="Average Delay by Approach (s)",
            text_auto='.1f',
            color="Delay Range 1",
            color_continuous_scale="RdYlGn_r"  # High delay = Red
        )
        fig_delay.update_layout(showlegend=False)
        st.plotly_chart(fig_delay, use_container_width=True)

    with col_chart_2:
        # Comparison of Volume vs Split Failures
        fig_combo = go.Figure()

        # Bar for Volume
        fig_combo.add_trace(go.Bar(
            x=df_app["Approach"],
            y=df_app["Turning Movement Range 1"],
            name="Volume",
            marker_color='rgb(55, 83, 109)'
        ))

        # Line for Split Failures
        fig_combo.add_trace(go.Scatter(
            x=df_app["Approach"],
            y=df_app["Split Failures Range 1"],
            name="Split Failure %",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color='rgb(219, 64, 82)', width=3)
        ))

        fig_combo.update_layout(
            title="Volume vs. Split Failures",
            yaxis=dict(title="Volume"),
            yaxis2=dict(title="Split Failures %", overlaying="y", side="right", tickformat=".0%"),
            legend=dict(x=0, y=1.2, orientation="h")
        )
        st.plotly_chart(fig_combo, use_container_width=True)

    # 3. Detailed Movement Analysis (By Movement Sheet)
    st.markdown("---")
    st.subheader("Movement Details")

    # Filter controls
    col_filter, col_display = st.columns([1, 3])

    with col_filter:
        st.write("**Filter Data**")
        selected_approaches = st.multiselect(
            "Select Approach",
            options=df_mov["Approach"].unique(),
            default=df_mov["Approach"].unique()
        )

        metric_to_plot = st.selectbox(
            "Select Metric to Visualize",
            ["Delay Range 1", "Arrivals On Green Range 1", "Split Failures Range 1", "Turning Movement Range 1"]
        )

    with col_display:
        filtered_df = df_mov[df_mov["Approach"].isin(selected_approaches)]

        if metric_to_plot == "Arrivals On Green Range 1" or metric_to_plot == "Split Failures Range 1":
            text_fmt = '.1%'
        else:
            text_fmt = '.1f'

        fig_mov = px.bar(
            filtered_df,
            x="Approach",
            y=metric_to_plot,
            color="Movement",
            barmode="group",
            title=f"{metric_to_plot} by Movement",
            text_auto=text_fmt
        )
        st.plotly_chart(fig_mov, use_container_width=True)

    # Raw Data Expander
    with st.expander("View Raw Data"):
        st.write("Intersection Data", df_int)
        st.write("Approach Data", df_app)
        st.write("Movement Data", df_mov)


if __name__ == "__main__":
    main()