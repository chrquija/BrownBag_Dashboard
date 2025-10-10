import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Set page config
st.set_page_config(
    page_title="BrownBag Dashboard",
    page_icon="📊",
    layout="wide"
)
# Title
st.title("BrownBag Dashboard")


# Create sample data (you'll replace this with your actual data loading)
@st.cache_data
def load_data():
    # Load data from GitHub raw URL
    try:
        url = "https://raw.githubusercontent.com/chrquija/BrownBag_Dashboard/refs/heads/main/FULL_TravelTime_North111_SB_NB.csv"
        df = pd.read_csv(url)

        # If the CSV has different column names, you may need to rename them
        # For now, let's assume it has the columns we need or we'll adapt

        # Convert datetime column if it exists (adjust column name as needed)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif 'Timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['Timestamp'])
        elif 'Date' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'])
        else:
            # If no datetime column, create one with current date range
            dates = pd.date_range(start='2025-08-20', end='2025-09-19', periods=len(df))
            df['datetime'] = dates

        # Ensure we have the required columns, create them if missing
        required_columns = ['corridor_id', 'Corridor_segment', 'direction', 'metric', 'Strength', 'Firsts', 'Lasts',
                            'Minimum', 'Maximum']

        for col in required_columns:
            if col not in df.columns:
                if col == 'corridor_id':
                    df[col] = 'Washington Street'
                elif col == 'Corridor_segment':
                    df[col] = 'Hwy111 to Country Club Drive'
                elif col == 'direction':
                    df[col] = np.random.choice(['NB', 'SB'], size=len(df))
                elif col == 'metric':
                    df[col] = 'TravelTime'
                else:
                    # For numeric columns, use random data if not present
                    if col == 'Strength':
                        df[col] = np.random.uniform(20, 100, size=len(df))
                    elif col in ['Firsts', 'Lasts']:
                        df[col] = np.random.randint(1, 20, size=len(df))
                    elif col == 'Minimum':
                        df[col] = np.random.uniform(10, 30, size=len(df))
                    elif col == 'Maximum':
                        df[col] = np.random.uniform(40, 80, size=len(df))

        return df

    except Exception as e:
        st.error(f"Error loading data from GitHub: {e}")
        st.info("Falling back to sample data...")

        # Fallback to sample data if URL fails
        dates = pd.date_range(start='2025-08-20', end='2025-09-19', freq='15min')
        data = []
        for date in dates:
            data.append({
                'datetime': date,
                'corridor_id': 'Washington Street',
                'Corridor_segment': 'Hwy111 to Country Club Drive',
                'direction': np.random.choice(['NB', 'SB']),
                'metric': 'TravelTime',
                'Strength': np.random.uniform(20, 100),
                'Firsts': np.random.randint(1, 20),
                'Lasts': np.random.randint(1, 20),
                'Minimum': np.random.uniform(10, 30),
                'Maximum': np.random.uniform(40, 80)
            })
        return pd.DataFrame(data)


# Load data
df = load_data()

# Sidebar
st.sidebar.header("Filters")

# Corridor selection
corridor_options = df['corridor_id'].unique()
selected_corridor = st.sidebar.selectbox(
    "Select Corridor",
    options=corridor_options,
    index=0
)

# Date and Time filter
st.sidebar.subheader("Date And Time")

# Date range buttons
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    last_7 = st.button("Last 7 Days")
with col2:
    last_30 = st.button("Last 30 Days")
with col3:
    full_range = st.button("Full Range")

# Custom date range
st.sidebar.write("Custom Date Range")
start_date = st.sidebar.date_input(
    "Start Date",
    value=df['datetime'].min().date(),
    min_value=df['datetime'].min().date(),
    max_value=df['datetime'].max().date()
)
end_date = st.sidebar.date_input(
    "End Date",
    value=df['datetime'].max().date(),
    min_value=df['datetime'].min().date(),
    max_value=df['datetime'].max().date()
)

# Granularity selection
st.sidebar.subheader("Granularity")
granularity = st.sidebar.selectbox(
    "Data Aggregation",
    options=["15 min", "Hourly", "Daily", "Weekly", "Monthly"]
)

# Direction filter
st.sidebar.subheader("Direction Filter")
direction_options = ["All Directions"] + list(df['direction'].unique())
selected_direction = st.sidebar.selectbox(
    "Direction",
    options=direction_options
)

# Apply filters
filtered_df = df.copy()

# Filter by corridor
filtered_df = filtered_df[filtered_df['corridor_id'] == selected_corridor]

# Apply date range filter
if last_7:
    end_dt = df['datetime'].max()
    start_dt = end_dt - timedelta(days=7)
    filtered_df = filtered_df[
        (filtered_df['datetime'] >= start_dt) &
        (filtered_df['datetime'] <= end_dt)
        ]
elif last_30:
    end_dt = df['datetime'].max()
    start_dt = end_dt - timedelta(days=30)
    filtered_df = filtered_df[
        (filtered_df['datetime'] >= start_dt) &
        (filtered_df['datetime'] <= end_dt)
        ]
elif full_range:
    # Use full range (no additional filtering needed)
    pass
else:
    # Use custom date range
    start_dt = pd.Timestamp.combine(start_date, datetime.min.time())
    end_dt = pd.Timestamp.combine(end_date, datetime.max.time())
    filtered_df = filtered_df[
        (filtered_df['datetime'] >= start_dt) &
        (filtered_df['datetime'] <= end_dt)
        ]

# Filter by direction
if selected_direction != "All Directions":
    filtered_df = filtered_df[filtered_df['direction'] == selected_direction]


# Aggregate data based on granularity
def aggregate_data(df, granularity):
    if granularity == "15 min":
        freq = '15min'
    elif granularity == "Hourly":
        freq = 'H'
    elif granularity == "Daily":
        freq = 'D'
    elif granularity == "Weekly":
        freq = 'W'
    elif granularity == "Monthly":
        freq = 'M'

    # Group by time period and direction (if multiple directions exist)
    if selected_direction == "All Directions":
        grouped = df.groupby([pd.Grouper(key='datetime', freq=freq), 'direction']).agg({
            'Strength': 'mean',
            'Firsts': 'sum',
            'Lasts': 'sum',
            'Minimum': 'min',
            'Maximum': 'max'
        }).reset_index()
    else:
        grouped = df.groupby(pd.Grouper(key='datetime', freq=freq)).agg({
            'Strength': 'mean',
            'Firsts': 'sum',
            'Lasts': 'sum',
            'Minimum': 'min',
            'Maximum': 'max'
        }).reset_index()
        grouped['direction'] = selected_direction

    return grouped


# Aggregate the filtered data
aggregated_df = aggregate_data(filtered_df, granularity)

# Main content area
if not aggregated_df.empty:
    # Create line chart for Strength values
    st.subheader("Strength Values Over Time")

    if selected_direction == "All Directions" and len(aggregated_df['direction'].unique()) > 1:
        fig = px.line(
            aggregated_df,
            x='datetime',
            y='Strength',
            color='direction',
            title=f"Strength Values - {granularity} Aggregation",
            labels={'datetime': 'Date/Time', 'Strength': 'Strength Value'}
        )
    else:
        fig = px.line(
            aggregated_df,
            x='datetime',
            y='Strength',
            title=f"Strength Values - {granularity} Aggregation",
            labels={'datetime': 'Date/Time', 'Strength': 'Strength Value'}
        )

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Strength", f"{aggregated_df['Strength'].mean():.2f}")
    with col2:
        st.metric("Total Records", len(filtered_df))
    with col3:
        st.metric("Max Strength", f"{aggregated_df['Strength'].max():.2f}")
    with col4:
        st.metric("Min Strength", f"{aggregated_df['Strength'].min():.2f}")

    # Raw data expander
    with st.expander("View Raw Data"):
        st.dataframe(
            filtered_df.sort_values('datetime', ascending=False),
            use_container_width=True
        )
else:
    st.warning("No data available for the selected filters.")

# Footer
st.markdown("---")
st.markdown("BrownBag Dashboard - Data Analysis Tool")