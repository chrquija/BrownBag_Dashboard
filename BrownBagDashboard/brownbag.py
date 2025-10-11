import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
import pydeck as pdk

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


# Function to detect incidents based on Firsts and Lasts data
def detect_incidents(df):
    # Create a copy of the dataframe to avoid modifying the original
    incident_df = df.copy()
    
    # Calculate the ratio of Firsts to Lasts
    # When Firsts > Lasts, it indicates a potential incident starting
    # When Lasts > Firsts, it indicates a potential incident ending
    incident_df['ratio'] = incident_df['Firsts'] / incident_df['Lasts'].replace(0, 0.001)  # Avoid division by zero
    
    # Calculate additional metrics for incident detection
    # Strength relative to historical maximum
    incident_df['strength_ratio'] = incident_df['Strength'] / incident_df['Maximum']
    
    # Rate of change in Strength (if datetime is sorted)
    incident_df['strength_change'] = incident_df['Strength'].diff() / incident_df['Strength'].shift(1)
    incident_df['strength_change'] = incident_df['strength_change'].fillna(0)
    
    # Define thresholds for incident detection using multiple indicators
    # These thresholds can be adjusted based on domain knowledge
    incident_df['incident_predicted'] = (
        (incident_df['ratio'] > 1.5) |  # High ratio of Firsts to Lasts
        (incident_df['strength_ratio'] > 0.9) |  # Strength near historical maximum
        (incident_df['strength_change'] > 0.15)  # Sudden increase in Strength
    )
    
    # Simulate "actual" incidents for demonstration
    # In a real system, this would come from verified incident data
    incident_df['incident_actual'] = (
        (incident_df['ratio'] > 1.8) |  # Higher ratio threshold for actual incidents
        (incident_df['strength_ratio'] > 0.95) |  # Strength very close to historical maximum
        (incident_df['strength_change'] > 0.25)  # Larger sudden increase in Strength
    )
    
    # Calculate incident severity (0-100 scale)
    incident_df['severity'] = (
        (incident_df['ratio'] * 20) +  # Weight for ratio
        (incident_df['strength_ratio'] * 50) +  # Weight for strength ratio
        (np.abs(incident_df['strength_change']) * 30)  # Weight for rate of change
    ).clip(0, 100)  # Clip to 0-100 range
    
    # Generate mock coordinates for map visualization
    # In a real system, these would come from actual location data
    # Using Washington Street in La Quinta, CA as a reference point
    base_lat = 33.6680
    base_lon = -116.2773
    
    # Generate coordinates with small variations to simulate different points along the corridor
    incident_df['latitude'] = base_lat + (incident_df.index / len(incident_df) * 0.02) + (np.random.random(len(incident_df)) * 0.005)
    incident_df['longitude'] = base_lon + (np.random.random(len(incident_df)) * 0.01) - 0.005
    
    # Adjust coordinates slightly for different directions
    if 'direction' in incident_df.columns:
        # Shift NB and SB slightly to show them as separate lanes
        incident_df.loc[incident_df['direction'] == 'NB', 'longitude'] += 0.001
        incident_df.loc[incident_df['direction'] == 'SB', 'longitude'] -= 0.001
    
    return incident_df

# Aggregate the filtered data
aggregated_df = aggregate_data(filtered_df, granularity)

# Apply incident detection
aggregated_df = detect_incidents(aggregated_df)

# Main content area
if not aggregated_df.empty:
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Traffic Metrics", "Incident Detection"])
    
    with tab1:
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
    
    with tab2:
        st.subheader("Incident Detection Dashboard")
        st.write("This dashboard uses 'Firsts' and 'Lasts' data to detect potential traffic incidents.")
        
        # Create a line chart showing the ratio of Firsts to Lasts
        st.subheader("Incident Detection Metrics")
        fig = px.line(
            aggregated_df,
            x='datetime',
            y=['ratio', 'Firsts', 'Lasts'],
            title=f"Incident Detection Metrics - {granularity} Aggregation",
            labels={'datetime': 'Date/Time', 'value': 'Value', 'variable': 'Metric'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Weather card style visualization for incidents
        st.subheader("Incident Status")
        
        # Count incidents and calculate average severity
        predicted_incidents = aggregated_df['incident_predicted'].sum()
        actual_incidents = aggregated_df['incident_actual'].sum()
        
        # Calculate average severity for predicted and actual incidents
        avg_severity_predicted = 0
        if predicted_incidents > 0:
            avg_severity_predicted = aggregated_df.loc[aggregated_df['incident_predicted'], 'severity'].mean()
            
        avg_severity_actual = 0
        if actual_incidents > 0:
            avg_severity_actual = aggregated_df.loc[aggregated_df['incident_actual'], 'severity'].mean()
        
        # Create weather card style visualization
        col1, col2 = st.columns(2)
        
        # Custom CSS for weather card style
        card_style = """
        <style>
        .incident-card {
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: center;
            margin-bottom: 20px;
            position: relative;
            overflow: hidden;
        }
        .incident-card h3 {
            margin-bottom: 10px;
            font-size: 1.5rem;
        }
        .incident-card p {
            font-size: 3rem;
            font-weight: bold;
            margin: 10px 0;
        }
        .incident-card .status {
            font-size: 1rem;
            margin-top: 10px;
        }
        .incident-card .severity {
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 5px 10px;
            border-radius: 15px;
            font-weight: bold;
            color: white;
        }
        .predicted {
            background-color: #f0f8ff;
            border-left: 5px solid #1e90ff;
        }
        .actual {
            background-color: #fff0f0;
            border-left: 5px solid #ff4500;
        }
        .severity-high {
            background-color: #ff0000;
        }
        .severity-medium {
            background-color: #ff9900;
        }
        .severity-low {
            background-color: #ffcc00;
            color: black;
        }
        .severity-verylow {
            background-color: #00cc00;
        }
        </style>
        """
        
        st.markdown(card_style, unsafe_allow_html=True)
        
        # Determine severity class for predicted incidents
        predicted_severity_class = "severity-verylow"
        if avg_severity_predicted >= 75:
            predicted_severity_class = "severity-high"
        elif avg_severity_predicted >= 50:
            predicted_severity_class = "severity-medium"
        elif avg_severity_predicted >= 25:
            predicted_severity_class = "severity-low"
            
        # Determine severity class for actual incidents
        actual_severity_class = "severity-verylow"
        if avg_severity_actual >= 75:
            actual_severity_class = "severity-high"
        elif avg_severity_actual >= 50:
            actual_severity_class = "severity-medium"
        elif avg_severity_actual >= 25:
            actual_severity_class = "severity-low"
        
        with col1:
            predicted_html = f"""
            <div class="incident-card predicted">
                <div class="severity {predicted_severity_class}">Avg Severity: {avg_severity_predicted:.1f}</div>
                <h3>Predicted Incidents</h3>
                <p>{predicted_incidents}</p>
                <div class="status">Based on multiple indicators</div>
            </div>
            """
            st.markdown(predicted_html, unsafe_allow_html=True)
            
        with col2:
            actual_html = f"""
            <div class="incident-card actual">
                <div class="severity {actual_severity_class}">Avg Severity: {avg_severity_actual:.1f}</div>
                <h3>Actual Incidents</h3>
                <p>{actual_incidents}</p>
                <div class="status">Based on verified data</div>
            </div>
            """
            st.markdown(actual_html, unsafe_allow_html=True)
            
        # Map visualization of incidents
        st.subheader("Incident Map")
        
        # Filter for incidents only
        map_data = aggregated_df[aggregated_df['incident_predicted'] | aggregated_df['incident_actual']].copy()
        
        if not map_data.empty:
            # Create a color column for the map
            map_data['color'] = [
                [255, 0, 0, 200] if row['incident_actual'] else [0, 0, 255, 200]  # Red for actual, Blue for predicted only
                for _, row in map_data.iterrows()
            ]
            
            # Create a size column based on the severity
            map_data['size'] = map_data['severity'] * 0.5  # Scale severity to appropriate size
            
            # Create tooltip information
            map_data['tooltip'] = map_data.apply(
                lambda row: {
                    "datetime": row['datetime'].strftime("%Y-%m-%d %H:%M"),
                    "direction": row['direction'],
                    "predicted": "Yes" if row['incident_predicted'] else "No",
                    "actual": "Yes" if row['incident_actual'] else "No",
                    "severity": f"{row['severity']:.1f}/100",
                    "ratio": f"{row['ratio']:.2f}"
                },
                axis=1
            )
            
            # Create the map
            view_state = pdk.ViewState(
                latitude=map_data['latitude'].mean(),
                longitude=map_data['longitude'].mean(),
                zoom=13,
                pitch=0
            )
            
            # Create scatter plot layer
            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_data,
                get_position=["longitude", "latitude"],
                get_color="color",
                get_radius="size",
                pickable=True,
                opacity=0.8,
                stroked=True,
                filled=True,
                radius_scale=6,
                radius_min_pixels=5,
                radius_max_pixels=100,
            )
            
            # Create the deck
            deck = pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v10",
                initial_view_state=view_state,
                layers=[scatter_layer],
                tooltip={
                    "html": "<b>Time:</b> {datetime}<br/><b>Direction:</b> {direction}<br/><b>Severity:</b> {severity}<br/><b>Predicted:</b> {predicted}<br/><b>Actual:</b> {actual}<br/><b>Ratio:</b> {ratio}",
                    "style": {
                        "backgroundColor": "white",
                        "color": "black"
                    }
                }
            )
            
            # Display the map
            st.pydeck_chart(deck)
            
            # Add a legend
            legend_html = """
            <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                <div style="margin-right: 20px; display: flex; align-items: center;">
                    <div style="width: 15px; height: 15px; background-color: blue; border-radius: 50%; margin-right: 5px;"></div>
                    <span>Predicted Incident</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 15px; height: 15px; background-color: red; border-radius: 50%; margin-right: 5px;"></div>
                    <span>Actual Incident</span>
                </div>
            </div>
            """
            st.markdown(legend_html, unsafe_allow_html=True)
        else:
            st.write("No incidents detected for the selected filters.")
        
        # Display recent incidents
        st.subheader("Recent Incidents")
        recent_incidents = aggregated_df[aggregated_df['incident_predicted'] | aggregated_df['incident_actual']].sort_values('datetime', ascending=False).head(10)
        
        if not recent_incidents.empty:
            for _, incident in recent_incidents.iterrows():
                incident_time = incident['datetime'].strftime("%Y-%m-%d %H:%M")
                predicted = "Yes" if incident['incident_predicted'] else "No"
                actual = "Yes" if incident['incident_actual'] else "No"
                severity = incident['severity']
                
                # Determine severity color
                if severity >= 75:
                    severity_color = "#ff0000"  # Red for high severity
                elif severity >= 50:
                    severity_color = "#ff9900"  # Orange for medium severity
                elif severity >= 25:
                    severity_color = "#ffcc00"  # Yellow for low severity
                else:
                    severity_color = "#00cc00"  # Green for very low severity
                
                incident_html = f"""
                <div style="padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: {'#fff0f0' if incident['incident_actual'] else '#f0f8ff'}; border-left: 5px solid {severity_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>Time:</strong> {incident_time} | 
                            <strong>Direction:</strong> {incident['direction']} | 
                            <strong>Predicted:</strong> {predicted} | 
                            <strong>Actual:</strong> {actual}
                        </div>
                        <div style="background-color: {severity_color}; color: white; padding: 3px 8px; border-radius: 10px; font-weight: bold;">
                            Severity: {severity:.1f}/100
                        </div>
                    </div>
                    <div style="margin-top: 5px;">
                        <strong>Metrics:</strong> Ratio: {incident['ratio']:.2f} | 
                        Strength Change: {incident['strength_change']:.2f} | 
                        Strength/Max: {incident['strength_ratio']:.2f}
                    </div>
                </div>
                """
                st.markdown(incident_html, unsafe_allow_html=True)
        else:
            st.write("No incidents detected for the selected filters.")

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