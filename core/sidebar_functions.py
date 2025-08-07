import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

#DATA LOADING FOR ALL DATASETS
@st.cache_data
def load_traffic_data():
    """
    Load and combine all corridor traffic data from GitHub
    """
    # Define all data sources with their URLs and descriptive names
    data_sources = {
        "Avenue 52 → Calle Tampico": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/1_2_LONG_NSB_Ave52_CalleTampico_WashSt_1hr_septojuly.csv",
        "Calle Tampico → Village Shopping Ctr": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/2_3_LONG_NSB_CalleTampico_VillageShoppingCtr_WashSt_1hr_septojuly.csv",
        "Village Shopping Ctr → Avenue 50": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/3_4_LONG_NSB_VillageShoppingCtr_Avenue50_WashSt_1hr_septojuly.csv",
        "Avenue 50 → Sagebrush Ave": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/4_5_LONG_NSB_Ave50_SagebrushAve_WashSt_1hr_septojuly.csv",
        "Sagebrush Ave → Eisenhower Dr": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/5_6_LONG_NSB_SagebrushAve_EisenhowerDr_WashSt_1hr_septojuly.csv",
        "Eisenhower Dr → Avenue 48": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/6_7_LONG_NSB_EisenhowerDr_Avenue48_WashSt_1hr_septojuly.csv",
        "Avenue 48 → Avenue 47": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/7_8_LONG_NSB_Ave48_Ave47_WashSt_1hr_septojuly.csv",
        "Avenue 47 → Point Happy Simon": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/8_9_LONG_NSB_Ave47_PointHappySimon_WashSt_1hr_septojuly.csv",
        "Point Happy Simon → Hwy 111": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/9_10_LONG_NSB_PointHappySimon_WashSt_1hr_septojuly.csv"
    }
    
    # Load and combine all datasets
    all_data = []
    for segment_name, url in data_sources.items():
        try:
            df = pd.read_csv(url)
            df['segment_name'] = segment_name  # Add readable segment name
            all_data.append(df)
        except Exception as e:
            st.error(f"Error loading {segment_name}: {e}")
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Convert datetime
        combined_df['local_datetime'] = pd.to_datetime(combined_df['local_datetime'])
        
        # Sort by datetime
        combined_df = combined_df.sort_values('local_datetime').reset_index(drop=True)
        
        return combined_df
    else:
        return pd.DataFrame()

@st.cache_data
def load_volume_data():
    """
    Load consolidated volume data for all Washington Street intersections
    """
    volume_url = "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/VOLUME/KMOB_LONG/LONG_MASTER_Avenue52_to_Avenue47_1hr_NS_VOLUME_OctoberTOJune.csv"
    
    try:
        volume_df = pd.read_csv(volume_url)
        
        # Convert datetime
        volume_df['local_datetime'] = pd.to_datetime(volume_df['local_datetime'])
        
        # Sort by datetime
        volume_df = volume_df.sort_values('local_datetime').reset_index(drop=True)
        
        # Create proper intersection names from intersection_id
        volume_df['intersection_name'] = (
            volume_df['intersection_id']
            .str.replace('_', ' ')  # Replace underscores with spaces
            .str.replace('Washington St and ', 'Washington St & ')  # Fix the main intersection format
            .str.replace(' and ', ' & ')  # Replace any remaining 'and' with '&'
        )
        
        # Create a sorting order for intersections (from south to north along Washington St)
        intersection_order = {
            'Washington St & Avenue52': 1,
            'Washington St & Calle Tampico': 2, 
            'Washington St & Village Shop Ctr': 3,
            'Washington St & Avenue50': 4,
            'Washington St & Sagebrush Ave': 5,
            'Washington St & Eisenhower': 6,
            'Washington St & Ave48': 7,
            'Washington St & Ave47': 8
        }
        
        # Add sorting column
        volume_df['sort_order'] = volume_df['intersection_name'].map(intersection_order)
        
        # Sort by the order (fill NaN with high number to put unknowns at end)
        volume_df['sort_order'] = volume_df['sort_order'].fillna(999)
        volume_df = volume_df.sort_values('sort_order')
        
        # Drop the sorting column
        volume_df = volume_df.drop('sort_order', axis=1)
        
        return volume_df
        
    except Exception as e:
        st.error(f"Error loading volume data: {e}")
        return pd.DataFrame()

#FUNCTIONS FOR DATE RANGE FUNCTIONALITY
def process_traffic_data(df, date_range, granularity, time_filter=None, start_hour=None, end_hour=None):
    """
    Process traffic data based on date range and granularity selections
    """
    # Convert datetime if not already done
    df['local_datetime'] = pd.to_datetime(df['local_datetime'])

    # Filter by date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[
            (df['local_datetime'].dt.date >= start_date) &
            (df['local_datetime'].dt.date <= end_date)
            ]

    # Apply time filters for hourly data
    if granularity == "Hourly" and time_filter:
        if time_filter == "Peak Hours (7-9 AM, 4-6 PM)":
            df = df[
                (df['local_datetime'].dt.hour.between(7, 9)) |
                (df['local_datetime'].dt.hour.between(16, 18))
                ]
        elif time_filter == "AM Peak (7-9 AM)":
            df = df[df['local_datetime'].dt.hour.between(7, 9)]
        elif time_filter == "PM Peak (4-6 PM)":
            df = df[df['local_datetime'].dt.hour.between(16, 18)]
        elif time_filter == "Off-Peak":
            df = df[
                ~(df['local_datetime'].dt.hour.between(7, 9)) &
                ~(df['local_datetime'].dt.hour.between(16, 18))
                ]
        elif time_filter == "Custom Range" and start_hour is not None and end_hour is not None:
            df = df[df['local_datetime'].dt.hour.between(start_hour, end_hour - 1)]

    # Determine data type and aggregate accordingly
    if 'segment_name' in df.columns:  # Corridor data (delay/speed/travel time)
        if granularity == "Daily":
            df['date_group'] = df['local_datetime'].dt.date
            grouped = df.groupby(['date_group', 'corridor_id', 'direction', 'segment_name']).agg({
                'average_delay': 'mean',
                'average_traveltime': 'mean',
                'average_speed': 'mean'
            }).reset_index()
            grouped['local_datetime'] = pd.to_datetime(grouped['date_group'])

        elif granularity == "Weekly":
            df['week_group'] = df['local_datetime'].dt.to_period('W').dt.start_time
            grouped = df.groupby(['week_group', 'corridor_id', 'direction', 'segment_name']).agg({
                'average_delay': 'mean',
                'average_traveltime': 'mean',
                'average_speed': 'mean'
            }).reset_index()
            grouped['local_datetime'] = grouped['week_group']

        elif granularity == "Monthly":
            df['month_group'] = df['local_datetime'].dt.to_period('M').dt.start_time
            grouped = df.groupby(['month_group', 'corridor_id', 'direction', 'segment_name']).agg({
                'average_delay': 'mean',
                'average_traveltime': 'mean',
                'average_speed': 'mean'
            }).reset_index()
            grouped['local_datetime'] = grouped['month_group']

        else:  # Hourly - no aggregation needed
            grouped = df
            
    elif 'intersection_id' in df.columns:  # Volume data
        if granularity == "Daily":
            df['date_group'] = df['local_datetime'].dt.date
            grouped = df.groupby(['date_group', 'intersection_id', 'direction', 'intersection_name']).agg({
                'total_volume': 'sum'
            }).reset_index()
            grouped['local_datetime'] = pd.to_datetime(grouped['date_group'])

        elif granularity == "Weekly":
            df['week_group'] = df['local_datetime'].dt.to_period('W').dt.start_time
            grouped = df.groupby(['week_group', 'intersection_id', 'direction', 'intersection_name']).agg({
                'total_volume': 'sum'
            }).reset_index()
            grouped['local_datetime'] = grouped['week_group']

        elif granularity == "Monthly":
            df['month_group'] = df['local_datetime'].dt.to_period('M').dt.start_time
            grouped = df.groupby(['month_group', 'intersection_id', 'direction', 'intersection_name']).agg({
                'total_volume': 'sum'
            }).reset_index()
            grouped['local_datetime'] = grouped['month_group']

        else:  # Hourly - no aggregation needed
            grouped = df
    
    else:
        # Fallback - just return filtered data
        grouped = df

    return grouped