import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

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

    # Aggregate based on granularity
    if granularity == "Daily":
        df['date_group'] = df['local_datetime'].dt.date
        grouped = df.groupby(['date_group', 'corridor_id', 'direction']).agg({
            'average_delay': 'mean',
            'average_traveltime': 'mean',
            'average_speed': 'mean'
        }).reset_index()
        grouped['local_datetime'] = pd.to_datetime(grouped['date_group'])

    elif granularity == "Weekly":
        df['week_group'] = df['local_datetime'].dt.to_period('W').dt.start_time
        grouped = df.groupby(['week_group', 'corridor_id', 'direction']).agg({
            'average_delay': 'mean',
            'average_traveltime': 'mean',
            'average_speed': 'mean'
        }).reset_index()
        grouped['local_datetime'] = grouped['week_group']

    elif granularity == "Monthly":
        df['month_group'] = df['local_datetime'].dt.to_period('M').dt.start_time
        grouped = df.groupby(['month_group', 'corridor_id', 'direction']).agg({
            'average_delay': 'mean',
            'average_traveltime': 'mean',
            'average_speed': 'mean'
        }).reset_index()
        grouped['local_datetime'] = grouped['month_group']

    else:  # Hourly - no aggregation needed
        grouped = df

    return grouped