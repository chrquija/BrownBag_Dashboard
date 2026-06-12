import streamlit as st
import pandas as pd
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium


def render_map(
    latitude: float = None,
    longitude: float = None,
    *,
    height: int = 620,
    zoom: int = None,
    label: str = "Intersection",
    registry: list = None,
    use_satellite: bool = False,
    show_labels: bool = True,
    highlight_labels: list = None,
    study_period: str = None,
    intersections: list = None,
    segments: pd.DataFrame = None,
    intersections_title: str = "Intersections",
    label_size: int = 11,
):
    """Render an interactive map centered on the given coordinates using Folium.

    - Places circle markers for all intersections
    - Highlights current/corridor intersections with larger red markers and text labels
    - Supports satellite mode via ESRI World Imagery
    - Intended to be used in the right-hand column of the dashboard
    """

    # Prepare data for all points
    if registry:
        df = pd.DataFrame(registry)
        # Ensure 'name' is present for tooltips and labels
        if 'label' in df.columns and 'name' not in df.columns:
            df['name'] = df['label']
    else:
        # Fallback to single-point if no registry provided
        if latitude is None or longitude is None:
            st.warning("Map coordinates not provided.")
            return

        df = pd.DataFrame({
            "lat": [latitude],
            "lon": [longitude],
            "name": [label],
        })

    if highlight_labels is None:
        highlight_labels = [label]

    # Filter points to focus on highlighted ones for initial center/zoom
    focus_df = df[df['name'].isin(highlight_labels)] if highlight_labels else df
    if focus_df.empty:
        focus_df = df

    # Dynamic center calculation
    if latitude is None or longitude is None:
        if not focus_df.empty:
            latitude = focus_df['lat'].mean()
            longitude = focus_df['lon'].mean()
        else:
            latitude, longitude = 33.7, -116.3 # Fallback center
    
    # Heuristic for zoom if not provided
    if zoom is None:
        if len(focus_df) > 1:
            lat_span = focus_df['lat'].max() - focus_df['lat'].min()
            lon_span = focus_df['lon'].max() - focus_df['lon'].min()
            span = max(lat_span, lon_span)
            
            if span > 0.1: zoom = 11
            elif span > 0.05: zoom = 12
            elif span > 0.02: zoom = 13
            elif span > 0.005: zoom = 14
            else: zoom = 15
        else:
            zoom = 15

    # Define tiles and attribution
    if use_satellite:
        tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        attr = "Esri World Imagery"
    else:
        # Carto Voyager tiles (via cartodb.com)
        tiles = "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
        attr = "&copy; <a href=\"https://www.openstreetmap.org/copyright\">OpenStreetMap</a> contributors &copy; <a href=\"https://carto.com/attributions\">CARTO</a>"

    # Create the Folium Map
    m = folium.Map(
        control_scale=True,
        location=[latitude, longitude],
        zoom_start=zoom,
        tiles=tiles,
        attr=attr,
        height=height
    )
    Fullscreen().add_to(m)

    # Inject CSS into Folium map to affect tooltips inside its iframe
    m.get_root().header.add_child(folium.Element("""
        <style>
        .leaflet-tooltip {
            font-size: 18px !important;
            font-family: sans-serif !important;
        }
        </style>
    """))

    # Add segment polylines if provided
    if segments is not None and not segments.empty:
        for subseg, group in segments.groupby("Subsegment", sort=False):
            # Get from/to labels
            from_label = group["From Intersection"].iloc[0]
            to_label = group["To Intersection"].iloc[0]
            
            # Lookup coordinates in the registry (df)
            from_pt = df[df['name'] == from_label]
            to_pt = df[df['name'] == to_label]
            
            if not from_pt.empty and not to_pt.empty:
                from_coords = [from_pt['lat'].iloc[0], from_pt['lon'].iloc[0]]
                to_coords = [to_pt['lat'].iloc[0], to_pt['lon'].iloc[0]]
                
                # Prepare tooltip content
                two_way_adt = group["Two-Way Segment ADT"].iloc[0]
                date_range = group["DateRange"].iloc[0]
                
                # Build direction info
                dir_info = ""
                for i, (_, row) in enumerate(group.iterrows()):
                    dir_letter = chr(65 + i) # A, B, ...
                    dir_label = row["Direction Label"]
                    adt = row["Directional ADT"]
                    dir_info += f"<b>Direction {dir_letter}:</b> {dir_label} = {adt:,.0f} vehicles/day<br>"
                
                tooltip_html = f"""
                    <div style="font-family: sans-serif; font-size: 14pt; min-width: 300px; padding: 5px;">
                        <div style="font-weight: bold; border-bottom: 2px solid #1f4582; margin-bottom: 8px; padding-bottom: 3px; color: #1f4582;">
                            {subseg}
                        </div>
                        <div style="margin-bottom: 4px;"><b>Two-Way Segment ADT:</b> {two_way_adt:,.0f} vehicles/day</div>
                        <div style="margin-bottom: 8px; font-size: 12pt;">{dir_info}</div>
                        <div style="font-size: 11pt; opacity: 0.8; border-top: 1px solid #eee; padding-top: 4px;">
                            <b>Study Period:</b> {date_range}
                        </div>
                    </div>
                """
                
                folium.PolyLine(
                    locations=[from_coords, to_coords],
                    color='#1f4582',
                    weight=10,
                    opacity=0.4,
                    tooltip=folium.Tooltip(tooltip_html, sticky=True)
                ).add_to(m)

    # Add markers for all intersections
    for _, row in df.iterrows():
        is_highlighted = row['name'] in highlight_labels
        is_primary = row['name'] == label
        
        color = '#E63946' if is_highlighted else '#1F4582' # Red vs Blue
        radius = 12 if is_primary else 10 if is_highlighted else 6
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=radius,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            tooltip=row['name'] if show_labels else None,
            popup=folium.Popup(row['name'], parse_html=True)
        ).add_to(m)
        
        # Always visible text labels for highlighted intersections
        if is_highlighted and show_labels:
            label_color = "white" if use_satellite else "#E63946"
            label_shadow = "1px 1px 2px black" if use_satellite else "none"
            label_bg = "rgba(0,0,0,0.4)" if use_satellite else "transparent"
            
            folium.map.Marker(
                [row['lat'], row['lon']],
                icon=folium.DivIcon(
                    icon_size=(200, 36),
                    icon_anchor=(100, 42), # Position above marker
                    html=f"""
                        <div style="
                            font-family: sans-serif; 
                            font-size: {label_size}pt; 
                            color: {label_color}; 
                            font-weight: bold; 
                            text-shadow: {label_shadow};
                            background: {label_bg};
                            padding: 2px 5px;
                            border-radius: 4px;
                            text-align: center;
                            pointer-events: none;
                        ">
                            {row['name']}
                        </div>
                    """
                )
            ).add_to(m)

    # Special landmark: Indian Wells Tennis Garden
    folium.Marker(
        [33.723664, -116.305232],
        icon=folium.Icon(color='orange', icon='star', prefix='fa'), # Using FontAwesome star
        tooltip="Indian Wells Tennis Garden",
        popup="Indian Wells Tennis Garden"
    ).add_to(m)

    # Special landmark: Coachella Valley Music Festival
    folium.Marker(
        [33.6784, -116.2372],
        icon=folium.Icon(color='purple', icon='music', prefix='fa'), # Using FontAwesome music icon
        tooltip="Coachella Valley Music Festival",
        popup="Coachella Valley Music Festival (Empire Polo Club)"
    ).add_to(m)

    # Festival Area Square (Empire Polo Club grounds)
    folium.Rectangle(
        bounds=[[33.671, -116.247], [33.685, -116.230]],
        color="purple",
        weight=2,
        fill=True,
        fill_opacity=0.1,
        tooltip="Festival Grounds (Empire Polo Club)",
        popup="Empire Polo Club grounds"
    ).add_to(m)


    # Add Legend to the map
    legend_html = """
    <div style="
        position: fixed; 
        top: 10px; 
        right: 10px; 
        width: auto; 
        height: auto; 
        z-index:9999; 
        font-size:12px;
        background-color: white; 
        padding: 10px; 
        border: 2px solid grey; 
        border-radius: 6px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    ">
        <div style="font-weight: bold; margin-bottom: 5px; border-bottom: 1px solid #ccc; padding-bottom: 3px;">Map Legend</div>
        <div style="display: grid; grid-template-columns: 1fr; gap: 5px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="height: 12px; width: 12px; background-color: rgb(230, 57, 70); border-radius: 50%; display: inline-block; border: 1px solid white; box-shadow: 0 0 0 1px rgb(230, 57, 70);"></span>
                <span style="font-weight: 600;">Active Intersection(s)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="height: 12px; width: 12px; background-color: rgb(31, 69, 130); border-radius: 50%; display: inline-block; border: 1px solid white; box-shadow: 0 0 0 1px rgb(31, 69, 130);"></span>
                <span style="font-weight: 600;">Other Intersections</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.1rem; line-height: 1; color: orange;">★</span>
                <span style="font-weight: 600;">Tennis Garden</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.1rem; line-height: 1; color: purple;">♫</span>
                <span style="font-weight: 600;">Music Festival</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="height: 12px; width: 12px; border: 2px solid purple; background-color: rgba(128, 0, 128, 0.1); display: inline-block;"></span>
                <span style="font-weight: 600;">Festival Grounds</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="height: 4px; width: 20px; background-color: rgba(31, 69, 130, 0.4); display: inline-block; border-radius: 2px;"></span>
                <span style="font-weight: 600;">Segment ADT (Hover)</span>
            </div>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Display Study Period and Intersections KPIs above the map
    if study_period or intersections:
        # Build the HTML string without leading indentation to avoid Markdown code block triggers
        html = f'<div style="background: var(--secondary-background-color); padding: 18px 14px; border-radius: 14px; border: 3px solid #1f4582; margin-bottom: 16px; text-align: center; box-shadow: 0 6px 12px rgba(0,0,0,0.1);">'
        
        if study_period:
            html += f'<div style="font-size: 0.95rem; color: #1f4582; text-transform: uppercase; font-weight: 900; letter-spacing: 2px; margin-bottom: 6px;">Study Period</div>'
            html += f'<div style="font-size: 1.6rem; font-weight: 900; color: var(--text-color); line-height: 1.2; margin-bottom: {16 if intersections else 0}px; letter-spacing: -0.5px;">{study_period}</div>'
        
        if study_period and intersections:
            html += "<div style='height: 2px; background: #1f4582; opacity: 0.2; margin: 16px 0;'></div>"
            
        if intersections:
            html += f'<div style="font-size: 0.95rem; color: #1f4582; text-transform: uppercase; font-weight: 900; letter-spacing: 2px; margin-bottom: 12px;">{intersections_title}</div>'
            html += f'<div style="text-align: left; max-height: 200px; overflow-y: auto; padding: 0 5px;">'
            for item in intersections:
                # Handle both simple strings and dictionaries for backward compatibility
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    url = item.get("url")
                else:
                    name = item
                    url = None
                
                if url:
                    html += f'<a href="{url}" target="_blank" style="text-decoration: none;">'
                    html += f'<div style="margin-bottom: 8px; padding: 10px 14px; background: rgba(31, 69, 130, 0.05); border-radius: 8px; font-weight: 700; font-size: 1.1rem; color: #1f4582; border-left: 4px solid #1f4582; line-height: 1.2; transition: all 0.2s; cursor: pointer;" '
                    html += f'onmouseover="this.style.background=\'rgba(31, 69, 130, 0.1)\'; this.style.transform=\'translateX(2px)\'" '
                    html += f'onmouseout="this.style.background=\'rgba(31, 69, 130, 0.05)\'; this.style.transform=\'translateX(0)\'">'
                    html += f'{name}</div></a>'
                else:
                    html += f"<div style='margin-bottom: 8px; padding: 10px 14px; background: rgba(31, 69, 130, 0.05); border-radius: 8px; font-weight: 700; font-size: 1.1rem; color: var(--text-color); border-left: 4px solid #1f4582; line-height: 1.2;'>{name}</div>"
            html += '</div>'
            
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    # Render Folium in Streamlit
    st_folium(
        m,
        width="100%",
        height=height,
        key=f"map-{use_satellite}-{latitude}-{longitude}-{zoom}-{label}",
        returned_objects=[] # We don't need any data back from the map
    )
