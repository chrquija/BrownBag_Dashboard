import streamlit as st
import pandas as pd
import pydeck as pdk


def render_map(
    latitude: float,
    longitude: float,
    *,
    height: int = 620,
    zoom: int = 14,
    label: str = "Intersection",
):
    """Render an interactive map centered on the given coordinates with a label.

    - Places a marker and text label at the intersection location
    - Uses OpenStreetMap tiles (no Mapbox token required)
    - Intended to be used in the right-hand column of the dashboard
    """

    # Single-point dataframe for layers
    df = pd.DataFrame({
        "lat": [latitude],
        "lon": [longitude],
        "name": [label],
    })

    view_state = pdk.ViewState(
        latitude=latitude,
        longitude=longitude,
        zoom=zoom,
        pitch=0,
        bearing=0,
    )

    # Base map without requiring a Mapbox token
    tile_layer = pdk.Layer(
        "TileLayer",
        data="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
    )

    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_radius=50,
        get_fill_color=[230, 57, 70, 220],  # red marker
        pickable=True,
    )

    text_layer = pdk.Layer(
        "TextLayer",
        data=df,
        get_position="[lon, lat]",
        get_text="name",
        get_color=[255, 255, 255, 230],
        get_size=16,
        get_alignment_baseline="bottom",
        get_pixel_offset=[0, 14],
        background=True,
        background_color=[30, 30, 30, 180],
    )

    deck = pdk.Deck(
        initial_view_state=view_state,
        map_style=None,
        layers=[tile_layer, point_layer, text_layer],
        tooltip={"text": "{name}\n({lat}, {lon})"},
    )

    st.subheader("Corridor Map")
    st.pydeck_chart(deck, use_container_width=True, height=height)
