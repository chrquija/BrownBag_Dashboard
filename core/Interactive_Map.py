import pandas as pd

data = [
    {"name": "Ave 52", "lat": 33.66109, "lon": -116.29970},
    {"name": "Avenida Nuestra", "lat": 33.66578, "lon": -116.29970},
    {"name": "Calle Tampico", "lat": 33.67629, "lon": -116.29970},
    # Add more intersections as needed
]
df = pd.DataFrame(data)

import streamlit as st
import pydeck as pdk

# DataFrame from above (df)

# Create a line for the corridor
corridor_line = pdk.Layer(
    "LineLayer",
    data=df,
    get_source_position="[lon, lat]",
    get_target_position="[lon, lat]",
    get_color=[0, 150, 255],
    get_width=4,
    pickable=False,
)

# Create dots for intersections
dots = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position="[lon, lat]",
    get_color=[255, 0, 0],
    get_radius=40,
    pickable=True,
)

# Pydeck map
view_state = pdk.ViewState(
    latitude=df['lat'].mean(),
    longitude=df['lon'].mean(),
    zoom=13,
    pitch=0
)

st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=view_state,
        layers=[corridor_line, dots],
        tooltip={"text": "{name}"},
    )
)
