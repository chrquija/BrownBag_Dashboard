import pandas as pd
import streamlit as st
import pydeck as pdk

# Sample data for intersections
data = [
    {"name": "Ave 52", "lat": 33.66109, "lon": -116.29970},
    {"name": "Avenida Nuestra", "lat": 33.66578, "lon": -116.29970},
    {"name": "Calle Tampico", "lat": 33.67629, "lon": -116.29970},
    # Add more intersections as needed
]
df = pd.DataFrame(data)

st.title("Interactive Dashboard Map")
st.write("Hover over the red dots to see intersection names")

# Option 1: Just show the intersection points (recommended for your current data)
dots = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position="[lon, lat]",
    get_color=[255, 0, 0, 160],  # Added transparency
    get_radius=50,
    pickable=True,
)

# Option 2: If you want to connect the points with a path, use PathLayer
path_data = [{"path": [[row['lon'], row['lat']] for _, row in df.iterrows()]}]
path_layer = pdk.Layer(
    "PathLayer",
    data=path_data,
    get_path="path",
    get_color=[0, 150, 255, 100],
    get_width=5,
    pickable=False,
)

# Set up the map view
view_state = pdk.ViewState(
    latitude=df['lat'].mean(),
    longitude=df['lon'].mean(),
    zoom=13,
    pitch=0
)

# Create the map (choose one of the options below)

# Option A: Just intersection points
st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=view_state,
        layers=[dots],
        tooltip={"text": "📍 {name}"},
    )
)

# Option B: Points connected by a path (uncomment to use)
# st.pydeck_chart(
#     pdk.Deck(
#         map_style="mapbox://styles/mapbox/light-v9",
#         initial_view_state=view_state,
#         layers=[path_layer, dots],
#         tooltip={"text": "📍 {name}"},
#     )
# )