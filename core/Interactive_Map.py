import pydeck as pdk

def create_corridor_map(df, variable="travel_time"):
    # Add a color column based on the selected variable
    def compute_color(row):
        if variable == "travel_time":
            g = max(0, 255 - int(row['travel_time']*40))
            return [255, g, 0]
        else:
            r = max(0, 255 - int(row['speed']*7))
            return [r, 255, 0]
    df = df.copy()
    df['color'] = df.apply(compute_color, axis=1)

    layer = pdk.Layer(
        "LineLayer",
        df,
        get_source_position=["lon1", "lat1"],
        get_target_position=["lon2", "lat2"],
        get_width=7,
        get_color="color",
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(
        latitude=df["lat1"].mean(),
        longitude=df["lon1"].mean(),
        zoom=13,
        pitch=0,
    )

    tooltip = {
        "text": f"From: {{origin}}\nTo: {{destination}}\nTravel Time: {{travel_time}} min\nSpeed: {{speed}} mph"
    }

    return pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)

