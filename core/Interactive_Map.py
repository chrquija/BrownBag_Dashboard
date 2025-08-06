import pydeck as pdk

def create_corridor_map(df, variable="travel_time"):
    # Color: green (good), yellow, red (bad)
    if variable == "travel_time":
        get_color = "[255, 255 - min(travel_time*40, 255), 0]"
    else:  # speed: higher = greener
        get_color = "[255 - min(speed*7, 255), 255, 0]"

    layer = pdk.Layer(
        "LineLayer",
        df,
        get_source_position=["lon1", "lat1"],
        get_target_position=["lon2", "lat2"],
        get_width=7,
        get_color=get_color,
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
