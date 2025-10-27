import os
import sys

try:
    from dash import Dash, dcc, html, Input, Output
except ModuleNotFoundError:
    sys.stderr.write("Error: Dash is not installed in your environment.\n")
    sys.stderr.write("To fix: activate your virtualenv and run `pip install -r requirements.txt`.\n")
    sys.stderr.write("Example (PowerShell): `& .\\.venv\\Scripts\\Activate.ps1; pip install -r requirements.txt`\n")
    sys.exit(1)

try:
    import plotly.express as px
except ModuleNotFoundError:
    sys.stderr.write("Error: Plotly is not installed in your environment.\n")
    sys.stderr.write("To fix: activate your virtualenv and run `pip install -r requirements.txt`.\n")
    sys.exit(1)

import pandas as pd

# Sample dataset from Plotly Express
# This keeps the app self-contained with no external files.
df = px.data.gapminder()

app = Dash(__name__)
server = app.server  # Expose for production servers like gunicorn

app.title = "Brown Bag - Dash Plotly Demo"

# Precompute option lists
continents = sorted(df['continent'].unique())
continent_options = [{"label": c, "value": c} for c in continents]

# App layout
app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "margin": "0 auto", "maxWidth": "1100px", "padding": "20px"},
    children=[
        html.H1("Brown Bag Dashboard (Dash + Plotly)"),
        html.P("A quick interactive dashboard using Plotly's built-in Gapminder dataset."),

        html.Div(
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "flex-end"},
            children=[
                html.Div([
                    html.Label("Continent"),
                    dcc.Dropdown(
                        id="continent-dd",
                        options=continent_options,
                        value=continents[0] if continents else None,
                        clearable=False,
                        style={"minWidth": "220px"}
                    ),
                ]),
                html.Div([
                    html.Label("Country"),
                    dcc.Dropdown(
                        id="country-dd",
                        options=[],
                        value=None,
                        clearable=False,
                        style={"minWidth": "260px"}
                    ),
                ]),
                html.Div([
                    html.Label("Year Range"),
                    dcc.RangeSlider(
                        id="year-range",
                        min=int(df['year'].min()),
                        max=int(df['year'].max()),
                        step=5,
                        value=[int(df['year'].min()), int(df['year'].max())],
                        marks={int(y): str(int(y)) for y in sorted(df['year'].unique())},
                        allowCross=False,
                    ),
                ], style={"flex": 1, "minWidth": "320px"}),
            ],
        ),

        html.Hr(),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "16px"},
            children=[
                dcc.Graph(id="lifeexp-line"),
                dcc.Graph(id="gdp-lifeexp-scatter"),
            ],
        ),

        html.Footer(
            style={"marginTop": "24px", "fontSize": "12px", "color": "#666"},
            children="Data: Plotly Gapminder | Built with Dash and Plotly"
        ),
    ],
)


@app.callback(
    Output("country-dd", "options"),
    Output("country-dd", "value"),
    Input("continent-dd", "value"),
)
def update_country_options(selected_continent: str):
    if not selected_continent:
        return [], None
    subset = df[df["continent"] == selected_continent]
    countries = sorted(subset["country"].unique())
    options = [{"label": c, "value": c} for c in countries]
    # Default to the first country for convenience
    return options, (countries[0] if countries else None)


@app.callback(
    Output("lifeexp-line", "figure"),
    Output("gdp-lifeexp-scatter", "figure"),
    Input("continent-dd", "value"),
    Input("country-dd", "value"),
    Input("year-range", "value"),
)
def update_charts(selected_continent: str, selected_country: str, year_range):
    year_min, year_max = year_range if year_range else (int(df['year'].min()), int(df['year'].max()))

    # Filter by year range and continent
    filt = (df["year"] >= year_min) & (df["year"] <= year_max)
    if selected_continent:
        filt &= (df["continent"] == selected_continent)

    dff = df[filt]

    # Line chart: Life expectancy over time by country (within continent and range)
    line_fig = px.line(
        dff,
        x="year",
        y="lifeExp",
        color="country",
        title=f"Life Expectancy over Time — {selected_continent or 'All Continents'}",
        labels={"lifeExp": "Life Expectancy", "year": "Year"},
    )

    # Scatter: GDP per capita vs Life Expectancy for the selected country (or first available)
    if selected_country:
        dff_country = dff[dff["country"] == selected_country]
    else:
        # pick first country in filtered set if none selected
        any_country = dff["country"].iloc[0] if not dff.empty else None
        dff_country = dff[dff["country"] == any_country] if any_country else dff.head(0)

    scatter_fig = px.scatter(
        dff_country,
        x="gdpPercap",
        y="lifeExp",
        size="pop",
        color="year",
        title=f"GDP per Capita vs Life Expectancy — {selected_country or 'Country'}",
        labels={"gdpPercap": "GDP per Capita", "lifeExp": "Life Expectancy"},
        hover_name="country",
    )

    scatter_fig.update_layout(legend_title_text="Year")

    return line_fig, scatter_fig


if __name__ == "__main__":
    # Allow host/port overrides via environment variables for flexible deployment
    host = os.getenv("HOST", "0.0.0.0")
    port_str = os.getenv("PORT", "8050")
    try:
        port = int(port_str)
    except ValueError:
        port = 8050

    app.run_server(host=host, port=port, debug=True)
