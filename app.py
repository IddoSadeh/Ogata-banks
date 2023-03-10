# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import math
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt

app = Dash(__name__)
server = app.server


# The concentration distribution in a column is completely described by the Ogata-Banks equation
# V - velocity
# D - Coefficient of hydrodynamic dispersion
# C - initial concentration
def ogata_banks_c_vs_d(V=0.1, D=0.1, C=1, time=300):
    Vt = V * time
    Dt = D * time
    distance = [1] + list(range(5, 105, 1))
    denomanator = math.sqrt(Dt) * 2
    first_term = [(math.exp(V * d / D)) *
                  math.erfc((d + Vt) / denomanator) for d in distance]

    second_term = [1 + math.erf(-(d - Vt) / denomanator) if (d - Vt) <= 0
                   else math.erfc((d - Vt) / denomanator) for d in distance]
    first_term = [(C/2)*f for f in first_term]
    second_term = [(C / 2) * s for s in second_term]
    concentration = [(f + s)
                     for f, s in zip(first_term, second_term)]
    return concentration, distance, first_term, second_term


# The concentration distribution at distance in a column over time is completely described by the Ogata-Banks equation
# V - velocity
# D - Coefficient of hydrodynamic dispersion
# C - initial concentration
# dist - distance
def ogata_banks_c_vs_t(V=0.1, D=0.1, C=1, dist=5):
    time = [1] + list(range(5, 105, 1))
    Vt = [V * t for t in time]
    Dt = [D * t for t in time]

    denominator = [math.sqrt(dt) * 2 for dt in Dt]
    first_term = [(math.exp(V * dist / D)) * math.erfc((dist + vt) / den)
                  for vt, den in zip(Vt, denominator)]

    second_term = [1 + math.erf(-(dist - vt) / den) if (dist - vt) <= 0
                   else math.erfc((dist - vt) / den) for vt, den in zip(Vt, denominator)]

    concentration = [(C / 2) * (f + s)
                     for f, s in zip(first_term, second_term)]

    return concentration, time


# figure 1 concentration vs distance
df1 = pd.DataFrame(dict(
    distance=ogata_banks_c_vs_d()[1],
    concentration=ogata_banks_c_vs_d()[0]
))
fig1 = px.line(df1, x="distance", y="concentration")

# figure 3 concentration vs time
df3 = pd.DataFrame(dict(
    time=ogata_banks_c_vs_t()[1],
    concentration=ogata_banks_c_vs_t()[0]
))
fig3 = px.line(df3, x="time", y="concentration")

# figure 4 animated concentration vs distance over time
time = []
distance = []
concentration = []
for t in range(10, 400, 5):
    func_call = ogata_banks_c_vs_d(time=t)
    distance.extend(func_call[1])
    concentration.extend(func_call[0])
    time.extend(([t] * len(func_call[0])))
df4 = pd.DataFrame(dict(
    time=time,
    distance=distance,
    concentration=concentration
))
fig4 = px.line(df4, x="distance", y="concentration", animation_frame="time", range_y=[0, 1])

# ------------------------------------------------------------------------

# app layout
app.layout = html.Div(children=[
    html.H1(children='Modeling the Transport of Dissolved contaminants'),

    # Fig 1, concentration vs depth; adjustable time
    html.H2(children='Concentration vs depth down the column at a selectable time.'),

    html.Div([
        html.H4(children="Set value for D (0.02 to 1.0)"),
        dcc.Input(
            id="D-value",
            type="number",
            value=0.1,
            placeholder="Set value for D",
            min=0.02, max=1.0, step=0.02
        ),
    ], style={"width": "25%", "display": "inline-block", "vertical-align": "middle", },
    ),

    html.Div([
        html.H4(children="Set value for V (0.02 to 1.0)"),
        dcc.Input(
            id="V-value",
            type="number",
            value=0.1,
            placeholder="Set value for V",
            min=0.02, max=.5, step=0.02
        ),
    ], style={"width": "25%", "display": "inline-block", "vertical-align": "middle", },
    ),
    html.Div([
        html.H4(children="Plot components"),
        dcc.Checklist(options=['Term 1', 'Term 2'], id='component_check')
    ], style={"width": "25%", "display": "inline-block", "vertical-align": "middle", },
    ),
    html.H4(children="Adjust the time"),
    dcc.Slider(1, 1000, 40,
               value=300,
               id='time-slider1'
               ),

    dcc.Graph(
        id='1d-graph-dist',
        figure=fig1  # concentraiton vs distance
    ),
    html.Hr(),

    # Fig 2, Concentration vs time at an adjustable distance.
    html.H2(children='Concentration vs time at a selectable distance. V=0.1, D=0.1'),
    html.H4(children="Adjust distance along the column for this time-plot"),
    dcc.Slider(0, 15, 1,
               value=6,
               id='dist-slider2'
               ),

    dcc.Graph(
        id='1d-graph-time',
        figure=fig3  # concentraiton vs time at given distance
    ),

    html.Hr(),

    # Fig 3, Animated concentration vs time at an adjustable distance.
    html.H2(children='Animated concentration vs time at an adjustable distance.'),

    dcc.Graph(
        id='1d-graph-dt-anim',
        figure=fig4  # concentraiton vs distance animated across time
    ),
],
    style={
        "width": "65%",
        "display": "inline-block",
        "padding": "0 20",
        "vertical-align": "middle",
        "margin-bottom": 30,
        "margin-right": 50,
        "margin-left": 20,
    },
)


# ------------------------------------------------------------------------

@app.callback(
    Output('1d-graph-dist', 'figure'),
    Input('V-value', 'value'),
    Input('D-value', 'value'),
    Input('time-slider1', 'value'),
    Input('component_check', 'value'))
def update_output(v, d, t, component):
    ob = ogata_banks_c_vs_d(V=v, D=d, C=1, time=t)
    df = pd.DataFrame(dict(
        distance=ob[1],
        concentration=ob[0]
    ))
    fig = px.line(df, x="distance", y="concentration")
    if component:
        if 'Term 1' in component:
            fig.add_scatter(x=ob[1], y=ob[2], name='Term 1')
        if 'Term 2' in component:
            fig.add_scatter(x=ob[1], y=ob[3], name='Term 2')
    return fig


@app.callback(
    Output('1d-graph-time', 'figure'),
    Input('dist-slider2', 'value'))
def update_output(d):
    ob = ogata_banks_c_vs_t(C=1, dist=d)
    df = pd.DataFrame(dict(
        time=ob[1],
        concentration=ob[0]
    ))
    fig = px.line(df, x="time", y="concentration", range_y=[0, 1])
    return fig


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)
