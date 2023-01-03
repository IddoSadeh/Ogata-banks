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
server=app.server

# The concentration distribution in a column is completely described by the Ogata-Banks equation
# V - velocity
# D - Coefficient of hydrodynamic dispersion
# C - initial concentration
def ogata_banks_c_vs_d(V=0.1, D=0.1, C=1, time=300):
    Vt = V * time
    Dt = D * time
    distance = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    denomanator = math.sqrt(Dt) * 2
    first_term = [(math.exp(V * d / D)) * math.erfc((d + Vt) / denomanator) for d in distance]

    second_term = [1 + math.erf(-(d - Vt) / denomanator) if (d - Vt) <= 0 else math.erfc(d - Vt) / denomanator for d in
                   distance]
    concentration = [(C / 2) * (f + s) for f, s in zip(first_term, second_term)]

    return concentration, distance


# The concentration distribution at distance in a column over time is completely described by the Ogata-Banks equation
# V - velocity
# D - Coefficient of hydrodynamic dispersion
# C - initial concentration
# dist - distance
def ogata_banks_c_vs_t(V=0.1, D=10, C=1, dist=10):
    time = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    Vt = [V * t for t in time]
    Dt = [D * t for t in time]

    denominator = [math.sqrt(dt) * 2 for dt in Dt]
    first_term = [(math.exp(V * dist / D)) * math.erfc((dist + vt)) for vt in Vt]
    first_term = [f / den for f, den in zip(first_term, denominator)]

    second_term = [1 + math.erf(-(dist - vt)) if (dist - vt) <= 0 else math.erfc(dist - vt) for vt in
                   Vt]
    second_term = [s / den for s, den in zip(second_term, denominator)]

    concentration = [(C / 2) * (f + s) for f, s in zip(first_term, second_term)]

    return concentration, time


# figure 1 concentration vs distance
df1 = pd.DataFrame(dict(
    distance=ogata_banks_c_vs_d()[1],
    concentration=ogata_banks_c_vs_d()[0]
))
fig1 = px.line(df1, x="distance", y="concentration")

# figure 2, 3d contour plot
# for different colors for each surface see https://stackoverflow.com/questions/53992606/plotly-different-color-surfaces
cmap = plt.get_cmap("tab10")

colorscale = [[0, 'rgb' + str(cmap(1)[0:3])],
              [0.5, 'rgb' + str(cmap(2)[0:3])],
              [1, 'rgb' + str(cmap(5)[0:3])], ]

# 3 contours, can be changeable
fig2 = go.Figure(data=[
    go.Surface(x=ogata_banks_c_vs_d(C=1)[1], y=list(range(1, 11)), z=11 * [ogata_banks_c_vs_d(C=1)[0]],
               surfacecolor=np.full(shape=np.array(11 * [ogata_banks_c_vs_d(C=1)[0]]).shape, fill_value=1),
               colorscale=colorscale, cmin=1,
               cmax=3, ),
    go.Surface(x=ogata_banks_c_vs_d(C=2)[1], y=list(range(1, 11)), z=11 * [ogata_banks_c_vs_d(C=2)[0]],
               surfacecolor=np.full(shape=np.array(11 * [ogata_banks_c_vs_d(C=2)[0]]).shape, fill_value=2),
               colorscale=colorscale, cmin=1,
               cmax=3, ),
    go.Surface(x=ogata_banks_c_vs_d(C=3)[1], y=list(range(1, 11)), z=11 * [ogata_banks_c_vs_d(C=3)[0]],
               surfacecolor=np.full(shape=np.array(11 * [ogata_banks_c_vs_d(C=3)[0]]).shape, fill_value=3),
               colorscale=colorscale, cmin=1,
               cmax=3, ),
], )
fig2.update_layout(scene=dict(zaxis_title="concentration"))

# figure 3 concentration vs time
df3 = pd.DataFrame(dict(
    time=ogata_banks_c_vs_t()[1],
    concentration=ogata_banks_c_vs_t()[0]
))
fig3 = px.line(df3, x="time", y="concentration")

# figure 4 conentration vs distance over time

time = []
distance = []
concentration = []
for t in range(10,400,10):
    func_call = ogata_banks_c_vs_d(time=t)
    distance.extend(func_call[1])
    concentration.extend(func_call[0])
    time.extend(([t]*len(func_call[0])))
df4 = pd.DataFrame(dict(
    time=time,
    distance=distance,
    concentration=concentration
))

fig4 = px.line(df4, x="distance", y="concentration", animation_frame="time",range_y=[0,max(concentration)])
# app layout
app.layout = html.Div(children=[
    html.H1(children='Modeling the Transport of Dissolved contaminants'),

    html.H2(children='''
        The spatial variation in relative concentration along the column at one time:
    '''),

    dcc.Graph(
        id='1d-graph-dist',
        figure=fig1  # concentraiton vs distance
    ),
    html.H4(children="Adjust time1"),
    dcc.Slider(1, 1000, 40,
               value=300,
               id='time-slider1'
               ),
    html.Div(id='time-output-container1'),
    html.H4(children="Adjust intial concentration"),
    dcc.Slider(1, 10, 1,
               value=1,
               id='conc-slider1'
               ),
    html.Div(id='conc-output-container'),
    html.H2([
        html.Div(children='''Contour plot of the spatial variation in relative concentration along the column at one 
        time for varying initial concentraitons: '''),

        dcc.Graph(
            id='count-graph',
            figure=fig2  # 3d contour plot
        ),
    ]),
    html.H2(children='''
       the time variation in relative concentration along the column at one distance:
   '''),

    dcc.Graph(
        id='1d-graph-time',
        figure=fig3  # concentraiton vs time at given distance
    ),
    html.H4(children="Adjust dist2"),
    dcc.Slider(0, 30, 3,
               value=10,
               id='dist-slider2'
               ),
    html.Div(id='time-output-container2'),
    html.H4(children="Adjust intial concentration"),
    dcc.Slider(1, 10, 1,
               value=1,
               id='conc-slider2'
               ),
    html.H2(children='''
       the spatial variation in relative concentration along the column, animated as time changes: 
   '''),

    dcc.Graph(
        id='1d-graph-dist-time',
        figure=fig4  # concentraiton vs distance animated across time
    ),
])


@app.callback(
    Output('1d-graph-dist', 'figure'),
    Input('time-slider1', 'value'),
    Input('conc-slider1', 'value'))
def update_output(t, c):
    ob = ogata_banks_c_vs_d(C=c, time=t)
    df = pd.DataFrame(dict(
        distance=ob[1],
        concentration=ob[0]
    ))
    fig = px.line(df, x="distance", y="concentration")
    return fig


@app.callback(
    Output('1d-graph-time', 'figure'),
    Input('dist-slider2', 'value'),
    Input('conc-slider2', 'value'))
def update_output(d, c):
    ob = ogata_banks_c_vs_t(C=c, dist=d)
    df = pd.DataFrame(dict(
        time=ob[1],
        concentration=ob[0]
    ))
    fig = px.line(df, x="time", y="concentration")
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
