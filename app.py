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
    distance = [1] + list(range(5, 105, 5))
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
    time = [1] + list(range(5, 105, 5))
    Vt = [V * t for t in time]
    Dt = [D * t for t in time]

    denominator = [math.sqrt(dt) * 2 for dt in Dt]
    first_term = [(math.exp(V * dist / D)) * math.erfc((dist + vt) / den) for vt, den in zip(Vt, denominator)]

    second_term = [1 + math.erf(-(dist - vt) / den) if (dist - vt) <= 0 else math.erfc((dist - vt) / den) for vt, den in
                   zip(Vt, denominator)]

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
    go.Surface(x=ogata_banks_c_vs_d(C=1)[1], y=list(range(1, 17)), z=17 * [ogata_banks_c_vs_d(C=1)[0]],
               surfacecolor=np.full(shape=np.array(17 * [ogata_banks_c_vs_d(C=1)[0]]).shape, fill_value=1),
               colorscale=colorscale, cmin=1,
               cmax=3, ),
    go.Surface(x=ogata_banks_c_vs_d(C=2)[1], y=list(range(1, 17)), z=17 * [ogata_banks_c_vs_d(C=2)[0]],
               surfacecolor=np.full(shape=np.array(17 * [ogata_banks_c_vs_d(C=2)[0]]).shape, fill_value=2),
               colorscale=colorscale, cmin=1,
               cmax=3, ),
    go.Surface(x=ogata_banks_c_vs_d(C=3)[1], y=list(range(1, 17)), z=17 * [ogata_banks_c_vs_d(C=3)[0]],
               surfacecolor=np.full(shape=np.array(17 * [ogata_banks_c_vs_d(C=3)[0]]).shape, fill_value=3),
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

fig4 = px.line(df4, x="distance", y="concentration", animation_frame="time", range_y=[0, max(concentration)])

# concentration vs depth as shaded column graph

fig5 = go.Figure(data=
go.Contour(
    z=[[i, i] for i in ogata_banks_c_vs_d()[0]],
    y=ogata_banks_c_vs_d()[1],
    x=[1]
    , colorbar=dict(
        title='Concentration',  # title here
        titleside='right',
        titlefont=dict(
            size=14,
            family='Arial, sans-serif')
    )
))
fig5.update_yaxes(autorange='reversed', title="distance from surface")
fig5.update_xaxes(visible=False, showticklabels=False)
fig5.update_traces(hovertemplate='Distance: %{y}' + '<br>Concentration: %{z:.2f}')

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
    html.H2(children='''
    concentraiton vs depth as a shaded column: 
'''),

    dcc.Graph(
        id='1d-contour-dist-conc',
        figure=fig5  # concentraiton vs depth as a shaded column
    ),
    html.H4(children="Adjust time"),
    dcc.Slider(1, 1000, 40,
               value=300,
               id='time-slider5'
               ),
    html.Div(id='time-output-container5'),
    html.H4(children="Adjust intial concentration"),
    dcc.Slider(1, 10, 1,
               value=1,
               id='conc-slider5'
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


@app.callback(
    Output('1d-contour-dist-conc', 'figure'),
    Input('time-slider5', 'value'),
    Input('conc-slider5', 'value'))
def update_output(t, c):
    fig5 = go.Figure(data=
    go.Contour(
        z=[[i, i] for i in ogata_banks_c_vs_d(C=c, time=t)[0]],
        y=ogata_banks_c_vs_d(C=c, time=t)[1],
        x=[1]
        , colorbar=dict(
            title='Concentration',  # title here
            titleside='right',
            titlefont=dict(
                size=14,
                family='Arial, sans-serif')
        ),
        colorscale=[[0, "white"], [0.5 , "red"], [1, "black"]]
    ))
    fig5.update_yaxes(autorange='reversed', title="distance from surface")
    fig5.update_xaxes(visible=False, showticklabels=False)
    fig5.update_traces(hovertemplate='Distance: %{y}' + '<br>Concentration: %{z:.2f}')
    return fig5


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)
