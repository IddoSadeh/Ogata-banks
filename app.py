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


# The concentration distribution in a column is completely described by the Ogata-Banks equation
# V - velocity
# D - Coefficient of hydrodynamic dispersion
# C - initial concentration
def ogata_banks(V=0.1, D=0.1, C=1, time=300):
    Vt = V * time
    Dt = D * time
    distance = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    denomanator = math.sqrt(Dt) * 2
    first_term = [(math.exp(V * d / D)) * math.erfc((d + Vt) / denomanator) for d in distance]

    second_term = [1 + math.erf(-(d - Vt) / denomanator) if (d - Vt) <= 0 else math.erfc(d - Vt) / denomanator for d in
                   distance]
    concentration = [(C / 2) * (f + s) for f, s in zip(first_term, second_term)]

    return concentration, distance


print(ogata_banks())
df1 = pd.DataFrame(dict(
    distance=ogata_banks()[1],
    concentration=ogata_banks()[0]
))
fig1 = px.line(df1, x="distance", y="concentration")

# for different colors for each surface see https://stackoverflow.com/questions/53992606/plotly-different-color-surfaces
cmap = plt.get_cmap("tab10")

colorscale = [[0, 'rgb' + str(cmap(1)[0:3])],
              [0.5, 'rgb' + str(cmap(2)[0:3])],
              [1, 'rgb' + str(cmap(5)[0:3])], ]

fig2 = go.Figure(data=[
    go.Surface(x=ogata_banks(C=1)[1], y=list(range(1, 11)), z=11 * [ogata_banks(C=1)[0]],
               surfacecolor=np.full(shape=np.array(11 * [ogata_banks(C=1)[0]]).shape,fill_value=1), colorscale=colorscale, cmin=1,
               cmax=3,),
    go.Surface(x=ogata_banks(C=2)[1], y=list(range(1, 11)), z=11 * [ogata_banks(C=2)[0]],
               surfacecolor=np.full(shape=np.array(11 * [ogata_banks(C=2)[0]]).shape,fill_value=2), colorscale=colorscale, cmin=1,
               cmax=3, ),
    go.Surface(x=ogata_banks(C=3)[1], y=list(range(1, 11)), z=11 * [ogata_banks(C=3)[0]],
               surfacecolor=np.full(shape=np.array(11 * [ogata_banks(C=3)[0]]).shape, fill_value=3),
               colorscale=colorscale, cmin=1,
               cmax=3, ),
], )
fig2.update_layout(scene=dict(zaxis_title="concentration"))


app.layout = html.Div(children=[
    html.H1(children='Modeling the Transport of Dissolved contaminants'),

    html.Div(children='''
        the spatial variation in relative concentration along the column at one time. 
    '''),

    dcc.Graph(
        id='1d-graph',
        figure=fig1
    ),
    html.H4(children="Adjust time"),
    dcc.Slider(1, 1000, 40,
               value=300,
               id='time-slider'
               ),
    html.Div(id='time-output-container'),
    html.H4(children="Adjust intial concentration"),
    dcc.Slider(1, 10, 1,
               value=1,
               id='conc-slider'
               ),
    html.Div(id='conc-output-container'),
    html.H2([
        html.Div(children='''
       contour plot of the spatial variation in relative concentration along the column at one time. 
   '''),

        dcc.Graph(
            id='count-graph',
            figure=fig2
        ),
    ])
])


@app.callback(
    Output('1d-graph', 'figure'),
    Input('time-slider', 'value'),
    Input('conc-slider', 'value'))
def update_output(t, c):
    ob = ogata_banks(C=c, time=t)
    df = pd.DataFrame(dict(
        distance=ob[1],
        concentration=ob[0]
    ))
    fig = px.line(df, x="distance", y="concentration")
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
