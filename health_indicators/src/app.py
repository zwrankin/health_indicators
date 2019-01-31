import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_hdf('../models/data_clustered.hdf').reset_index()

graph_text = "Plotly graphs are interactive and responsive. \
Hover over points to see their values, click on legend items to toggle traces, \
click and drag to zoom, hold down shift, and click and drag to pan."

app.layout = html.Div([
    html.H1('SDG Clustering'),
    dcc.Graph(
        id='life-exp-vs-gdp',
        figure={
            'data': [
                go.Scatter(
                    x=df[df['cluster'] == i]['SDG Index'],
                    y=df[df['cluster'] == i]['Under-5 Mort'],
                    text=df[df['cluster'] == i]['location_name'],
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=f'Cluster {i}'
                ) for i in df.cluster.unique()
            ],
            'layout': go.Layout(
                xaxis={'title': 'SDG Index'},
                yaxis={'title': 'U5MR Index'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    ),
    html.Div(graph_text),
])

if __name__ == '__main__':
    app.run_server(debug=True)