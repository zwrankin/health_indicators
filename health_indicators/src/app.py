import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_hdf('../models/data_clustered.hdf').reset_index()
available_indicators = df.columns.tolist()

graph_text = "Plotly graphs are interactive and responsive. \
Hover over points to see their values, click on legend items to toggle traces, \
click and drag to zoom, hold down shift, and click and drag to pan."

app.layout = html.Div([
    html.H1('SDG Clustering'),

    html.Div([

            html.Div([
                dcc.Dropdown(
                    id='xaxis-column',
                    options=[{'label': i, 'value': i} for i in available_indicators],
                    value='SDG Index'
                ),
            ],
                style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                dcc.Dropdown(
                    id='yaxis-column',
                    options=[{'label': i, 'value': i} for i in available_indicators],
                    value='Under-5 Mort'
                ),
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
        ]),

    dcc.Graph(id='scatterplot'),
    html.Div(graph_text),
])


@app.callback(
    dash.dependencies.Output('scatterplot', 'figure'),
    [dash.dependencies.Input('xaxis-column', 'value'),
     dash.dependencies.Input('yaxis-column', 'value')])
def update_graph(xaxis_column_name, yaxis_column_name):

    return {
        'data': [
            go.Scatter(
                x=df[df['cluster'] == i][xaxis_column_name],
                y=df[df['cluster'] == i][yaxis_column_name],
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
            xaxis={'title': xaxis_column_name},
            yaxis={'title': yaxis_column_name},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1},
            hovermode='closest'
        )
    }


if __name__ == '__main__':
    app.run_server(debug=True)