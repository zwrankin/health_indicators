import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import plotly.graph_objs as go
from src.visualization.utils import palette

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

################################################################################################################
# LOAD AND PROCESS DATA

# Load
df = pd.read_csv('./data/processed/GBD_child_health_indicators.csv')
location_metadata = pd.read_csv('./data/metadata/gbd_location_metadata.csv')
indicators = list(df.indicator.unique())
n_neighbors = 4
year_ids = list(df.year_id.unique())

# Indicator Value by country in wide format
index_vars = ['location_name', 'year_id']
df_wide = df.pivot_table(index=index_vars, columns='indicator', values='val').reset_index()
df_wide = pd.merge(location_metadata, df_wide)
################################################################################################################


top_markdown_text = '''
### Global Burden of Disease - Child Health Indicators
'''

overview_markdown_text = '''
The Global Burden of Disease produces many indicators relevant to child health. Results from the GBD 2016 study can 
be downloaded [here](http://ghdx.healthdata.org/gbd-2016).  
**Indicators values are scaled 0-100**. For risks and diseases, 0 represents the lowest burden observed and 100 the highest. 
For other quantities (e.g. Socio-Demographic Index), higher values are better.  
In this clustering analysis, I examine how epidemiologic patterns can both follow and defy geographic proximity.  
Clusters are assigned by a k-means clustering algorithm using the user's selected indicators and number of clusters.  
'''

bottom_markdown_text = '''
Estimates by the [Institute for Health Metrics and Evaluation](http://www.healthdata.org/) and available 
[here](http://ghdx.healthdata.org/gbd-2016)  
Visualization by [Zane Rankin](https://github.com/zwrankin/health_indicators)
'''

app.layout = html.Div([

    # LEFT - Global options and map
    html.Div([

        dcc.Markdown(children=top_markdown_text),

        html.P('Number of clusters'),
        dcc.Slider(
            id='n-clusters',
            min=2,
            max=8,
            step=1,
            marks={i: str(i) for i in range(2, 8 + 1)},
            value=7,
        ),
        html.P('_'),
        dcc.RadioItems(
            id='year',
            options=[{'label': i, 'value': i} for i in year_ids],
            value=2016,
            labelStyle={'display': 'inline-block'},
        ),

        html.P('Indicators to include in clustering algorithm'),
        dcc.Dropdown(
            id='indicators',
            options=[{'label': i, 'value': i} for i in indicators],
            multi=True,
            value=[i for i in indicators]
        ),

        dcc.Graph(id='county-choropleth'),
    ], style={'float': 'left', 'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),

    # RIGHT - Tabs
    html.Div([
        dcc.Tabs(id="tabs", style={
            'textAlign': 'left', 'margin': '48px 0', 'fontFamily': 'system-ui'}, children=[

            dcc.Tab(label='Clustering', children=[

                # Hidden div stores the clustering model results to share between callbacks
                html.Div(id='clustered-data', style={'display': 'none'}),

                html.Div([
                    dcc.Markdown(children=overview_markdown_text),

                    # Dropdown options are set dynamically through callback
                    dcc.Dropdown(
                        id='xaxis-column',
                        value='log_LDI'
                    ),
                    dcc.Dropdown(
                        id='yaxis-column',
                        value='U5MR'
                    ),
                ]),

                html.Div([
                    dcc.Graph(id='scatterplot'),
                ]),

            ]),
            dcc.Tab(label='Comparisons', children=[
                # RIGHT SIDE
                # html.Div([
                dcc.RadioItems(
                    id='entity-type',
                    options=[{'label': i, 'value': i} for i in ['Countries', 'Clusters']],
                    value='Countries',
                    labelStyle={'display': 'inline-block'},
                ),
                dcc.RadioItems(
                    id='comparison-type',
                    options=[{'label': i, 'value': i} for i in ['Value', 'Comparison']],
                    value='Value',
                    labelStyle={'display': 'inline-block'},
                ),
                dcc.Graph(id='similarity_scatter'),

            ]),

            dcc.Tab(label='Time Trends', children=[
                dcc.Graph(id='time-series'),
            ]),

        ]),
    ], style={'float': 'right', 'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),

    html.Div([
        dcc.Markdown(children=bottom_markdown_text),
    ], style={'float': 'left'}),

])


@app.callback(dash.dependencies.Output('xaxis-column', 'options'),
              [dash.dependencies.Input('indicators', 'value')])
def set_xaxis_options(indicators):
    return [{'label': i, 'value': i} for i in indicators]


@app.callback(dash.dependencies.Output('yaxis-column', 'options'),
              [dash.dependencies.Input('indicators', 'value')])
def set_yaxis_options(indicators):
    return [{'label': i, 'value': i} for i in indicators]


@app.callback(dash.dependencies.Output('clustered-data', 'children'),
              [dash.dependencies.Input('n-clusters', 'value'),
               dash.dependencies.Input('indicators', 'value'),
               dash.dependencies.Input('year', 'value')])
def cluster_kmeans(n_clusters, indicators, year):
    df_c = df_wide.query(f'year_id == {year}')[['location_name'] + indicators].set_index('location_name')
    kmean = KMeans(n_clusters=n_clusters, random_state=0)
    kmean.fit(df_c)
    df_c['cluster'] = kmean.labels_
    df_c = pd.merge(location_metadata, df_c.reset_index())
    df_c['color'] = df_c.cluster.map(palette)
    return df_c.to_json()


@app.callback(
    dash.dependencies.Output('county-choropleth', 'figure'),
    [dash.dependencies.Input('clustered-data', 'children')])
def update_map(data_json):
    df_c = pd.read_json(data_json)
    n_clusters = len(df_c.cluster.unique())
    colorscale = [[k / (n_clusters - 1), palette[k]] for k in
                  range(0, n_clusters)]  # choropleth colorscale seems to need 0-1 range

    return dict(
        data=[dict(
            locations=df_c['ihme_loc_id'],
            z=df_c['cluster'].astype('float'),
            text=df_c['location_name'],
            colorscale=colorscale,
            autocolorscale=False,
            type='choropleth',
            showscale=False,  # Color key unnecessary since clusters are arbitrary and have key in scatterplot
        )],
        layout=dict(
            title='Hover over map to select scatterplot country or cluster',
            height=500,
            geo=dict(showframe=False,
                     projection={'type': 'Mercator'}))  # 'natural earth
    )


@app.callback(
    dash.dependencies.Output('scatterplot', 'figure'),
    [dash.dependencies.Input('xaxis-column', 'value'),
     dash.dependencies.Input('yaxis-column', 'value'),
     dash.dependencies.Input('clustered-data', 'children')])
def update_graph(xaxis_column_name, yaxis_column_name, data_json):
    df_c = pd.read_json(data_json).sort_values('cluster')
    return {
        'data': [
            go.Scatter(
                x=df_c[df_c['cluster'] == i][xaxis_column_name],
                y=df_c[df_c['cluster'] == i][yaxis_column_name],
                text=df_c[df_c['cluster'] == i]['location_name'],
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 12,
                    'color': df_c[df_c['cluster'] == i]['color'],  # palette[i], #
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=f'Cluster {i}'
            ) for i in df_c.cluster.unique()
        ],
        'layout': go.Layout(
            height=500,
            xaxis={'title': xaxis_column_name},
            yaxis={'title': yaxis_column_name},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1},
            hovermode='closest'
        )
    }


@app.callback(
    dash.dependencies.Output('similarity_scatter', 'figure'),
    [dash.dependencies.Input('county-choropleth', 'hoverData'),
     dash.dependencies.Input('entity-type', 'value'),
     dash.dependencies.Input('comparison-type', 'value'),
     dash.dependencies.Input('indicators', 'value'),
     dash.dependencies.Input('year', 'value'),
     dash.dependencies.Input('clustered-data', 'children')])
def update_scatterplot(hoverData, entity_type, comparison_type, indicators, year, data_json):
    if hoverData is None:  # Initialize before any hovering
        location_name = 'Nigeria'
        cluster = 0
    else:
        location_name = hoverData['points'][0]['text']
        cluster = hoverData['points'][0]['z']

    if entity_type == 'Countries':
        data = df_wide.query(f'year_id == {year}')[['location_name'] + indicators].set_index('location_name')
        l_data = data.loc[location_name]
        similarity = np.abs(data ** 2 - l_data ** 2).sum(axis=1).sort_values()
        idx_similar = similarity[:n_neighbors + 1].index
        df_similar = data.loc[idx_similar]
        if comparison_type == 'Value':
            title = f'Indicators of {location_name} and similar countries'
        elif comparison_type == 'Comparison':
            df_similar = (df_similar - l_data)
            title = f'Indicators of countries relative to {location_name}'
        df_similar = df_similar.reset_index().melt(id_vars='location_name', var_name='indicator')
        df_similar.sort_values(['location_name', 'indicator'], ascending=[True, False], inplace=True)

        plot = [go.Scatter(
            x=df_similar[df_similar['location_name'] == i]['value'],
            y=df_similar[df_similar['location_name'] == i]['indicator'],
            text=str(i),
            mode='markers',
            opacity=0.7,
            marker={
                'size': 10,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name=str(i)
        ) for i in df_similar.location_name.unique()]

    elif entity_type == 'Clusters':
        df_c = pd.read_json(data_json)[['cluster'] + indicators]
        df_cluster = df_c.groupby('cluster').mean()

        if comparison_type == 'Value':
            title = 'Cluster means'
        elif comparison_type == 'Comparison':
            df_cluster = (df_cluster - df_cluster.loc[cluster])
            title = f'Clusters relative to cluster {cluster}'

        df_cluster = df_cluster.reset_index().melt(id_vars='cluster')
        df_cluster['color'] = df_cluster.cluster.map(palette)
        df_cluster.sort_values(['cluster', 'variable'], ascending=[True, False], inplace=True)

        plot = [go.Scatter(
            x=df_cluster[df_cluster['cluster'] == i]['value'],
            y=df_cluster[df_cluster['cluster'] == i]['variable'],
            text=str(i),
            mode='markers',
            opacity=0.7,
            marker={
                'size': 10,
                'color': df_cluster[df_cluster['cluster'] == i]['color'],
                'line': {'width': 0.5, 'color': 'white'}
            },
            name=f'Cluster {i}'
        ) for i in df_cluster.cluster.unique()
        ]
    return {
        'data': plot,
        'layout': go.Layout(
            title=title,
            height=50 + 20 * len(indicators),
            margin={'l': 220, 'b': 30, 't': 30, 'r': 0},
            hovermode='closest'
        )
    }


@app.callback(
    dash.dependencies.Output('time-series', 'figure'),
    [dash.dependencies.Input('county-choropleth', 'hoverData'),
     dash.dependencies.Input('indicators', 'value'), ])
def update_timeseries(hoverData, indicators):
    if hoverData is None:  # Initialize before any hovering
        location_name = 'Nigeria'
    else:
        location_name = hoverData['points'][0]['text']
    df_l = df.query(f'location_name == "{location_name}"')
    df_l = df_l.query(f'indicator in {indicators}')
    return {
        'data': [
            go.Scatter(
                x=df_l[df_l['indicator'] == i]['year_id'],
                y=df_l[df_l['indicator'] == i]['val'],
                mode='lines',
                opacity=0.7,
                marker={
                    'size': 10,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in df.indicator.unique()
        ],
        'layout': go.Layout(
            title=location_name,
            height=550,
            margin={'l': 22, 'b': 30, 't': 30, 'r': 0},
            hovermode='closest',
        )
    }


if __name__ == '__main__':
    app.run_server(debug=True)
