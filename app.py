import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import plotly.graph_objs as go
from src.visualization.utils import get_palette

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
The Global Burden of Disease estimates many child health indicators. Understanding temporal and geographic 
patterns can provide insights to attaining Sustainable Development Goal 3.2 (reduce child mortality to <25 per 100K births).  
This clustering analysis examines how epidemiologic patterns can both follow and defy traditional geographic categories. 
Clusters are assigned by a k-means clustering algorithm using selected indicators and number of clusters.  
**Indicator values are scaled 0-100 with 100 representing highest burden**: 
'''
# 0 represents the 2.5th percentile of globally observed values and 100 the 97.5th percentile.
# Available indicators include the top global risks and causes from 2017.

bottom_markdown_text = '''
Estimates by the [Institute for Health Metrics and Evaluation](http://www.healthdata.org/) and available 
[here](http://ghdx.healthdata.org/gbd-2017)  
Visualization by [Zane Rankin](https://github.com/zwrankin/health_indicators)  
'''
# Visit [GBD Compare](https://vizhub.healthdata.org/gbd-compare/#) for complete GBD results visualization.


def make_colorscale(n):
    """Maps [0,n] palette to [0,1] scale to fit Plotly colorscale"""
    return [[k / (n - 1), get_palette(n)[k]] for k in range(0, n)]


app.layout = html.Div([

    # LEFT - Global options and map
    html.Div([

        dcc.Markdown(children=top_markdown_text),

        html.P('Number of clusters'),
        dcc.Slider(
            id='n-clusters',
            min=2,
            max=7,
            step=1,
            marks={i: str(i) for i in range(2, 7 + 1)},
            value=6,
        ),
        html.P('_'),
        dcc.RadioItems(
            id='year',
            options=[{'label': i, 'value': i} for i in year_ids],
            value=2017,
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
        dcc.Markdown(children=bottom_markdown_text)
    ], style={'float': 'left', 'width': '39%', 'display': 'inline-block', 'padding': '0 20'}),

    # RIGHT - Tabs
    html.Div([
        dcc.Tabs(id="tabs", style={
            'textAlign': 'left', 'margin': '48px 0', 'fontSize': 18, 'color': 'blue'}, children=[

            dcc.Tab(label='Clustering', children=[

                # Hidden div stores the clustering model results to share between callbacks
                html.Div(id='clustered-data', style={'display': 'none'}),

                html.Div([
                    dcc.Markdown(children=overview_markdown_text),

                    # Dropdown options are set dynamically through callback
                    dcc.Dropdown(
                        id='xaxis-column',
                        value='Low GDP per capita'
                    ),
                    dcc.Dropdown(
                        id='yaxis-column',
                        value='Under-5 Mortality Rate'
                    ),
                ]),

                html.Div([
                    dcc.Graph(id='scatterplot'),
                ]),

            ]),

            dcc.Tab(label='Comparisons', children=[
                dcc.Markdown('*Locations to compare*'),
                dcc.RadioItems(
                    id='entity-type',
                    options=[{'label': i, 'value': i} for i in ['Countries', 'Clusters']],
                    value='Countries',
                    labelStyle={'display': 'inline-block'},
                ),
                dcc.Markdown('*Whether to plot value or comparison to selected location*'),
                dcc.RadioItems(
                    id='comparison-type',
                    options=[{'label': i, 'value': i} for i in ['Value', 'Comparison']],
                    value='Value',
                    labelStyle={'display': 'inline-block'},
                ),
                daq.BooleanSwitch(
                    id='connect-dots',
                    label="Connect Dots",
                    on=True,
                ),
                dcc.Markdown('*Additional countries to plot*'),
                dcc.Dropdown(
                    id='countries',
                    options=[{'label': i, 'value': i} for i in df_wide.location_name.unique()],
                    multi=True,
                ),
                dcc.Graph(id='similarity_scatter'),

            ]),

            dcc.Tab(label='Time Trends', children=[
                dcc.Markdown(
                    '*For Under-5 Mortality forecasted until 2030, see [SDG Visualization](http://ihmeuw.org/4prj)*'),
                dcc.Graph(id='time-series'),
            ]),

            dcc.Tab(label='Parallel Coordinates', children=[
                dcc.Graph(id='parallel-coords'),
                dcc.Markdown('*Tips: drag along y axis to subset lines, and drag indicator names to reorder columns*')
            ]),

        ]),
    ], style={'float': 'right', 'width': '59%', 'display': 'inline-block', 'padding': '0 20'}),

])


@app.callback(Output('xaxis-column', 'options'),
              [Input('indicators', 'value')])
def set_xaxis_options(indicators):
    return [{'label': i, 'value': i} for i in indicators]


@app.callback(Output('yaxis-column', 'options'),
              [Input('indicators', 'value')])
def set_yaxis_options(indicators):
    return [{'label': i, 'value': i} for i in indicators]


@app.callback(Output('clustered-data', 'children'),
              [Input('n-clusters', 'value'),
               Input('indicators', 'value'),
               Input('year', 'value')])
def cluster_kmeans(n_clusters, indicators, year):
    df_c = df_wide.query(f'year_id == {year}')[['location_name'] + indicators].set_index('location_name')
    kmean = KMeans(n_clusters=n_clusters, random_state=0)
    kmean.fit(df_c)

    # Rank cluster ids by mean U5MR within cluster
    df_ordered = df_wide.query(f'year_id == {year}')
    df_ordered['cluster'] = kmean.labels_
    df_ordered = df_ordered.groupby('cluster')['Under-5 Mortality Rate'].mean().reset_index()
    df_ordered['U5MR_rank'] = df_ordered['Under-5 Mortality Rate'].rank().astype(
        'int') - 1  # rank starts at 1, we want 0-indexed
    cluster_map = df_ordered.set_index('cluster')['U5MR_rank'].to_dict()

    # Set cluster equal to U5MR rank
    df_c.reset_index(inplace=True)
    df_c['cluster'] = pd.Series(kmean.labels_).map(cluster_map)
    df_c = pd.merge(location_metadata, df_c)
    df_c['color'] = df_c.cluster.map(get_palette(n_clusters))
    return df_c.to_json()


@app.callback(
    Output('county-choropleth', 'figure'),
    [Input('clustered-data', 'children')])
def update_map(data_json):
    df_c = pd.read_json(data_json)
    n_clusters = len(df_c.cluster.unique())
    colorscale = make_colorscale(n_clusters)

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
            title='Hover over map to select country to plot',
            height=600,
            geo=dict(showframe=False,
                     projection={'type': 'Mercator'}))
    )


@app.callback(
    Output('scatterplot', 'figure'),
    [Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value'),
     Input('county-choropleth', 'hoverData'),
     Input('clustered-data', 'children')])
def update_graph(xaxis_column_name, yaxis_column_name, hoverData, data_json):
    if hoverData is None:  # Initialize before any hovering
        location_name = 'Nigeria'
    else:
        location_name = hoverData['points'][0]['text']
    df_c = pd.read_json(data_json).sort_values('cluster')
    # Make size of marker respond to map hover
    df_c['size'] = 12
    df_c.loc[df_c.location_name == location_name, 'size'] = 30
    # Make selected country last (so it plots on top)
    df_c = pd.concat([df_c[df_c.location_name != location_name], df_c[df_c.location_name == location_name]])
    return {
        'data': [
            go.Scatter(
                x=df_c[df_c['cluster'] == i][xaxis_column_name],
                y=df_c[df_c['cluster'] == i][yaxis_column_name],
                text=df_c[df_c['cluster'] == i]['location_name'],
                mode='markers',
                opacity=0.7,
                marker={
                    'size': df_c[df_c['cluster'] == i]['size'],  # 12,
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
    Output('similarity_scatter', 'figure'),
    [Input('county-choropleth', 'hoverData'),
     Input('entity-type', 'value'),
     Input('comparison-type', 'value'),
     Input('indicators', 'value'),
     Input('year', 'value'),
     Input('countries', 'value'),
     Input('connect-dots', 'on'),
     Input('clustered-data', 'children')])
def update_scatterplot(hoverData, entity_type, comparison_type, indicators, year, countries, connect_dots, data_json):
    if hoverData is None:  # Initialize before any hovering
        location_name = 'Nigeria'
        cluster = 0
    else:
        location_name = hoverData['points'][0]['text']
        cluster = hoverData['points'][0]['z']

    if connect_dots:
        mode = 'lines+markers'
    else:
        mode = 'markers'

    if entity_type == 'Countries':
        data = df_wide.query(f'year_id == {year}')[['location_name'] + indicators].set_index('location_name')
        l_data = data.loc[location_name]
        similarity = np.abs(data ** 2 - l_data ** 2).sum(axis=1).sort_values()
        locs = similarity[:n_neighbors + 1].index.tolist()
        if countries is not None:
            locs += countries
        df_similar = data.loc[locs]
        if comparison_type == 'Value':
            title = f'Indicators of {location_name} and similar countries'
        elif comparison_type == 'Comparison':
            df_similar = (df_similar - l_data)
            title = f'Indicators of countries relative to {location_name}'
        df_similar = df_similar.reset_index().melt(id_vars='location_name', var_name='indicator')
        df_similar.sort_values(['location_name', 'indicator'], ascending=[True, False], inplace=True)
        df_similar['size'] = 10
        df_similar.loc[df_similar.location_name == location_name, 'size'] = 14
        plot = [go.Scatter(
            x=df_similar[df_similar['location_name'] == i]['value'],
            y=df_similar[df_similar['location_name'] == i]['indicator'],
            text=str(i),
            mode=mode,
            opacity=0.7,
            marker={
                'size': df_similar[df_similar['location_name'] == i]['size'],
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
        n_clusters = len(df_c.cluster.unique())
        df_cluster['color'] = df_cluster.cluster.map(get_palette(n_clusters))
        df_cluster.sort_values(['cluster', 'variable'], ascending=[True, False], inplace=True)

        plot = [go.Scatter(
            x=df_cluster[df_cluster['cluster'] == i]['value'],
            y=df_cluster[df_cluster['cluster'] == i]['variable'],
            text=str(i),
            mode=mode,
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
            height=150 + 20 * len(indicators),
            margin={'l': 200, 'b': 30, 't': 30, 'r': 0},
            hovermode='closest'
        )
    }


@app.callback(
    Output('time-series', 'figure'),
    [Input('county-choropleth', 'hoverData'),
     Input('indicators', 'value'), ])
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
            height=650,
            margin={'l': 22, 'b': 30, 't': 30, 'r': 0},
            hovermode='closest',
        )
    }


@app.callback(
    Output('parallel-coords', 'figure'),
    [Input('indicators', 'value'),
     Input('clustered-data', 'children')])
def update_parallel_coords(indicators, data_json):
    df_c = pd.read_json(data_json).sort_values('cluster')
    n_clusters = len(df_c.cluster.unique())
    colorscale = make_colorscale(n_clusters)

    # Since I can't seem to get vertical axis labels, only plot a subset of indicators for clarity
    if 'Under-5 Mortality Rate' in indicators:
        indicators = ['Under-5 Mortality Rate'] + indicators[:9]
    else:
        indicators = indicators[:10]

    # Want U5MR to be the first column with special constraint range, if in indicator list
    dims = list([dict(
        range=[0, 100],
        tickvals=[100],
        name=i,
        label=i, values=df_c[i], )
        for i in indicators if i != 'Under-5 Mortality Rate'])
    if 'Under-5 Mortality Rate' in indicators:
        dims = [dict(
            range=[0, 100], constraintrange=[0, 100],
            name='U5MR',
            label='U5MR', values=df_c['Under-5 Mortality Rate'])] + dims

    return {
        'data': [
            go.Parcoords(
                line=dict(color=df_c['cluster'], colorscale=colorscale),
                dimensions=dims,
                # hoverinfo='name', hovering doesn't seem to work for paarcords
            )
        ],
        'layout': go.Layout(
            title='Beta version - Only first 10 indicators are plotted ',
            xaxis=dict(visible=False, tickangle=-90),  # Can't seem to rotate axis tick labels
            height=650,
        )
    }


if __name__ == '__main__':
    app.run_server(debug=True)
