import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objs as go
# from src.data.utils import load_indicator_dictionary
from src.visualization.utils import palette
from src.data.process_SDG import load_2017_sdg_data

n_neighbors = 4  # Number of neighbors to plot

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

df = load_2017_sdg_data()
clusters = pd.read_csv('./models/clusters.csv')
location_metadata = pd.read_csv('./data/metadata/gbd_location_metadata.csv')
with open('./data/metadata/indicator_dictionary.pickle', 'rb') as handle:
    indicator_dict = pickle.load(handle)

# Indicator Values in wide format with cluster
df_c = df.pivot(index='location_name', columns='indicator_short', values='scaled_value')
df_c = pd.merge(location_metadata, df_c.reset_index())
df_c = pd.merge(clusters, df_c)
colorscale = [[k/(len(palette)-1), palette[k]] for k in palette.keys()]  #choropleth colorscale seems to need 0-1 range
df_c['color'] = df_c.cluster.map(palette)

# Indicator Value by country in wide format
data = df.pivot(index='location_name', columns='indicator_short', values='scaled_value')


top_markdown_text = '''
### Sustainable Development Goals
#### Zane Rankin, 2/2/2019
In 2015, the United Nations established the Sustainable Development Goals (SDGs). 
The Institute for Health Metrics and Evaluation (IHME) provides estimates for 41 health-related SDG indicators for 195 countries and territories, along with a [data visualization](https://vizhub.healthdata.org/sdg/) the [underlying data](http://ghdx.healthdata.org/record/global-burden-disease-study-2017-gbd-2017-health-related-sustainable-development-goals-sdg).  
In this analysis, rather than grouping countries by geography, I have use a k-means clustering algorithm (details forthcoming) to cluster 
countries based on their SDG indicator values. Indicators are scaled 0-100, with 0 being poor (e.g. high mortality) and 100 being excellent.  
Visualization made using Ploty and Dash - [Github repo](https://github.com/zwrankin/health_indicators)
'''

bottom_markdown_text = '''
  
*Plotly interaction tips*  
*Hover over points to see their values, click on legend items to toggle traces,* 
*click and drag to zoom (double click to unzoom), hold down shift, and click and drag to pan.*
'''

app.layout = html.Div([

    dcc.Markdown(children=top_markdown_text),


html.Div([
    dcc.Markdown('*Indicator abbreviation lookup*'),
    dcc.Dropdown(
                    id='indicator-dropdown',
                    options=[{'label': i, 'value': i} for i in indicator_dict.keys()],
                    value='Under-5 Mort'
                ),
    dcc.Markdown(id='indicator-key')
    ]),

    html.Div([

    dcc.Graph(
			id = 'county-choropleth',
			figure = dict(
				data=[dict(
                    locations = df_c['ihme_loc_id'],
                    z = df_c['cluster'].astype('float'),
					text = df_c['location_name'],
                    colorscale= colorscale,
                    autocolorscale=False,
					type = 'choropleth',
                    # colorbar = {'title': 'Cluster'}
                    showscale=False,
				)],
            layout = dict(
                    title='Hover over map to select scatterplot country',
                    height=400,
                    geo=dict(showframe=False,
                             projection={'type':'Mercator'})) # 'natural earth
			)
		),
        dcc.Dropdown(
                    id='xaxis-column',
                    options=[{'label': i, 'value': i} for i in indicator_dict.keys()],
                    value='SDG Index'
                ),
        dcc.Dropdown(
                    id='yaxis-column',
                    options=[{'label': i, 'value': i} for i in indicator_dict.keys()],
                    value='Under-5 Mort'
                ),
    dcc.Graph(id='scatterplot'),

    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),


    html.Div([

        dcc.RadioItems(
                id='comparison-type',
                options=[{'label': i, 'value': i} for i in ['Value', 'Comparison']],
                value='Value',
                labelStyle={'display': 'inline-block'},
            ),
        dcc.Graph(id='similarity_scatter'),

    ], style={'display': 'inline-block', 'float': 'right', 'width': '49%'}),

    dcc.Markdown(children=bottom_markdown_text),


])


@app.callback(
    dash.dependencies.Output('indicator-key', 'children'),
    [dash.dependencies.Input('indicator-dropdown', 'value')])
def update_graph(i):
    return f'{indicator_dict[i]}'


@app.callback(
    dash.dependencies.Output('similarity_scatter', 'figure'),
    [dash.dependencies.Input('county-choropleth', 'hoverData'),
     dash.dependencies.Input('comparison-type', 'value')])
def update_scatterplot(hoverData, comparison_type):

    if hoverData is None:  # Initialize with country before any hovering
        location_name = 'United States'
    else:
        location_name = hoverData['points'][0]['text']
    l_data = data.loc[location_name]
    similarity = np.abs(data ** 2 - l_data ** 2).sum(axis=1).sort_values()
    idx_similar = similarity[:n_neighbors + 1].index
    df_similar = data.loc[idx_similar]
    if comparison_type == 'Value':
        title=f'Value of {location_name} and similar countries'
    elif comparison_type == 'Comparison':
        df_similar = (df_similar - l_data)
        title = f'Difference between {location_name} and similar countries'
    df_similar = df_similar.reset_index().melt(id_vars='location_name')
    df_similar.sort_values(['location_name', 'indicator_short'], inplace=True, ascending=False)

    # dif_similar = pd.merge(dif_similar, LOCATION_METADATA)
    # loc_id = 2
    return {
        'data': [
            go.Scatter(
                x=df_similar[df_similar['location_name'] == i]['value'],
                y=df_similar[df_similar['location_name'] == i]['indicator_short'],
                text=str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 10,
                    # 'color': df[df['cluster'] == i]['color'],  # palette[i],
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in df_similar.location_name.unique()
        ],
        'layout': go.Layout(
            title=title,
            height=850,
            margin={'l': 120, 'b': 40, 't': 40, 'r': 0},
            hovermode='closest'
        )
    }


@app.callback(
    dash.dependencies.Output('scatterplot', 'figure'),
    [dash.dependencies.Input('xaxis-column', 'value'),
     dash.dependencies.Input('yaxis-column', 'value')])
def update_graph(xaxis_column_name, yaxis_column_name):

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
                    'color': df_c[df_c['cluster'] == i]['color'], # palette[i],
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=f'Cluster {i}'
            ) for i in df_c.cluster.unique()
        ],
        'layout': go.Layout(
            height=400,
            xaxis={'title': xaxis_column_name},
            yaxis={'title': yaxis_column_name},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1},
            hovermode='closest'
        )
    }


if __name__ == '__main__':
    app.run_server(debug=True)