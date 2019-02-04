import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from src.visualization.utils import palette
LOCATION_METADATA = pd.read_csv(f'./data/raw/gbd_location_metadata.csv')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

# df = pd.read_hdf('./models/data_clustered.hdf') #.reset_index()
df = pd.read_csv('./models/data_clustered.csv')

df_c = df.drop('location_id', axis=1).groupby('cluster').mean().reset_index().melt(id_vars='cluster')

colorscale = [[k/(len(palette)-1), palette[k]] for k in palette.keys()]  #choropleth colorscale seems to need 0-1 range
df['color'] = df.cluster.map(palette)
df_c['color'] = df_c.cluster.map(palette)


# LOAD FULL DATASET
df_all = pd.read_csv(f'./data/raw/IHME_GBD_2017_HEALTH_SDG_1990_2030_SCALED_Y2018M11D08.csv')
data = df_all.query('year_id == 2017')
assert data.duplicated(['location_name', 'indicator_id']).sum() == 0
data = data.pivot(index='location_name', columns='indicator_short', values='scaled_value')
n_neighbors = 4

indicators = df_all[['indicator_short', 'ihme_indicator_description']].drop_duplicates().sort_values('indicator_short')
indicator_dict = pd.Series(indicators.ihme_indicator_description.values, index=indicators.indicator_short).to_dict()



top_markdown_text = '''
### Sustainable Development Goals
#### Zane Rankin, 2/2/2019
In 2015, the United Nations established the Sustainable Development Goals (SDGs). 
The Institute for Health Metrics and Evaluation (IHME) provides estimates for 41 health-related SDG indicators for 195 countries and territories, along with a [data visualization](https://vizhub.healthdata.org/sdg/) the [underlying data](http://ghdx.healthdata.org/record/global-burden-disease-study-2017-gbd-2017-health-related-sustainable-development-goals-sdg).  
In this analysis, rather than grouping countries by geography, I have use a k-means clustering algorithm (details forthcoming) to cluster 
countries based on their SDG indicator values. Indicators are scaled 0-100, with 0 being poor (e.g. high mortality) and 100 being excellent.  
Visualization made using Ploty and Dash - [Github repo](https://github.com/zwrankin/health_indicators)
'''

app.layout = html.Div([
    # html.H1('SDG Clustering'),
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
                    locations = df['ihme_loc_id'],
                    z = df['cluster'].astype('float'),
					text = df['location_name'],
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

    # dcc.Graph(id='cluster_scatter',
    #           figure=
    #           {
    #     'data': [
    #         go.Scatter(
    #             x=df_c[df_c['cluster'] == i]['value'],
    #             y=df_c[df_c['cluster'] == i]['variable'],
    #             text=str(i),  # df[df['cluster'] == i]['location_name'],
    #             mode='markers',
    #             opacity=0.7,
    #             marker={
    #                 'size': 10,
    #                 'color': df_c[df_c['cluster'] == i]['color'],  # palette[i],
    #                 'line': {'width': 0.5, 'color': 'white'}
    #             },
    #             name=f'Cluster {i}'
    #         ) for i in df_c.cluster.unique()
    #     ],
    #     'layout':
    #         go.Layout(
    #             title='SDG Indicator Index by Cluster',
    #             height=1000,
    #             xaxis={'title': 'Index Value'},
    #             margin={'l': 120, 'b': 40, 't': 40, 'r': 0},
    #             hovermode='closest')
    # },
    #           ),

    html.Div([
    # dcc.Graph(id='similarity_scatter',
    #           figure=
    #           {
    #               'data': [
    #                 go.Scatter(
    #                     x=dif_similar[dif_similar['location_id'] == i]['value'],
    #                     y=dif_similar[dif_similar['location_id'] == i]['indicator_short'],
    #                     text=dif_similar[dif_similar['location_id'] == i]['location_name'],
    #                     mode='markers',
    #                     opacity=0.7,
    #                     marker={
    #                         'size': 10,
    #                         # 'color': dif_similar[dif_similar['cluster'] == i]['color'],  # palette[i],
    #                         'line': {'width': 0.5, 'color': 'white'}
    #                     },
    #                     name=f'Location {i}'
    #                 ) for i in dif_similar.location_id.unique()
    #             ],
    #               'layout':
    #                   go.Layout(
    #                       title=f'Difference between {loc_id} and similar countries',
    #                       height=1000,
    #                       xaxis={'title': 'Index Value'},
    #                       margin={'l': 120, 'b': 40, 't': 40, 'r': 0},
    #                       hovermode='closest')
    #           },
    #           ),

        dcc.RadioItems(
                id='comparison-type',
                options=[{'label': i, 'value': i} for i in ['Value', 'Comparison']],
                value='Value',
                labelStyle={'display': 'inline-block'},
            ),
        dcc.Graph(id='similarity_scatter',
                #figure=make_similarity_scatterplot('Somalia', 'Value')
                  ),

    ], style={'display': 'inline-block', 'float': 'right', 'width': '49%'}),


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

    # dif_similar = pd.merge(dif_similar, LOCATION_METADATA)
    # loc_id = 2
    return {
        'data': [
            go.Scatter(
                x=df_similar[df_similar['location_name'] == i]['value'],
                y=df_similar[df_similar['location_name'] == i]['indicator_short'],
                text=df[df['cluster'] == i]['location_name'],
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
            height=800,
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
                x=df[df['cluster'] == i][xaxis_column_name],
                y=df[df['cluster'] == i][yaxis_column_name],
                text=df[df['cluster'] == i]['location_name'],
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 12,
                    'color': df[df['cluster'] == i]['color'], # palette[i],
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=f'Cluster {i}'
            ) for i in df.cluster.unique()
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