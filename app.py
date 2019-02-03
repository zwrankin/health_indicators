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
available_indicators = df.columns.tolist()

colorscale = [[k/(len(palette)-1), palette[k]] for k in palette.keys()]  #choropleth colorscale seems to need 0-1 range
df['color'] = df.cluster.map(palette)
df_c['color'] = df_c.cluster.map(palette)


# LOAD FULL DATASET
df_all = pd.read_csv(f'./data/raw/IHME_GBD_2017_HEALTH_SDG_1990_2030_SCALED_Y2018M11D08.csv')
data = df_all.query('year_id == 2017')
assert data.duplicated(['location_name', 'indicator_id']).sum() == 0
data = data.pivot(index='location_name', columns='indicator_short', values='scaled_value')
n_neighbors = 4


def make_similarity_scatterplot(location_name):
    l_data = data.loc[location_name]
    similarity = np.abs(data ** 2 - l_data ** 2).sum(axis=1).sort_values()
    idx_similar = similarity[:n_neighbors + 1].index
    df_similar = data.loc[idx_similar]
    df_similar = (df_similar - l_data).reset_index().melt(id_vars='location_name')
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
                    'size': 15,
                    # 'color': df[df['cluster'] == i]['color'],  # palette[i],
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in df_similar.location_name.unique()
        ],
        'layout': go.Layout(
            title=f'Difference between {location_name} and similar countries',
            height=1000,
            margin={'l': 120, 'b': 40, 't': 40, 'r': 0},
            hovermode='closest'
        )
    }


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
'''

graph_text = "Plotly graphs are interactive and responsive. \
Hover over points to see their values, click on legend items to toggle traces, \
click and drag to zoom, hold down shift, and click and drag to pan."


app.layout = html.Div([
    # html.H1('SDG Clustering'),
    dcc.Markdown(children=top_markdown_text),

    # html.Div([

            # html.Div([
            #     dcc.Dropdown(
            #         id='xaxis-column',
            #         options=[{'label': i, 'value': i} for i in available_indicators],
            #         value='SDG Index'
            #     ),
            # ],
            #     style={'width': '48%', 'display': 'inline-block'}),

        #     html.Div([
        #         dcc.Dropdown(
        #             id='yaxis-column',
        #             options=[{'label': i, 'value': i} for i in available_indicators],
        #             value='Under-5 Mort'
        #         ),
        #     ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
        # ]),



    html.Div([
dcc.Dropdown(
                    id='xaxis-column',
                    options=[{'label': i, 'value': i} for i in available_indicators],
                    value='SDG Index'
                ),
dcc.Dropdown(
                    id='yaxis-column',
                    options=[{'label': i, 'value': i} for i in available_indicators],
                    value='Under-5 Mort'
                ),
dcc.Dropdown(
                    id='testing',
                    options=[{'label': i, 'value': i} for i in [1,2]],
                    value=1
                ),
    dcc.Graph(id='scatterplot'),
    # html.Div(graph_text),
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
                    # title='Clustering countries based on SDG Indicators',
                    height=600,
                    geo=dict(showframe=False,
                             projection={'type':'Mercator'})) # 'natural earth
			)
		),


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

        dcc.Graph(id='similarity_scatter',
                figure=make_similarity_scatterplot('Somalia')
                  ),

    ], style={'display': 'inline-block', 'width': '49%'}),

    dcc.Markdown(children=bottom_markdown_text),




])




@app.callback(
    dash.dependencies.Output('similarity_scatter', 'figure'),
    [dash.dependencies.Input('county-choropleth', 'hoverData'), ])
def update_scatterplot(hoverData):
    # print(len(hoverData['points']))
    print(hoverData['points'][0]['text'])
    location_name = hoverData['points'][0]['text']
    return make_similarity_scatterplot(location_name)


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
                    'color': df[df['cluster'] == i]['color'], # palette[i],
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