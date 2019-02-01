import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_hdf('../models/data_clustered.hdf') #.reset_index()
df_c = df.drop('location_id', axis=1).groupby('cluster').mean().reset_index().melt(id_vars='cluster')
available_indicators = df.columns.tolist()
# https://www.w3schools.com/colors/colors_picker.asp
palette = {
    0: '#ff0000',
    1: '#ff8000',
    2: '#ffff00',
    3: '#40ff00',
    4: '#00ffff',
    5: '#0000ff',
    6: '#ff00ff',
}
colorscale = [[k/(len(palette)-1), palette[k]] for k in palette.keys()]  #choropleth colorscale seems to need 0-1 range
df['color'] = df.cluster.map(palette)
df_c['color'] = df_c.cluster.map(palette)

top_markdown_text = '''
### Sustainable Development Goals

#### Cluster Analysis 
Rather than using geographic regions, I have use a k-means clustering algorithm (details forthcoming) to cluster 
countries based on their SDG indicator values.  
'''

bottom_markdown_text = '''
Data is downloaded from the [Institute for Health Metrics and Evaluation](http://ghdx.healthdata.org/record/global-burden-disease-study-2017-gbd-2017-health-related-sustainable-development-goals-sdg)  
*The United Nations established, in September 2015, the Sustainable Development Goals (SDGs), 
which specify 17 universal goals, 169 targets, and 232 indicators leading up to 2030. 
Drawing from GBD 2017, this dataset provides estimates on progress for 41 health-related SDG indicators 
for 195 countries and territories from 1990 to 2017, and projections, based on past trends, for 2018 to 2030. 
Estimates are also included for the health-related SDG index, a summary measure of overall performance across the health-related SDGs.*  
See further visualizations [here](https://vizhub.healthdata.org/sdg/)
'''

graph_text = "Plotly graphs are interactive and responsive. \
Hover over points to see their values, click on legend items to toggle traces, \
click and drag to zoom, hold down shift, and click and drag to pan."


app.layout = html.Div([
    # html.H1('SDG Clustering'),
    dcc.Markdown(children=top_markdown_text),

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

    dcc.Graph(id='cluster_scatter',
              figure=
              {
        'data': [
            go.Scatter(
                x=df_c[df_c['cluster'] == i]['value'],
                y=df_c[df_c['cluster'] == i]['variable'],
                text=str(i),  # df[df['cluster'] == i]['location_name'],
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 10,
                    'color': df_c[df_c['cluster'] == i]['color'],  # palette[i],
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=f'Cluster {i}'
            ) for i in df_c.cluster.unique()
        ],
        'layout':
            go.Layout(
                title='SDG Indicator Index by Cluster',
                height=1000,
                xaxis={'title': 'Index Value'},
                margin={'l': 120, 'b': 40, 't': 40, 'r': 0},
                hovermode='closest')
    },
              ),

    dcc.Markdown(children=bottom_markdown_text),




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