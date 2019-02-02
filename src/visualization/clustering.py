import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ..data.download_DHS import LOCATION_METADATA

def plot_cluster_heatmap(df_cluster):
    """
    Heatmap of indicator mean by cluster
    :param df_cluster: pd.DataFrame of indicator data indexed by cluster
    :return: None
    """

    plt.figure(figsize=[8, 10])
    sns.heatmap(df_cluster.T, annot=True, fmt=".0f", cmap="coolwarm_r", center=50)


def plot_cluster_indicators(df_cluster, reference_cluster=None, clusters='all'):
    """
    Categorical scatterplot of indicator mean by cluster, either absolute or differenced with reference_cluster
    :param df_cluster: pd.DataFrame of indicator data indexed by cluster
    :param reference_cluster: None, or integer of cluster id
    :return: None
    """

    fig, ax = plt.subplots(figsize=(8, 10))

    if clusters != 'all':
        df_cluster = df_cluster.query(f'cluster in {clusters}')

    if reference_cluster is None:
        df_cluster_long = df_cluster.reset_index().melt(id_vars='cluster')
    else:
        df_cluster_long = (df_cluster - df_cluster.loc[reference_cluster]).reset_index().melt(id_vars='cluster')

    sns.catplot(x='value', y='indicator_short', hue='cluster', palette='bright', data=df_cluster_long, ax=ax)


def plot_country_vs_cluster(data, df_cluster, loc_name, plot_neighbors=True, n_neighbors=3):
    """
    Barplot of country compared to cluster mean, overlaid with scatterplot of neighbors compared to cluster mean
    TODO - should I just calculate df_cluster rather than passing it as an arg?
    :param data: pd.DataFrame with indicators and cluster, indexed by location_id
    :param df_cluster: pd.DataFrame of indicator data indexed by cluster
    :param loc_name: Name of country for comparisons
    :param plot_neighbors: Whether to include the scatterplot of country's neighbors
    :param n_neighbors: How many countries to compare
    :return: None
    """

    loc_id = int(LOCATION_METADATA.loc[LOCATION_METADATA.location_name == loc_name].location_id)

    # Compare country indicators to cluster mean
    l_data = data.loc[loc_id].drop('cluster')
    l_cluster = int(data.loc[loc_id].cluster)
    c_data = df_cluster.loc[l_cluster]
    # If you wanted cluster centroid
    # centroid_data = kmean.cluster_centers_[int(l_cluster)]

    dif = l_data - c_data
    # dif = l_data - 100  # IF YOU WANT REFERENCE TO BE 100 RATHER THAN CLUSTER
    dif = dif.sort_values(ascending=False)

    # Plot difference
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.barplot(x=dif, y=dif.index, palette='coolwarm', ax=ax)
    ax.set_title(f"Difference between {loc_name} and cluster mean")
    ax.set_xlabel("Difference between country and cluster mean")
    #     plt.title(loc_name)
    #     plt.xlabel("Difference between country and cluster mean")

    # Plot other countries
    if plot_neighbors:
        # Get order of indicators
        i_order = pd.Series(dif.index)
        i_order = i_order.reset_index()
        i_order.columns = ['i_order', 'indicator_short']

        similarity = np.abs(data ** 2 - l_data ** 2).sum(axis=1).sort_values()
        idx_similar = similarity[:n_neighbors + 1].index
        df_similar = data.loc[idx_similar]
        dif_similar = (df_similar - c_data).drop('cluster', axis=1).reset_index().melt(id_vars='location_id')
        # dif_similar = (df_similar - 100).drop('cluster', axis=1).reset_index().melt(id_vars='location_id') # IF YOU WANT REFERENCE TO BE 100 RATHER THAN CLUSTER
        dif_similar = pd.merge(dif_similar, LOCATION_METADATA)
        dif_similar = pd.merge(dif_similar, i_order).sort_values('i_order')
        sns.catplot(x='value', y='indicator_short', hue='location_name', palette='bright', data=dif_similar, ax=ax)



