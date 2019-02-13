import numpy as np
import pandas as pd
from db_queries import get_location_metadata
from db_queries import get_cause_metadata, get_rei_metadata, get_covariate_estimates
from db_queries import get_outputs as go
from .utils import DATA_DIR, min_max_scaler_df

URL = 'https://api.healthdata.org/healthdata/v1/data/gbd'

location_metadata = get_location_metadata(location_set_id=35)
YEAR_IDS = [1990, 1995, 2000, 2005, 2010, 2017]
GBD_ROUND_ID = 5


def get_top_cause_ids(level=3, n=15, gbd_round_id=GBD_ROUND_ID):
    """
    Gets the cause_ids of the causes with the highest mortality burden in 2017
    :param level: level of the GBD cause hieracy
    :param n: number of causes to return
    :param gbd_round_id: GBD round ID
    :return: list of cause_ids
    """
    # Get cause ids of chosen level of the hierarchy
    cause_meta = get_cause_metadata(cause_set_id=3, gbd_round_id=gbd_round_id)
    level_3_cause_ids = cause_meta.query(f'level == {level}').cause_id.unique().tolist()

    # cause specific deaths
    df = go("cause", cause_id=level_3_cause_ids, location_id=1, age_group_id=1, sex_id=3, metric_id=[1], measure_id=[1],
            year_id=2017, gbd_round_id=gbd_round_id)

    # Assing rank and return top cause ids
    df.dropna(inplace=True)
    df['rank'] = df['val'].rank(ascending=False).astype('int')
    df.sort_values('rank', inplace=True)

    return df.query(f'rank <= {n}').cause_id.unique().tolist()


def get_top_risk_ids(n=14, gbd_round_id=GBD_ROUND_ID):
    """
    Gets the risk_ids of the risks with the highest mortality burden in 2017
    Note that only most-detailed risks have SEVs
    :param n: number of risks to return
    :param gbd_round_id:
    :return: list of risk_ids
    """
    # Get risk ids of chosen level of the hierarchy
    risk_meta = get_rei_metadata(rei_set_id=1, gbd_round_id=gbd_round_id)
    risk_ids = risk_meta.query('most_detailed == 1').rei_id.unique().tolist()

    # Attributable burden for all causes and risks (as counts)
    df = go("rei", cause_id=294, rei_id=risk_ids, location_id=1, age_group_id=1, sex_id=3, metric_id=[1],
            measure_id=[1], gbd_round_id=gbd_round_id)

    df.dropna(inplace=True)
    df['rank'] = df['val'].rank(ascending=False).astype('int')
    df.sort_values('rank', inplace=True)

    return df.query(f'rank <= {n}').rei_id.unique().tolist()


def download_cause_data(include_all_cause=True):
    """
    Downloads mortality rate of top causes
    :param include_all_cause: whether to get all-cause mortality
    :return: None
    """
    cause_ids = get_top_cause_ids()
    if include_all_cause:
        cause_ids.append(294)
    df = go("cause", cause_id=cause_ids, location_id='all', age_group_id=1, sex_id=3, metric_id=[3], measure_id=[1],
            year_id=YEAR_IDS, gbd_round_id=GBD_ROUND_ID)
    df.to_csv(f'{DATA_DIR}/raw/gbd_cause_data.csv', index=False)


def download_risk_data():
    """
    Downloads summary exposure variable (SEV) of top risks
    :return: None
    """
    risk_ids = get_top_risk_ids()
    df = go("rei", rei_id=risk_ids, location_id='all', age_group_id=1, sex_id=3, metric_id=[3], measure_id=[29],
            year_id=YEAR_IDS, gbd_round_id=GBD_ROUND_ID)
    df.to_csv(f'{DATA_DIR}/raw/gbd_risk_data.csv', index=False)


def download_covariate_data():
    """Downloads custom list of covariates"""
    cov_key = {7: 'Low ANC coverage',
               # 881: 'Socio-Demographic Index',
               57: 'Low GDP per capita',  # Technically LDI
               32: 'Low DTP3 coverage'}
    cov_ids = list(cov_key.keys())
    df = pd.concat(
        [get_covariate_estimates(covariate_id=i, year_id=YEAR_IDS, gbd_round_id=GBD_ROUND_ID) for i in cov_ids])
    df['indicator'] = df.covariate_id.map(cov_key)
    df.to_csv(f'{DATA_DIR}/raw/gbd_covariate_data.csv', index=False)


def load_cause_data():
    df = pd.read_csv(f'{DATA_DIR}/raw/gbd_cause_data.csv')
    df['indicator'] = df.cause_name
    df.loc[df.cause_name == "All causes", 'indicator'] = 'Under-5 Mortality Rate'
    return df[['location_id', 'year_id', 'indicator', 'val']]


def load_risk_data():
    df = pd.read_csv(f'{DATA_DIR}/raw/gbd_risk_data.csv')
    df['indicator'] = df.rei_name
    return df[['location_id', 'year_id', 'indicator', 'val']]


def load_covariate_data():
    df = pd.read_csv(f'{DATA_DIR}/raw/gbd_covariate_data.csv')

    # Log transform LDI
    df.loc[df.indicator == 'Low GDP per capita', 'mean_value'] = np.log(
        df.loc[df.indicator == 'Low GDP per capita', 'mean_value'])

    # Ensure same directionality as risks & causes (higher is higher risk)
    output = pd.DataFrame()
    for i in df.indicator.unique():
        data = df.loc[df.indicator == i]
        data['val'] = data.mean_value.max() - data.mean_value
        output = pd.concat([output, data])
    return output[['location_id', 'year_id', 'indicator', 'val']]


def download_GBD_data():
    download_cause_data()
    download_risk_data()
    download_covariate_data()


def process_GBD_data(save=True):
    """"Custom processing of GBD results, including subsetting, scaling, and renaming"""
    df_risk = load_risk_data()
    df_cause = load_cause_data()
    df_covariate = load_covariate_data()

    df = pd.concat([df_risk, df_cause, df_covariate])

    # Keep countries
    df = pd.merge(location_metadata, df)
    df = df.query('level == 3').drop(columns='level')

    # Scale
    df = min_max_scaler_df(df, 'val', 'indicator')

    # Custom Renaming of some indicators
    rename_map = {'Household air pollution from solid fuels': 'Household air pollution',
                  'Sexually transmitted infections excluding HIV': 'STDs excluding HIV',
                  'Ambient particulate matter pollution': 'Ambient air pollution'}
    df.indicator.replace(rename_map, inplace=True)

    df = df[['location_name', 'year_id', 'indicator', 'val']]
    df.sort_values(['location_name', 'year_id', 'indicator'], inplace=True)

    if save:
        df.to_csv(f'{DATA_DIR}/processed/GBD_child_health_indicators.csv', index=False)

    return df


if __name__ == '__main__':
    download_GBD_data()
    process_GBD_data()
