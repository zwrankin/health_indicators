import pandas as pd
import requests
from .utils import DATA_DIR, load_gbd_location_metadata
from .download_DHS import GBD_API_KEY

URL = 'https://api.healthdata.org/healthdata/v1/data/gbd'

cause_metadata = pd.read_csv(f'{DATA_DIR}/metadata/GBD_causes.csv')
cause_dict = cause_metadata.set_index('cause_id')['cause_name'].to_dict()
location_metadata = load_gbd_location_metadata()

risk_metadata = pd.read_csv(f'{DATA_DIR}/metadata/GBD_risks.csv')
risk_dict = risk_metadata.set_index('rei_id')['rei_name'].to_dict()

def get_risk_ids():
    df = pd.read_csv(f'{DATA_DIR}/metadata/GBD_risks.csv')
    return df[df.include == 1].rei_id.unique()


def get_cause_ids():
    df = pd.read_csv(f'{DATA_DIR}/metadata/GBD_causes.csv')
    return df[df.include == 1].cause_id.unique()


def check_status(r):
    status = r.json()['meta']['status']
    if status['code'] != '200':
        raise AssertionError(f'Failed HealthData API query - {status["message"]}')


def send_query(url):
    r = requests.get(url)
    check_status(r)
    cols = r.json()['meta']['fields']
    df =  pd.DataFrame(r.json()['data'], columns=cols)
    return df.apply(pd.to_numeric)

def min_max_scaler(values):
    return (values - values.min())/(values.max() - values.min())

def query_cause(cause_id, age_group_id=1, measure='CSMR'):
    if measure == 'CSMR':
        url = f'{URL}/cause/?cause_id={cause_id}&measure_id=1&metric_id=3&age_group_id={age_group_id}&sex_id=3&year_id=2016&authorization={GBD_API_KEY}'
    else:
        raise NotImplementedError(f'{measure} not implemented')

    df = send_query(url)
    df['indicator'] = cause_dict[cause_id]
    df['val'] = min_max_scaler(df.val)
    return df[['location_id', 'indicator', 'val']]


def query_risk(risk_id, age_group_id=1, measure='SEV'):
    if measure == 'SEV':
        url = f'{URL}/sev/?risk_id={risk_id}&measure_id=29&age_group_id={age_group_id}&sex_id=3&year_id=2016&authorization={GBD_API_KEY}'

    else:
        raise AssertionError(f'{measure} not implemented')

    df = send_query(url)
    df['indicator'] = risk_dict[risk_id]
    df['val'] = min_max_scaler(df.val)

    return df[['location_id', 'indicator', 'val']]


def download_GBD_data(save=True):
    risk_ids = get_risk_ids()
    df_risk = pd.concat([query_risk(i) for i in risk_ids])

    cause_ids = get_cause_ids()
    df_cause = pd.concat([query_cause(i) for i in cause_ids])

    df = pd.concat([df_risk, df_cause])
    df = df.sort_values(['location_id', 'indicator'])
    df = pd.merge(location_metadata, df)

    # HACK to only keep countries (which have true ISO-3 codes)
    df['is_country'] = df.ihme_loc_id.transform(lambda x: len(x) == 3)
    df = df[df.is_country].drop(columns='is_country')

    if save:
        df.to_csv(f'{DATA_DIR}/processed/GBD_child_health_indicators.csv', index=False)

    return df

if __name__ == '__main__':
    download_GBD_data()
