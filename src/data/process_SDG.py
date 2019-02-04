import pandas as pd
from .utils import DATA_DIR


def save_2017_sdg_data():
    df = pd.read_csv(f'{DATA_DIR}/raw/IHME_GBD_2017_HEALTH_SDG_1990_2030_SCALED_Y2018M11D08.csv')
    df = df.query('year_id == 2017')
    assert df.duplicated(['location_name', 'indicator_id']).sum() == 0
    df.to_csv(f'{DATA_DIR}/processed/sdg_data_2017.csv', index=False)


def load_2017_sdg_data():
    return pd.read_csv(f'{DATA_DIR}/processed/sdg_data_2017.csv')
