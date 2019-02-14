import numpy as np
import pandas as pd
import pickle
from pathlib import Path

DATA_DIR = str(Path(__file__).parent.parent.parent / 'data')
MODEL_DIR = str(Path(__file__).parent.parent.parent / 'models')
CREDENTIALS = str(Path(__file__).parent / 'credentials.yaml')


def load_gbd_location_metadata():
    return pd.read_csv(f'{DATA_DIR}/metadata/gbd_location_metadata.csv')


def load_indicator_dictionary():
    with open(f'{DATA_DIR}/metadata/indicator_dictionary.pickle', 'rb') as handle:
        d = pickle.load(handle)
    return d


def min_max_scaler(values, multiplier=100, q=(0.025, 0.975), decimals=2):
    minimum, maximum = values.quantile(q)
    # Bound all values to the quantiles
    values.clip(minimum, maximum, inplace=True)
    return np.round((values - minimum) / (maximum - minimum) * multiplier, decimals=decimals)


def min_max_scaler_df(df, val_col, strat_col, multiplier=100):
    output = pd.DataFrame()
    for i in df[strat_col].unique():
        data = df.loc[df[strat_col] == i]
        data['val'] = min_max_scaler(data[val_col], multiplier)
        output = pd.concat([output, data])
    return output
