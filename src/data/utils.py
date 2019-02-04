import pandas as pd
import pickle
from pathlib import Path

DATA_DIR = str(Path(__file__).parent.parent.parent/'data')
MODEL_DIR = str(Path(__file__).parent.parent.parent/'models')

CREDENTIALS = str(Path(__file__).parent/'credentials.yaml')

def load_gbd_location_metadata():
    return pd.read_csv(f'{DATA_DIR}/metadata/gbd_location_metadata.csv')

# LOCATION_METADATA = load_gbd_location_metadata()


def load_indicator_dictionary():
    with open(f'{DATA_DIR}/metadata/indicator_dictionary.pickle', 'rb') as handle:
        d = pickle.load(handle)
    return d

# INDICATOR_DICTIONARY = load_indicator_dictionary()

