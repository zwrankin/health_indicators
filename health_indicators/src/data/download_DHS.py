import pandas as pd
import requests
from pathlib import Path
import yaml

DATA_DIR = str(Path(__file__).parent.parent.parent/'data')
CREDENTIALS = str(Path(__file__).parent/'credentials.yaml')

def get_api_key(source):
    with open(CREDENTIALS, 'r') as stream:
        cred = yaml.load(stream)
    return cred[f'{source}_API_KEY']

DHS_API_KEY = get_api_key('DHS')
GBD_API_KEY = get_api_key('GBD')


########################################################
# LIST OF INDICATORS
IND_FERTILITY = ['FE_FRTR_W_GFR', 'FE_AAFB_W_M2A',
                 'FP_KMTA_W_MOD', 'FP_NADU_W_NUM', 'MA_AAFM_W_M2A', 'CM_HRFB_C_A1S', 'MM_MMRO_W_GFR']

IND_ANC_DELIVERY = ['RH_ANCP_W_TOT', 'RH_ANCP_W_SKP', 'RH_ANCN_W_N4P', 'RH_DELP_C_DHF', 'RH_DELA_C_TOT', 'RH_PAHC_W_PR1']

IND_CHILD_RISKS = ['CH_SZWT_C_L25', 'CN_NUTS_C_HA2', 'CN_NUTS_C_HA3', 'CN_NUTS_C_WH3', 'CN_NUTS_C_WH2',
             'CN_NUTS_C_WHP', 'CN_NUTS_C_WA3', 'CN_NUTS_C_WA2',
             'CN_BRFI_C_1HR', 'CN_BFDR_C_MNA', 'CN_BFDR_C_MNE',
             'CN_ANMC_C_ANY', 'CN_ANMC_C_MOD', 'CN_ANMC_C_SEV',
                   'HC_SMKH_H_DLY']

IND_WASH = ['WS_SRCE_H_IMP', 'WS_SRCE_H_PIP', 'WS_WTRT_H_APP', 'WS_TLET_H_NIM', ]

IND_ASSETS = ['HC_ELEC_H_ELC', 'HC_FLRM_H_NAT', 'HC_CKFL_H_SLD',]

IND_WOMEN_RISKS = ['AN_NUTS_W_SHT' ,'AN_NUTS_W_TH2', 'AN_NUTS_W_OWT', 'AN_ANEM_W_ANY', 'AN_ANEM_W_MOD', 'AN_ANEM_W_SEV',
                   'AH_TOBC_W_NON', 'AH_TOBC_M_NON']

IND_DISEASE = ['CH_ARIS_C_ARI', 'CH_FEVR_C_FEV', 'CH_DIAR_C_DIA', 'HA_HIVP_B_HIV']

IND_VACCINATION = ['CH_VACS_C_APP', 'CH_VAC1_C_NON']

IND_OTHER = ['AH_HINS_W_NON', 'WE_DMKH_W_WIF', 'FG_PPCG_C_NUM', 'DV_EXPV_W_12M', 'DV_EXSV_W_EVR',
             'CP_CLAB_C_CHL', 'ED_EDUC_W_MYR', 'ED_EDUC_W_CPR', 'ED_LITR_W_LIT',
             'HC_WIXQ_P_GNI']

INDICATORS = IND_FERTILITY + IND_ANC_DELIVERY + IND_CHILD_RISKS + IND_WASH + IND_ASSETS + \
              IND_WOMEN_RISKS + IND_DISEASE + IND_VACCINATION + IND_OTHER

# TO ADD - high risk birth categories, adult mortality rates, ANC components,
# component vaccinations, diet,
# Separate category for women's empowerment?

IND_MORTALITY = ['CM_ECMT_C_NNR', 'CM_ECMT_C_PNR', 'CM_ECMT_C_IMR', 'CM_ECMT_C_CMR', 'MM_MMRT_W_MRT', 'CM_ECMT_C_U5M', 'CM_PNMR_C_NSB']

########################################################


def load_indicator_codebook():
    """
    Queries DHS indicator metadata
    See also: https://api.dhsprogram.com/rest/dhs/indicators?returnFields=IndicatorId,Label,Definition&f=html
    :return: pd.DataFrame
    """
    r = requests.get('https://api.dhsprogram.com/rest/dhs/indicators?apiKey={API_KEY}&perpage=5000')
    df = pd.DataFrame(r.json()['Data'])
    return df

def load_country_codebook():
    r = requests.get('https://api.dhsprogram.com/rest/dhs/countries')
    return pd.DataFrame(r.json()['Data'])
    # countries = pd.DataFrame(r.json()['Data'])[['DHS_CountryCode', 'CountryName']]
    # country_dict = countries.set_index('CountryName').to_dict()
    # country_dict = country_dict['DHS_CountryCode']

def query_dhs_api(indicator_id:str):
    """
    Queries DHS indicators
    :param indicator_id: (e.g. FE_FRTR_W_TFR)
    :return: pd.DataFrame
    """
    url = f'https://api.dhsprogram.com/rest/dhs/data/{indicator_id}?apiKey={API_KEY}&perpage=5000'
    r = requests.get(url)
    df = pd.DataFrame(r.json()['Data'])

    if len(df) > 0:

        # Check duplicates
        if len(df.ByVariableId.unique()) > 1:
            df = df.query('ByVariableId == 14001')  # Use 5 year recall
        if df.duplicated('SurveyId').sum() > 1:
            import pdb;
            pdb.set_trace()
            raise AssertionError(f'Indicator {i} is not unique by SurveyId')

    return df


def save_DHS_data(indicators=INDICATORS, noisy=True, errors='warn', save=True, return_df=True):
    df = pd.DataFrame()
    # More succinct, but less control over missingess etc
    # df = pd.concat([query_dhs_api(i) for i in indicators])

    assert len(indicators) == len(set(indicators)), 'You have duplicate indicators in your list'

    for i in indicators:
        if noisy: print(f'Querying {i}')
        data = query_dhs_api(i)

        # Check existence of data
        if len(data) == 0:
            if errors == 'warn':
                print(f'WARNING - no data for indicator {i}')
            elif errors == 'ignore':
                continue
            else:
               raise AssertionError(f'No data returned for indicator {i}')

        df = pd.concat([df, data])
    cols = ['CountryName', 'DHS_CountryCode', 'SurveyYear', 'SurveyId', 'Indicator', 'IndicatorId', 'Value']
    if save:
        df[cols].to_hdf(f'{DATA_DIR}/raw/DHS_data.hdf', key='data')
    if return_df:
        return df[cols]


def load_DHS_data():
    return pd.read_hdf(f'{DATA_DIR}/raw/DHS_data.hdf')

def load_SDG_indicators():
    return pd.read_csv(f'{DATA_DIR}/raw/IHME_GBD_2017_HEALTH_SDG_1990_2030_SCALED_Y2018M11D08.csv')

def load_gbd_location_metadata():
    return pd.read_csv(f'{DATA_DIR}/raw/gbd_location_metadata.csv')

if __name__ == '__main__':
    save_DHS_data()
