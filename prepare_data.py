import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from covid19.sql import read_sql
from covid19.covidom.survey import get_merged_survey
from covid19.covidom.history import get_all_history
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

from joblib import dump
# Functions

def prepare(main, effect_col, cols, query_filter='',
            replace_names={}):
    # Filter
    df = main.copy()
    df = df.dropna(subset=[effect_col])
    if query_filter != "":
        df = df.query(query_filter)
    print(f'dropped {len(main) - len(df)} rows')

    # Replace
    for col, names in replace_names.items():
        df[col] = df[col].replace(names)

    # X, y
    X = df[cols].values
    y = df[effect_col].values

    return X, y

def get_pcr_eds():
    """
    Get EDS pcr results in a dataframe with columns 'person_id' and 'eds_pcr'
    Only 0 (for pcr- and na) or 1
    """
    all_pcr = read_sql(
        "select person_id, result_pcr, num_pcr, num_pcr_pos, num_pcr_neg from person"
    )
    assert len(all_pcr) == all_pcr.person_id.nunique()
    pcr = all_pcr.replace({'Positive': 1, 'Unrealized': -1, 'Negative': 0, 'Inconclusive': -1})
    pcr = pcr.rename(columns={i: f'eds_{i}' for i in ['result_pcr', 'num_pcr', 'num_pcr_pos', 'num_pcr_neg']})
    return pcr.reset_index()


def get_scan_unique_pcr_eds(max_n_pcr=10):
    all_pcr = read_sql(
        'select person_id, measurement_id, measurement_datetime, num_pcr, result_radio from measurement_pcr')
    all_pcr['rank_pcr'] = all_pcr.groupby('person_id')['measurement_datetime'].rank(method='min')
    assert all_pcr.groupby(['person_id', 'rank_pcr'])['measurement_id'].agg('count').unique()[0] == 1
    assert all_pcr.groupby(['person_id', 'result_radio'])['measurement_id'].agg('nunique').unique()[0] == 1

    all_pcr = all_pcr.query('rank_pcr<=@max_n_pcr')
    df = pd.pivot_table(all_pcr, index=['person_id', 'result_radio', 'num_pcr'],
                        columns='rank_pcr',
                        aggfunc='first',
                        fill_value=pd.NaT,
                        values='measurement_datetime')
    df = df.rename(columns={i: f'eds_pcr_{int(i)}' for i in df.columns}).reset_index()
    df = df.rename(columns={'result_radio': 'eds_scan', 'num_pcr': 'eds_num_pcr_from_measurement'})
    return df


# Load questionnaire

rech_df = get_merged_survey()
rech_df = rech_df.reset_index()

# Add pcr
pcr = get_pcr_eds()
print('PCR loaded')
merged = pd.merge(rech_df, pcr, on='person_id', how='left')
merged['eds_result_pcr'] = merged['eds_result_pcr'].fillna(-1)
merged['pcr_survey'] = merged['pcr_results'].replace({'Positif (infection à Covid-19 confirmée)': 1,
                                                      'Négatif': 0, 'Non testé': -1})
merged['pcr_results'] = merged[['pcr_survey', 'eds_result_pcr']].max(axis=1)
# Backward compat
merged['pcr_result'] = merged['pcr_results']

# Add scan and pcr date
scans = get_scan_unique_pcr_eds(max_n_pcr=10)
merged = pd.merge(merged, scans, how='left', on='person_id')
assert (merged['eds_num_pcr'] - merged['eds_num_pcr_from_measurement']).max() == 0
print('Scans an pcr dates loaded')

patient, suivi, _ = get_all_history()

doctors = read_sql('SELECT * FROM doctor', use_covidom=True)
doctors = doctors.drop_duplicates('doctor_id')

#Backward compat
merged.to_csv(f'data/merged_{pd.to_datetime("today"):%y-%m-%d}.csv')
patient.to_csv(f'data/patient_{pd.to_datetime("today"):%y-%m-%d}.csv')
doctors.to_csv(f'data/doctors_{pd.to_datetime("today"):%y-%m-%d}.csv')
suivi.to_csv(f'data/suivi_{pd.to_datetime("today"):%y-%m-%d}.csv')

data = merged, patient, doctors, suivi

dump(data, f'data/data_{pd.to_datetime("today"):%y-%m-%d}.pkl')