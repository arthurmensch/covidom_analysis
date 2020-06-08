import pandas as pd
import re
import numpy as np
from functools import partial

def mask_start_reason(dfs):
    print ('Number of people excluded because no start reason :')
    for df in dfs:
        mask = df['start_reason'] == '(Non renseigné)'
        print (df.loc[mask, 'start_reason'].shape[0])
        df.loc[mask, 'start_reason'] = np.nan
        df = df.loc[~df['start_reason'].isna()].copy()
    return dfs

def correct_dump_bug(df):
    if 'pcr_result' in df:
        # Correct a dump bug in 20/05/06 dump
        df['pcr_results'] = df['pcr_result'].replace({-1: np.nan, 1: True, 0:False})
    else:
        df['pcr_results'] = df['pcr_results'].replace({-1: np.nan, 1: True, 0:False})
        
    return df

def enrich_survey(df):
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['symptom_start_time'] = pd.to_datetime(df['symptom_start_time'])
    
    df = correct_dump_bug(df)
        
    df = mask_start_reason([df])[0]

    df = df.query('age >= 18')

    df = df.replace({'Oui': True, 'Non': False})

    df['start_from_hospitalisation'] = df['start_reason'] == "En sortie d'hospitalisation"

    mask = (df['height'] > 210) | (df['height'] < 100) | (df['weight'] > 250)
    df.loc[mask, 'height'] = np.nan
    df.loc[mask, 'weight'] = np.nan

    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    df['overweight'] = (df['bmi']  <= 30) & (df['bmi']  >= 25)
    df['obesity'] = df['bmi']  >= 30
    df['anosmia+ageusia'] = df['anosmia'] | df['ageusia']

    df['male'] = df['gender'] == 'Homme'
    df['female'] = df['gender'] == 'Femme'
    df['undetermined'] = df['gender'] == 'Indéterminé'

    df['pcr_positive'] = df['pcr_results']

    df['symptom_to_covidom_duration'] = df['start_time'] - df['symptom_start_time']
    df['fever_cough_dyspnea'] = df['fever'] & df['cough'] & df['breathlessness']

    headache = re.compile('((mal|maux) ([àa] la|de) t[êéeèë]te)|(migr((ai)|e|é)n)|(c[ée](f|ph)al)', re.IGNORECASE)
    upper_respiratory = re.compile('(angin)|((ph|f)ar[yi]n[gj]it)|(rhini)', re.IGNORECASE)
    abdo_pain = re.compile('douleur(s*) abdo', re.IGNORECASE)

    def check(x, regex):
        if x == np.nan:
            return False
        else:
            return regex.match(str(x)) is not None

    df['headache'] = df['other_sympt_str'].map(partial(check, regex=headache))
    df['upper_respiratory'] = df['other_sympt_str'].map(partial(check, regex=upper_respiratory))
    df['abdo_pain'] = df['other_sympt_str'].map(partial(check, regex=abdo_pain))
    df['any_test_done'] = ~df['pcr_results'].isna() | df['ctscan'] | df['xray']
    
    df['test_done'] = ~df['pcr_results'].isna()

    df['diabetes'] = df['diabetes'].fillna(False)
    df['smoker_current'] = df['smoker_current'].fillna(False)

    df['respiratory'] = df['resp_copd'] | df['resp_asthma']

    df['cardio-vascular'] = df['heart_hypertension'] | df['heart_hf'] | df['heart_hypertension']

    df['any_comorbidity'] = np.any(df[['diabetes', 'heart_hypertension', 'resp_copd', 'cancer',
           'heart_hf', 'heart_coronary', 'resp_asthma', 'kidney_disease',
           'other_chronical_disease', 'obesity']], axis=1).astype(bool)
    df['no_comorbidity'] = ~df['any_comorbidity']
    df['no_smoker'] = ~df['smoker_current'].astype(bool)

    df['binned_age'] = pd.cut(df['age'], [17,25,40,55,65,75,85,105])

    bins = df['binned_age'].cat.categories

    for b in bins:
        df[b] = df['binned_age'] == b
    AGE = bins.tolist()

    df['hospitalized'] = df['hospitalized'] | df['start_from_hospitalisation']

    df['non_hospitalized'] = ~(df['hospitalized'].fillna(True))

    df['start_from_consult'] = ~df['start_from_hospitalisation']
    df['all'] = True

    df['samu'] = df['start_reason'] == 'Régulation médicale SAMU'
    df['urgence'] = df['start_reason'] == "Vu en consultation à l'hôpital"
    
    return df

def had_answered(patient_ids, suivi):
    return patient_ids.isin(suivi.patient_id.values)

def enrich_flowchart(dfs, suivi, age_labels):
    
    return_dfs = []
    dfs = mask_start_reason(dfs)
    
    for df in dfs:
        df = df.reset_index()
        df = df.drop_duplicates('patient_id')
        
        if 'pcr_result' in df or 'pcr_results' in df:
            df = correct_dump_bug(df)
        
        df['one_answer'] = had_answered(df.patient_id, suivi)
        df['age_groups'] = pd.cut(df.age, [18, 30, 40, 50, 60, 70, 80, 150], labels=age_labels, right=False)
        df['over_18'] = pd.cut(df.age, [-10, 18, 150], labels=["under_18", "over_18"], right=False)
        df['start_reason_group'] = np.where(df.start_reason == "En sortie d'hospitalisation", "After hospitalization", "Initial outpatient management")
        return_dfs.append(df)
    
    return return_dfs