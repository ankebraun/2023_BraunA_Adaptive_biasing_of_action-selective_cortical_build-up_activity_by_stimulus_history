import os
import pandas as pd
import numpy as np
import pdb
import scipy
import regression_funcs as rf
from sklearn import linear_model
usage = "regr_baseline_ampl_db_clean.py"

long_help = """
Computes regression model for baseline beta lateralization dependent on bias 
"""
from os.path import join


meta_path = '/home/abraun/meg_data'
data_path = '/home/abraun/meg_data/sr_labeled/'
meg_data_path = '/home/abraun/meg_data/sr_contrasts'


subjects = [2, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 42]


regr_beta_bias_all_sj_up_resp_rep = []
regr_beta_bias_all_sj_down_resp_rep = []
regr_beta_bias_all_sj_pooled_rep = []

regr_beta_bias_all_sj_up_resp_neutr = []
regr_beta_bias_all_sj_down_resp_neutr = []
regr_beta_bias_all_sj_pooled_neutr = []

regr_beta_bias_all_sj_up_resp_alt = []
regr_beta_bias_all_sj_down_resp_alt = []
regr_beta_bias_all_sj_pooled_alt = []

regr_beta_bias_all_sj_up_resp = []
regr_beta_bias_all_sj_down_resp = []
regr_beta_bias_all_sj_pooled = []

 
regr = linear_model.LinearRegression()
def compute_reg_beta(X_, meg_data_):
    times = meg_data_.columns   
    idx_start =  np.min(np.where(times >= -0.35))
    idx_stop =  np.min(np.where(times >= -0.1))    
    Y_ = pd.DataFrame({'M1_beta_lat': scipy.stats.zscore(np.mean(meg_data_.query('12 <= freq <= 36 & cluster == "JWG_M1" & contrast == "hand" & epoch == "stimulus" & hemi == "lh_is_ipsi"').groupby(['subject', 'trial']).mean().values[:,idx_start:idx_stop], axis = 1))})		
    regr.fit(X_,Y_)
    regr_beta_bias = regr.coef_[0,0]
    return regr_beta_bias        

for subject in subjects:
    meta_data_filenames = []
    meg_data_filenames = []

    if int(subject) == 3:
        sessions = range(1,3)
    elif int(subject) == 10:
        sessions = range(2,4)
    elif int(subject) == 11:
        sessions = range(2,4)    
    elif int(subject) == 12:
        sessions = range(1,2)        
    else:
        sessions = range(1,4)
    
    for session in sessions: #range(1, 4):
        if int(subject) == 5 and session == 1:
            recs = [2]
        else:
            recs = [1]

        for rec in recs:
            if int(subject) < 10:
               meta_data_filenames.append(join(meta_path, 'P0{}/MEG/Locked/P0{}-S{}_rec{}_stim_new.hdf'.format(subject, subject, session, rec)))
            elif int(subject) >= 10:
               meta_data_filenames.append(join(meta_path, 'P{}/MEG/Locked/P{}-S{}_rec{}_stim_new.hdf'.format(subject, subject, session, rec)))
     
        meg_data_filenames.append(join(meg_data_path, 'S{}-lh_is_ipsi_single_trial_session_{}_new_dB.hdf'.format(subject, session)))

    meg_data = pd.concat([pd.read_hdf(meg_data_filename, sep = '\t') for meg_data_filename in meg_data_filenames])
    meta_data = pd.concat(
        [pd.read_hdf(meta_data_filename, 'df')
            for meta_data_filename in meta_data_filenames])
    M1_beta_lat = meg_data.query('cluster == "JWG_M1" & 12 <= freq <= 36 & epoch == "stimulus" & hemi == "lh_is_ipsi"').groupby('trial').mean()
    
    def get_regr(subject, condition, meta_data, regr_beta_bias_all_sj):
        single_trial_bias, meta_data = rf.get_single_trial_bias(subject, condition, meta_data)
        trials = meta_data['idx']
        trials_up = meta_data.query('resptype == 1')['idx']
        trials_down = meta_data.query('resptype == -1')['idx']
        meg_data_up_resp = meg_data[meg_data.index.get_level_values('trial').isin(trials_up)]
        meg_data_down_resp = meg_data[meg_data.index.get_level_values('trial').isin(trials_down)]
        meg_data_pooled = pd.concat([meg_data_up_resp, -meg_data_down_resp]).sort_values(['trial'])
        times = meg_data.columns   
        idx_start =  np.min(np.where(times >= -0.35))
        idx_stop =  np.min(np.where(times >= -0.1))
        freqs = np.unique(meg_data.index.get_level_values('freq'))        
        single_trial_bias_pooled = single_trial_bias[single_trial_bias.idx.isin(trials)].sort_values('idx').single_trial_bias
        X_pooled = pd.DataFrame({'single_trial_bias_pooled': single_trial_bias_pooled})
        X_pooled_zscore = pd.DataFrame({'single_trial_bias_pooled': scipy.stats.zscore(single_trial_bias_pooled)}) 
        X_pooled_zscore = X_pooled_zscore[['single_trial_bias_pooled']]
        regr_beta_bias_pooled = compute_reg_beta(X_pooled_zscore, meg_data_pooled)
        return regr_beta_bias_pooled
    
    regr_beta_bias_pooled = get_regr(subject, "all", meta_data, regr_beta_bias_all_sj_pooled)
    regr_beta_bias_pooled_rep = get_regr(subject, "repetitive", meta_data, regr_beta_bias_all_sj_pooled_rep)
    regr_beta_bias_pooled_alt = get_regr(subject, "alternating", meta_data, regr_beta_bias_all_sj_pooled_alt)
    regr_beta_bias_pooled_neutr = get_regr(subject, "neutral", meta_data, regr_beta_bias_all_sj_pooled_neutr)
    
    regr_beta_bias_all_sj_pooled.append(regr_beta_bias_pooled)    
    regr_beta_bias_all_sj_pooled_rep.append(regr_beta_bias_pooled_rep)    
    regr_beta_bias_all_sj_pooled_alt.append(regr_beta_bias_pooled_alt)    
    regr_beta_bias_all_sj_pooled_neutr.append(regr_beta_bias_pooled_neutr)    
    
df_all_sj_regr_beta_bias_pooled = pd.DataFrame(regr_beta_bias_all_sj_pooled)
df_all_sj_regr_beta_bias_pooled.loc[:, 'subject'] = subjects
df_all_sj_regr_beta_bias_pooled.set_index(['subject'], append=True, inplace=True)

df_all_sj_regr_beta_bias_pooled_rep = pd.DataFrame(regr_beta_bias_all_sj_pooled_rep)
df_all_sj_regr_beta_bias_pooled_rep.loc[:, 'subject'] = subjects
df_all_sj_regr_beta_bias_pooled_rep.set_index(['subject'], append=True, inplace=True)

df_all_sj_regr_beta_bias_pooled_alt = pd.DataFrame(regr_beta_bias_all_sj_pooled_alt)
df_all_sj_regr_beta_bias_pooled_alt.loc[:, 'subject'] = subjects
df_all_sj_regr_beta_bias_pooled_alt.set_index(['subject'], append=True, inplace=True)

df_all_sj_regr_beta_bias_pooled_neutr = pd.DataFrame(regr_beta_bias_all_sj_pooled_neutr)
df_all_sj_regr_beta_bias_pooled_neutr.loc[:, 'subject'] = subjects
df_all_sj_regr_beta_bias_pooled_neutr.set_index(['subject'], append=True, inplace=True)

df_all_sj_regr_beta_bias_pooled.to_csv('regr_baseline_ampl_bias_pooled__dB_zscore.csv', sep = '\t')
df_all_sj_regr_beta_bias_pooled_rep.to_csv('regr_baseline_ampl_bias_pooled_rep_dB_zscore.csv', sep = '\t')
df_all_sj_regr_beta_bias_pooled_neutr.to_csv('regr_baseline_ampl_bias_pooled_neutr_dB_zscore.csv', sep = '\t')
df_all_sj_regr_beta_bias_pooled_alt.to_csv('regr_baseline_ampl_bias_pooled_alt_dB_zscore.csv', sep = '\t')




