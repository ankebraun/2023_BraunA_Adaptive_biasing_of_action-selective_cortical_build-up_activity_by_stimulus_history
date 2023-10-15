
import os
import pandas as pd
import numpy as np
import pdb
from sklearn import linear_model
import scipy
import regression_funcs as rf
usage = "regression_model_sliding_window_beta_slope.py [options] <cluster>"

long_help = """
Computes regression model for slope of beta lateralization dependent on bias and stimulus strength 
"""

from optparse import OptionParser
from os.path import join
parser = OptionParser ( usage, epilog=long_help )

opts,args = parser.parse_args ()

meta_path = '/home/abraun/meg_data'
data_path = '/home/abraun/meg_data/sr_labeled/'
meg_data_path = '/home/abraun/meg_data/sr_contrasts'


cluster = args[0]
subjects = [2, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 42]
#cluster = "JWG_M1"


signed_stim = []

all_subjects = []


regr_beta_slope_signed_stim_all_sj_up_resp = []
regr_beta_slope_bias_all_sj_up_resp = []
regr_beta_slope_signed_stim_all_sj_down_resp = []
regr_beta_slope_bias_all_sj_down_resp = []

regr_beta_slope_bias_all_sj_pooled_resp = []
regr_beta_slope_signed_stim_all_sj_pooled_resp = []
beta_slope_all_sj = []




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
        
    for session in sessions: 
        if int(subject) == 5 and session == 1:
            recs = [2]
        else:
            recs = [1]

        for rec in recs:
            if int(subject) < 10:
                meta_data_filenames.append(join(meta_path, 'P0{}/MEG/Locked/P0{}-S{}_rec{}_stim_new.hdf'.format(subject, subject, session, rec)))
            elif int(subject) >= 10:
                meta_data_filenames.append(join(meta_path, 'P{}/MEG/Locked/P{}-S{}_rec{}_stim_new.hdf'.format(subject, subject, session, rec)))

        meg_data_filenames.append(join(meg_data_path, 'S{}-lh_is_ipsi_single_trial_session_{}_new.hdf'.format(subject, session)))         

    meg_data = pd.concat([pd.read_hdf(meg_data_filename, sep = '\t') for meg_data_filename in meg_data_filenames])
    meta_data = pd.concat(
        [pd.read_hdf(meta_data_filename, 'df')
            for meta_data_filename in meta_data_filenames])
   
    meta_data['respbutton'][meta_data['respbutton'] == 75] = 12 # fix mixed codings for respbuttons
    meta_data['respbutton'][meta_data['respbutton'] == 77] = 18 # fix mixed codings for respbuttons

    meta_data['respbutton'][meta_data['respbutton'] == 8] = 12 # fix mixed codings for respbuttons
    meta_data['respbutton'][meta_data['respbutton'] == 1] = 18 # fix mixed codings for respbuttons

    single_trial_bias_all, meta_data_all = rf.get_single_trial_bias(subject, "all", meta_data)
    meta_data_all = meta_data_all.sort_values('idx')
    
    signed_stim = meta_data_all.motion_coherence.values*meta_data_all.stimtype.values
    signed_stim_pooled_choice = signed_stim
    
    single_trial_bias_all = single_trial_bias_all.sort_values('idx')
    single_trial_bias_choice = single_trial_bias_all.sort_values('idx').single_trial_bias 
    single_trial_bias_pooled_choice = single_trial_bias_choice
    trials_up_resp = meta_data_all.query('resptype == 1')['idx']
    trials_down_resp = meta_data_all.query('resptype == -1')['idx']

    meg_data_up_resp = meg_data[meg_data.index.get_level_values('trial').isin(trials_up_resp)]
    meg_data_down_resp = meg_data[meg_data.index.get_level_values('trial').isin(trials_down_resp)]
    meg_data_pooled_choice = pd.concat([meg_data_up_resp, -meg_data_down_resp])
    
    M1_beta_lat_up_resp = meg_data_up_resp.query('cluster == "%s" & 12 <= freq <= 36 & epoch == "stimulus" & contrast == "hand" & hemi == "lh_is_ipsi"' % cluster).groupby('trial').mean()
    M1_beta_lat_down_resp = meg_data_down_resp.query('cluster == "%s" & 12 <= freq <= 36 & epoch == "stimulus" & contrast == "hand" & hemi == "lh_is_ipsi"' % cluster).groupby('trial').mean()
    
    times = meg_data.columns    
    start_plot = np.min(np.where(times >= -0.2275)) #start to plot from mean of baseline
    stop_plot = np.min(np.where(times >= 0.75 + 0.433333)) #end to plot at median rt plus 100 ms
    
    freqs = np.unique(meg_data.index.get_level_values('freq'))

    signed_stim_up_resp = meta_data_all.query('resptype == 1').motion_coherence.values*meta_data_all.query('resptype == 1').stimtype.values
    single_trial_bias_up_resp = single_trial_bias_all[single_trial_bias_all.idx.isin(trials_up_resp)].single_trial_bias
    X_up_resp_zscore = pd.DataFrame({'signed_stim_up_resp': scipy.stats.zscore(signed_stim_up_resp), 'single_trial_bias_up_resp': scipy.stats.zscore(single_trial_bias_up_resp)})  
    X_up_resp_zscore = X_up_resp_zscore[['signed_stim_up_resp', 'single_trial_bias_up_resp']]

    signed_stim_down_resp = meta_data_all.query('resptype == -1').motion_coherence.values*meta_data_all.query('resptype == -1').stimtype.values
    single_trial_bias_down_resp = single_trial_bias_all[single_trial_bias_all.idx.isin(trials_down_resp)].single_trial_bias
    X_down_resp_zscore = pd.DataFrame({'signed_stim_down_resp': scipy.stats.zscore(signed_stim_down_resp), 'single_trial_bias_down_resp': scipy.stats.zscore(single_trial_bias_down_resp)})    
    X_down_resp_zscore = X_down_resp_zscore[['signed_stim_down_resp', 'single_trial_bias_down_resp']] 
    
    X_pooled_choice_zscore = pd.DataFrame({'signed_stim_pooled': scipy.stats.zscore(signed_stim_pooled_choice), 'single_trial_bias_pooled': scipy.stats.zscore(single_trial_bias_pooled_choice)})
    
    beta_slope = rf.compute_beta_slope(meg_data, cluster, start_plot, stop_plot)

    beta_slope_all_sj.append(np.mean(beta_slope, axis = 0))

    regr_beta_slope_signed_stim_up_resp_zscore, regr_beta_slope_bias_up_resp_zscore = rf.compute_reg_beta_slope(X_up_resp_zscore, meg_data_up_resp, cluster, start_plot, stop_plot)
    regr_beta_slope_signed_stim_down_resp_zscore, regr_beta_slope_bias_down_resp_zscore = rf.compute_reg_beta_slope(X_down_resp_zscore, meg_data_down_resp, cluster, start_plot, stop_plot)
    
    regr_beta_slope_signed_stim_pooled_resp_zscore, regr_beta_slope_bias_pooled_resp_zscore = rf.compute_reg_beta_slope(X_pooled_choice_zscore, meg_data_pooled_choice, cluster, start_plot, stop_plot)
    
    all_subjects.extend(np.ones(len(freqs))*subject)

    regr_beta_slope_bias_all_sj_down_resp.append(regr_beta_slope_bias_down_resp_zscore)    
    regr_beta_slope_signed_stim_all_sj_down_resp.append(regr_beta_slope_signed_stim_down_resp_zscore)
    regr_beta_slope_bias_all_sj_up_resp.append(regr_beta_slope_bias_up_resp_zscore)    
    regr_beta_slope_signed_stim_all_sj_up_resp.append(regr_beta_slope_signed_stim_up_resp_zscore)

    regr_beta_slope_bias_all_sj_pooled_resp.append(regr_beta_slope_bias_pooled_resp_zscore)    
    regr_beta_slope_signed_stim_all_sj_pooled_resp.append(regr_beta_slope_signed_stim_pooled_resp_zscore)


df_all_sj_beta_slope = pd.DataFrame(beta_slope_all_sj, columns = times[start_plot: stop_plot])
df_all_sj_beta_slope.loc[:, 'subject'] = subjects
df_all_sj_beta_slope.set_index(['subject'], append=True, inplace=True)
df_all_sj_beta_slope.to_csv('beta_slope_sliding_window_new_' + cluster + '.csv', sep = '\t')

df_all_sj_regr_beta_slope_bias_down_resp = pd.DataFrame(regr_beta_slope_bias_all_sj_down_resp, columns = times[start_plot: stop_plot])
df_all_sj_regr_beta_slope_bias_down_resp.loc[:, 'subject'] = subjects
df_all_sj_regr_beta_slope_bias_down_resp.set_index(['subject'], append=True, inplace=True)

df_all_sj_regr_beta_slope_signed_stim_down_resp = pd.DataFrame(regr_beta_slope_signed_stim_all_sj_down_resp, columns = times[start_plot: stop_plot])
df_all_sj_regr_beta_slope_signed_stim_down_resp.loc[:, 'subject'] = subjects
df_all_sj_regr_beta_slope_signed_stim_down_resp.set_index(['subject'], append=True, inplace=True)

df_all_sj_regr_beta_slope_bias_up_resp = pd.DataFrame(regr_beta_slope_bias_all_sj_up_resp, columns = times[start_plot: stop_plot])
df_all_sj_regr_beta_slope_bias_up_resp.loc[:, 'subject'] = subjects
df_all_sj_regr_beta_slope_bias_up_resp.set_index(['subject'], append=True, inplace=True)

df_all_sj_regr_beta_slope_signed_stim_up_resp = pd.DataFrame(regr_beta_slope_signed_stim_all_sj_up_resp, columns = times[start_plot: stop_plot])
df_all_sj_regr_beta_slope_signed_stim_up_resp.loc[:, 'subject'] = subjects
df_all_sj_regr_beta_slope_signed_stim_up_resp.set_index(['subject'], append=True, inplace=True)

df_all_sj_regr_beta_slope_bias_pooled_resp = pd.DataFrame(regr_beta_slope_bias_all_sj_pooled_resp, columns = times[start_plot: stop_plot])
df_all_sj_regr_beta_slope_bias_pooled_resp.loc[:, 'subject'] = subjects
df_all_sj_regr_beta_slope_bias_pooled_resp.set_index(['subject'], append=True, inplace=True)

df_all_sj_regr_beta_slope_signed_stim_pooled_resp = pd.DataFrame(regr_beta_slope_signed_stim_all_sj_pooled_resp, columns = times[start_plot: stop_plot])
df_all_sj_regr_beta_slope_signed_stim_pooled_resp.loc[:, 'subject'] = subjects
df_all_sj_regr_beta_slope_signed_stim_pooled_resp.set_index(['subject'], append=True, inplace=True)


df_all_sj_regr_beta_slope_bias_up_resp.to_csv('regr_beta_slope_bias_up_resp_sliding_window_new_zscore_' + cluster + '.csv', sep = '\t')
df_all_sj_regr_beta_slope_bias_down_resp.to_csv('regr_beta_slope_bias_down_resp_sliding_window_new_zscore_' + cluster + '.csv', sep = '\t')

df_all_sj_regr_beta_slope_signed_stim_up_resp.to_csv('regr_beta_slope_signed_stim_up_resp_sliding_window_new_zscore_' + cluster + '.csv', sep = '\t')
df_all_sj_regr_beta_slope_signed_stim_down_resp.to_csv('regr_beta_slope_signed_stim_down_resp_sliding_window_new_zscore_' + cluster + '.csv', sep = '\t')

df_all_sj_regr_beta_slope_bias_pooled_resp.to_csv('regr_beta_slope_bias_pooled_resp_sliding_window_new_zscore_' + cluster + '.csv', sep = '\t')
df_all_sj_regr_beta_slope_signed_stim_pooled_resp.to_csv('regr_beta_slope_signed_stim_pooled_resp_sliding_window_new_zscore_' + cluster + '.csv', sep = '\t')

