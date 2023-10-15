import os
import pandas as pd
import numpy as np
import pdb
from sklearn import linear_model
import scipy


from optparse import OptionParser

regr = linear_model.LinearRegression()


def get_single_trial_bias(subject, condition, meta_data_condition):
    ################# load single trial biases #########################################################################

    single_trial_bias_condition = []

    def load_fold(subject, condition, fold):
        single_trial_bias_fold = pd.read_csv(
            '/home/abraun/meg_anke_behavdata/crossvalidation/P'
            + str(subject)
            + '_'
            + condition.lower()
            + '_meg_behav_unique_blocks_test_fold_'
            + str(fold)
            + '.single_trial_bias_incl_general_bias.csv',
            sep='\t',
        )
        return single_trial_bias_fold

    for i in range(1, 7):
        if condition == "all":
            if os.path.isfile(
                os.path.join(
                    '/home/abraun/meg_anke_behavdata/crossvalidation/P'
                    + str(subject)
                    + '_'
                    + "Repetitive".lower()
                    + '_meg_behav_unique_blocks_test_fold_'
                    + str(i)
                    + '.single_trial_bias_incl_general_bias.csv'
                )
            ):
                single_trial_bias_fold_rep = load_fold(subject, "Repetitive", i)
                single_trial_bias_fold_rep.loc[:, 'single_trial_bias'] = single_trial_bias_fold_rep[
                    'single_trial_bias_' + "Repetitive".lower()
                ]
                single_trial_bias_condition.append(single_trial_bias_fold_rep)
            if os.path.isfile(
                os.path.join(
                    '/home/abraun/meg_anke_behavdata/crossvalidation/P'
                    + str(subject)
                    + '_'
                    + "Alternating".lower()
                    + '_meg_behav_unique_blocks_test_fold_'
                    + str(i)
                    + '.single_trial_bias_incl_general_bias.csv'
                )
            ):
                single_trial_bias_fold_alt = load_fold(subject, "Alternating", i)
                single_trial_bias_fold_alt.loc[:, 'single_trial_bias'] = single_trial_bias_fold_alt[
                    'single_trial_bias_' + "Alternating".lower()
                ]
                single_trial_bias_condition.append(single_trial_bias_fold_alt)
            if os.path.isfile(
                os.path.join(
                    '/home/abraun/meg_anke_behavdata/crossvalidation/P'
                    + str(subject)
                    + '_'
                    + "Neutral".lower()
                    + '_meg_behav_unique_blocks_test_fold_'
                    + str(i)
                    + '.single_trial_bias_incl_general_bias.csv'
                )
            ):
                single_trial_bias_fold_neutr = load_fold(subject, "Neutral", i)
                single_trial_bias_fold_neutr.loc[:, 'single_trial_bias'] = single_trial_bias_fold_neutr[
                    'single_trial_bias_' + "Neutral".lower()
                ]
                single_trial_bias_condition.append(single_trial_bias_fold_neutr)
        elif condition == "biased":
            if os.path.isfile(
                os.path.join(
                    '/home/abraun/meg_anke_behavdata/crossvalidation/P'
                    + str(subject)
                    + '_'
                    + "Repetitive".lower()
                    + '_meg_behav_unique_blocks_test_fold_'
                    + str(i)
                    + '.single_trial_bias_incl_general_bias.csv'
                )
            ):
                single_trial_bias_fold_rep = load_fold(subject, "Repetitive", i)
                single_trial_bias_fold_rep.loc[:, 'single_trial_bias'] = single_trial_bias_fold_rep[
                    'single_trial_bias_' + "Repetitive".lower()
                ]
                single_trial_bias_condition.append(single_trial_bias_fold_rep)
            if os.path.isfile(
                os.path.join(
                    '/home/abraun/meg_anke_behavdata/crossvalidation/P'
                    + str(subject)
                    + '_'
                    + "Alternating".lower()
                    + '_meg_behav_unique_blocks_test_fold_'
                    + str(i)
                    + '.single_trial_bias_incl_general_bias.csv'
                )
            ):
                single_trial_bias_fold_alt = load_fold(subject, "Alternating", i)
                single_trial_bias_fold_alt.loc[:, 'single_trial_bias'] = single_trial_bias_fold_alt[
                    'single_trial_bias_' + "Alternating".lower()
                ]
                single_trial_bias_condition.append(single_trial_bias_fold_alt)
        else:
            if os.path.isfile(
                os.path.join(
                    '/home/abraun/meg_anke_behavdata/crossvalidation/P'
                    + str(subject)
                    + '_'
                    + condition.lower()
                    + '_meg_behav_unique_blocks_test_fold_'
                    + str(i)
                    + '.single_trial_bias_incl_general_bias.csv'
                )
            ):
                single_trial_bias_fold = load_fold(subject, condition, i)
                single_trial_bias_fold.loc[:, 'single_trial_bias'] = single_trial_bias_fold[
                    'single_trial_bias_' + condition.lower()
                ]
                single_trial_bias_condition.append(single_trial_bias_fold)

    single_trial_bias_condition = pd.concat(single_trial_bias_condition).sort_values('idx')
    idx = single_trial_bias_condition['idx']
    meta_data_idx = meta_data_condition[(meta_data_condition['idx']).isin(idx)]
    trials = meta_data_idx['idx']
    single_trial_bias_condition = single_trial_bias_condition[(single_trial_bias_condition['idx']).isin(trials)]
    single_trial_bias_condition = single_trial_bias_condition.sort_values('idx')

    return single_trial_bias_condition, meta_data_idx


def get_slope(M1_beta_lat, lower_idx):
    times = M1_beta_lat.columns
    upper_idx = lower_idx + 8  # 8 steps refer to 200 ms
    x = np.array(times[lower_idx:upper_idx])
    A = np.vstack([x, np.ones(len(x))]).T
    m_single_trials = []
    trials = np.unique(M1_beta_lat.index.get_level_values('trial'))
    y_all_trials = M1_beta_lat.mean(axis=0).values[lower_idx:upper_idx]
    m_all_trials, c_all_trials = np.linalg.lstsq(A, y_all_trials, rcond=None)[0]
    for trial in trials:
        y_single_trial = M1_beta_lat.query('~(trial == {})'.format(trial)).mean(axis=0).values[lower_idx:upper_idx]
        m_single_trial, c_single_trial = np.linalg.lstsq(A, y_single_trial, rcond=None)[0]
        m_trial = m_all_trials - m_single_trial
        m_single_trials.append(m_trial)
    return m_single_trials


def compute_beta_slope(meg_data_, cluster, start_plot, stop_plot):
    beta_slope_ = np.zeros(
        (
            len(
                meg_data_.query(
                    '12 <= freq <= 36 & cluster == "%s" & contrast == "hand" & epoch == "stimulus" & hemi == "lh_is_ipsi"'
                    % cluster
                )
                .groupby(['subject', 'trial'])
                .mean()
            ),
            len(range(start_plot, stop_plot)),
        )
    )
    time_index = -1
    for time_idx in range(start_plot, stop_plot):
        time_index = time_idx - start_plot
        beta_lat = (
            meg_data_.query(
                '12 <= freq <= 36 & cluster == "%s" & contrast == "hand" & epoch == "stimulus" & hemi == "lh_is_ipsi"'
                % cluster
            )
            .groupby(['subject', 'trial'])
            .mean()
        )
        M1_slope = get_slope(beta_lat, time_idx)
        beta_slope_[:, time_index] = M1_slope
    return beta_slope_


def compute_reg_beta_slope(X_, meg_data_, cluster, start_plot, stop_plot):
    regr_beta_signed_stim = np.zeros(len(range(start_plot, stop_plot)))
    regr_beta_bias = np.zeros(len(range(start_plot, stop_plot)))
    time_index = -1
    for time_idx in range(start_plot, stop_plot):
        time_index = time_idx - start_plot
        beta_lat = (
            meg_data_.query(
                '12 <= freq <= 36 & cluster == "%s" & contrast == "hand" & epoch == "stimulus" & hemi == "lh_is_ipsi"'
                % cluster
            )
            .groupby(['subject', 'trial'])
            .mean()
        )
        M1_slope = get_slope(beta_lat, time_idx)
        Y_ = pd.DataFrame({'M1_slopes': scipy.stats.zscore(M1_slope)})
        regr.fit(X_, Y_)
        regr_beta_signed_stim[time_index] = regr.coef_[0, 0]
        regr_beta_bias[time_index] = regr.coef_[0, 1]
    #   pdb.set_trace()
    return regr_beta_signed_stim, regr_beta_bias


def compute_reg_beta(X_, meg_data_, cluster, start_plot, stop_plot):
    regr_beta_signed_stim = np.zeros(len(range(start_plot, stop_plot)))
    regr_beta_bias = np.zeros(len(range(start_plot, stop_plot)))
    time_index = -1
    for time_idx in range(start_plot, stop_plot):
        time_index = time_idx - start_plot
        Y_ = pd.DataFrame(
            {
                'M1_beta_lat': scipy.stats.zscore(
                    meg_data_.query(
                        '12 <= freq <= 36 & cluster == "%s" & contrast == "hand" & epoch == "stimulus" & hemi == "lh_is_ipsi"'
                        % cluster
                    )
                    .groupby(['subject', 'trial'])
                    .mean()
                    .values[:, time_idx]
                )
            }
        )
        regr.fit(X_, Y_)
        regr_beta_signed_stim[time_index] = regr.coef_[0, 0]
        regr_beta_bias[time_index] = regr.coef_[0, 1]
    return regr_beta_signed_stim, regr_beta_bias
