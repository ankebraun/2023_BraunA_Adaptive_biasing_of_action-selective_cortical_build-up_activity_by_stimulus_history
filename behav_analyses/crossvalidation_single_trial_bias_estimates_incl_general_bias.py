__doc__ = """
Compute single trial bias estimates.
In order to run this file, you will need to first run the "analysis*.py" scripts. This will
generate backup files in the folder sim_backup.

This code was adapted from Fründ, Wichmann, Macke (2014): Quantifying the effect of inter-trial dependence on perceptual decisions. J Vis, 14(7): 9. 
See Copyright notice below

Copyright (C) 2014 Ingo Fründ

This code reproduces the analyses in the paper

    Fründ, Wichmann, Macke (2014): Quantifying the effect of inter-trial dependence on perceptual decisions. J Vis, 14(7): 9.


    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    If you use the Software for your own research, cite the paper.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import numpy as np
import os
import cPickle
import pandas as pd
import pdb


def generate_list_of_sj(condition, df):
    all_observers_training_fold_1 = [i.replace('_', '') + '_' + condition + '_meg_behav_unique_blocks_' + 'training_fold_1' + '.csv'\
    for i in df['subjects'] if os.path.isfile(os.path.join( "sim_backup",i.replace('_', '') + '_' + condition + '_meg_behav_unique_blocks_'\
     + 'training_fold_1' + '.csv' + df.query('subjects == "{}"'.format(i)).model_order.values[0][-7:] + ".pcl" )) ]
    all_observers_training_fold_2 = [i.replace('_', '') + '_' + condition + '_meg_behav_unique_blocks_' + 'training_fold_2' + '.csv'\
    for i in df['subjects'] if os.path.isfile(os.path.join( "sim_backup",i.replace('_', '') + '_' + condition + '_meg_behav_unique_blocks_'\
     + 'training_fold_2' + '.csv' + df.query('subjects == "{}"'.format(i)).model_order.values[0][-7:] + ".pcl" )) ]
    all_observers_training_fold_3 = [i.replace('_', '') + '_' + condition + '_meg_behav_unique_blocks_' + 'training_fold_3' + '.csv'\
    for i in df['subjects'] if os.path.isfile(os.path.join( "sim_backup",i.replace('_', '') + '_' + condition + '_meg_behav_unique_blocks_'\
     + 'training_fold_3' + '.csv' + df.query('subjects == "{}"'.format(i)).model_order.values[0][-7:] + ".pcl" )) ]
    all_observers_training_fold_4 = [i.replace('_', '') + '_' + condition + '_meg_behav_unique_blocks_' + 'training_fold_4' + '.csv'\
    for i in df['subjects'] if os.path.isfile(os.path.join( "sim_backup",i.replace('_', '') + '_' + condition + '_meg_behav_unique_blocks_'\
     + 'training_fold_4' + '.csv' + df.query('subjects == "{}"'.format(i)).model_order.values[0][-7:] + ".pcl" )) ]
    all_observers_training_fold_5 = [i.replace('_', '') + '_' + condition + '_meg_behav_unique_blocks_' + 'training_fold_5' + '.csv'\
    for i in df['subjects'] if os.path.isfile(os.path.join( "sim_backup",i.replace('_', '') + '_' + condition + '_meg_behav_unique_blocks_'\
     + 'training_fold_5' + '.csv' + df.query('subjects == "{}"'.format(i)).model_order.values[0][-7:] + ".pcl" )) ]
    all_observers_training_fold_6 = [i.replace('_', '') + '_' + condition + '_meg_behav_unique_blocks_' + 'training_fold_6' + '.csv'\
    for i in df['subjects'] if os.path.isfile(os.path.join( "sim_backup",i.replace('_', '') + '_' + condition + '_meg_behav_unique_blocks_'\
     + 'training_fold_6' + '.csv' + df.query('subjects == "{}"'.format(i)).model_order.values[0][-7:] + ".pcl" )) ]
    l = [all_observers_training_fold_1, all_observers_training_fold_2, all_observers_training_fold_3, all_observers_training_fold_4, all_observers_training_fold_5, all_observers_training_fold_6]
    all_observers_training = [item for sublist in l for item in sublist]
    return all_observers_training


def get_weights_training(observer_training, df):
    model_order = df.query('subjects == "{}"'.format(observer_training[0:3])).model_order.values[0][-7:]
    backup_file = os.path.join ( "sim_backup",observer_training+ model_order + ".pcl" )
    results_ =  cPickle.load ( open ( backup_file, 'r' ) )
    M = results_['model_w_hist']
    out = np.zeros(len(M.w[M.hf0:]) + 2)
    w = M.w[M.hf0:] # history weights
    alpha = M.w[range(1,M.hf0)].mean() #average slope across sessions
    out[0] = M.w[0] # general bias
    out[1] = alpha # average slope
    out[2:] = w # history weights
    weights = out
    pi = M.pi # lapse rate
    return weights, pi, model_order


def get_params_test(observer_test, model_order):
    if model_order == "_1_lags":
        from intertrial_no_bootstrap_1_lags import util
    elif model_order == "_2_lags":
        from intertrial_no_bootstrap_2_lags import util
    elif model_order == "_3_lags":
        from intertrial_no_bootstrap_3_lags import util
    elif model_order == "_4_lags":
        from intertrial_no_bootstrap_4_lags import util
    elif model_order == "_5_lags":
        from intertrial_no_bootstrap_5_lags import util
    elif model_order == "_6_lags":
        from intertrial_no_bootstrap_6_lags import util
    elif model_order == "_7_lags":
        from intertrial_no_bootstrap_7_lags import util
    data,w0,plotinfo = util.load_data_file (observer_test)
    return data.X


def combine_features_history ( X, w):
    """Combine features linearly and apply logistic

    :Parameters:
        *X_thres*
            design matrix (after potential application of the threshold)
        *w*
            feature weights
    """
    eta = np.dot( X[:,0], w[0]) + np.dot ( X[:,2:], w[2:] )
    return eta


def get_single_trial_bias(test_design_matrices_test_fold, weights_training_fold):
    if np.shape(weights_training_fold)[0] == np.shape(test_design_matrices_test_fold)[1]:
        single_trial_bias_fold = combine_features_history ( test_design_matrices_test_fold, weights_training_fold)
    else:
        single_trial_bias_fold = np.zeros(np.shape(test_design_matrices_test_fold)[0])
    return single_trial_bias_fold


df_alternating = pd.read_csv('likelihood_model_comparison_alternating_without_no_hist.csv', sep = '\t')
df_repetitive = pd.read_csv('likelihood_model_comparison_repetitive_without_no_hist.csv', sep = '\t')
df_neutral = pd.read_csv('likelihood_model_comparison_neutral_without_no_hist.csv', sep = '\t')

df_alternating = df_alternating[~df_alternating.model_order.isnull()]
df_repetitive = df_repetitive[~df_repetitive.model_order.isnull()]
df_neutral = df_neutral[~df_neutral.model_order.isnull()]

all_observers_alternating_training = generate_list_of_sj('alternating', df_alternating)
all_observers_repetitive_training = generate_list_of_sj('repetitive', df_repetitive)
all_observers_neutral_training = generate_list_of_sj('neutral', df_neutral)

for j,o in enumerate(all_observers_alternating_training):
    o_test = o.replace('training', 'test')
    df_test =  pd.read_csv(o_test.replace('.csv', '_idx.csv'), sep = '\t')
    weights_training, pi_training, model_order = get_weights_training(o, df_alternating)
    test_design_matrices_test = get_params_test(o_test, model_order)
    single_trial_bias = get_single_trial_bias(test_design_matrices_test, weights_training)
    df = pd.DataFrame({'idx': df_test["idx"],
                       'single_trial_bias_alternating': single_trial_bias})
    df.to_csv(o_test[:-3] + 'single_trial_bias_incl_general_bias.csv', sep = '\t', header = True, encoding = 'utf-8')


for j,o in enumerate(all_observers_repetitive_training):
    o_test = o.replace('training', 'test')
    df_test =  pd.read_csv(o_test, sep = '\t')
    df_test_idx =  pd.read_csv(o_test.replace('.csv', '_idx.csv'), sep = '\t')
    weights_training, pi_training, model_order = get_weights_training(o, df_repetitive)
    test_design_matrices_test = get_params_test(o_test, model_order)
    single_trial_bias = get_single_trial_bias(test_design_matrices_test, weights_training)
    df = pd.DataFrame({'idx': df_test_idx["idx"],
                       'single_trial_bias_repetitive': single_trial_bias})
    df.to_csv(o_test[:-3] + 'single_trial_bias_incl_general_bias.csv', sep = '\t', header = True, encoding = 'utf-8')

for j,o in enumerate(all_observers_neutral_training):
    o_test = o.replace('training', 'test')
    df_test =  pd.read_csv(o_test, sep = '\t')
    df_test_idx =  pd.read_csv(o_test.replace('.csv', '_idx.csv'), sep = '\t')
    weights_training, pi_training, model_order = get_weights_training(o, df_neutral)
    test_design_matrices_test = get_params_test(o_test, model_order)
    single_trial_bias = get_single_trial_bias(test_design_matrices_test, weights_training)
    df = pd.DataFrame({'idx': df_test_idx["idx"],
                       'single_trial_bias_neutral': single_trial_bias})
    df.to_csv(o_test[:-3] + 'single_trial_bias_incl_general_bias.csv', sep = '\t', header = True, encoding = 'utf-8')




