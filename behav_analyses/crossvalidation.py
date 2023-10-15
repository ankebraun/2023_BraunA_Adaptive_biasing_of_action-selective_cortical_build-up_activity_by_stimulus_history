__doc__ = """
Compute log likelihood values for crossvaldiation.

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


import os
import numpy as np
import pdb
import cPickle
import pandas as pd

num_lags = 1
if num_lags == 1:
    from intertrial_no_bootstrap_1_lags import util
elif num_lags == 2:
    from intertrial_no_bootstrap_2_lags import util
elif num_lags == 3:
    from intertrial_no_bootstrap_3_lags import util
elif num_lags == 4:
    from intertrial_no_bootstrap_4_lags import util
elif num_lags == 5:
    from intertrial_no_bootstrap_5_lags import util
elif num_lags == 6:
    from intertrial_no_bootstrap_6_lags import util
elif num_lags == 7:
    from intertrial_no_bootstrap_7_lags import util

observers = range(2,43)
def generate_list_of_sj(fold, num_lags):
    all_observers = []
    for i in observers:
        observer = 'P' + str(i) + '_repetitive_meg_behav_unique_blocks_training_fold_' + str(fold) + '.csv'
        backup_file = os.path.join ( "sim_backup",observer+"_" + str(num_lags) + "_lags.pcl" )
        if not os.path.isfile(backup_file):
            continue
        all_observers.append(observer)
    return all_observers


def generate_list_of_test_sj(observers_training):
    observers_test = [i.replace('training', 'test') for i in observers_training]
    return observers_test


def get_weights_training(all_observers_training, num_lags):
    weights = []
    pi = []

    weights_no_hist = []
    pi_no_hist = []

    for j, o in enumerate(all_observers_training):
        backup_file = os.path.join ( "sim_backup",o+"_" + str(num_lags) + "_lags.pcl" )
        results_ =  cPickle.load ( open ( backup_file, 'r' ) )
        M = results_['model_w_hist']
        M_no_hist = results_['model_nohist']
        out = np.zeros(len(M.w[M.hf0:]) + 2)
        w = M.w[M.hf0:] # history weights
        alpha = M.w[range(1,M.hf0)].mean() #average slope across sessions
        out[0] = M.w[0] # general bias
        out[1] = alpha # average slope
        out[2:] = w # history weights
        weights.append(out)
        pi.append(M.pi) # lapse rate
        out_no_hist = np.zeros(2)
        alpha_no_hist = M_no_hist.w[range(1,M_no_hist.hf0)].mean() #average slope across sessions
        out_no_hist[0] = M_no_hist.w[0] # general bias
        out_no_hist[1] = alpha_no_hist # average slope
        weights_no_hist.append(out_no_hist)
        pi_no_hist.append(M_no_hist.pi) # lapse rate
    return weights, pi, weights_no_hist, pi_no_hist


def get_params_test(all_observers_test):
    test_design_matrices_test_fold = []
    response_test_fold = []
    subjects_test_fold = []
    for i, o in enumerate(all_observers_test):
        data,w0,plotinfo = util.load_data_file (o)
        test_design_matrices_test_fold.append(data.X)
        response_test_fold.append(data.r)
        subjects_test_fold.append(o[:3])
    return test_design_matrices_test_fold, response_test_fold, subjects_test_fold


def combine_features ( X, w ):
    """Combine features linearly and apply logistic

    :Parameters:
        *X*
            design matrix
        *w*
            feature weights
    """
    eta = np.dot ( X, w )
    return logistic ( eta )

def combine_features_no_hist ( X, w ):
    """Combine features linearly and apply logistic

    :Parameters:
        *X*
            design matrix
        *w*
            feature weights
    """
    eta = np.dot ( X[:,:2], w )
    return logistic ( eta )

def logistic ( x ):
    """Logistic function"""
    return 1./( 1.+np.exp(-x) )

#@staticmethod
def likelihood ( gwx, r, p ):
    """Determine the log likelihood of the data r

    :Parameters:
        *gwx*
            response probabilities if the observer would look at the stimulus
            This is a linear combination of the features mapped through a logistic
            function.
        *r*
            response vector
        *q*
            prior state probabilities
    """
    prob1 = p[1] + p[2]*gwx
    return np.sum ( r*np.log(prob1) + (1-r)*np.log1p(-prob1) ) # log1p(x) returns the natural logarithm of 1+x


def get_likelihood(test_design_matrices_test_fold, weights_training_fold, weights_training_no_hist_fold, response_test_fold, pi_training_fold, pi_training_no_hist_fold, subjects_test_fold):
    likelihood_test_fold = []
    likelihood_test_no_hist_fold = []
    for j in range(0, len(test_design_matrices_test_fold)):
        if np.shape(weights_training_fold[j])[0] == np.shape(test_design_matrices_test_fold[j])[1]:
            gwx = combine_features ( test_design_matrices_test_fold[j], weights_training_fold[j] )
            likelihood_test_fold.append(likelihood ( gwx, response_test_fold[j], pi_training_fold[j] ))
        else:
            gwx = float('nan')
            likelihood_test_fold.append(float('nan'))
        gwx_no_hist = combine_features_no_hist ( test_design_matrices_test_fold[j], weights_training_no_hist_fold[j] )
        likelihood_test_no_hist_fold.append(likelihood ( gwx_no_hist, response_test_fold[j], pi_training_no_hist_fold[j] ))
    return likelihood_test_fold, likelihood_test_no_hist_fold


all_observers_training_fold = {}
for fold in range(1,7):
    all_observers_training_fold[fold] = generate_list_of_sj(fold, num_lags)

all_observers_test_fold = {}
for fold in range(1,7):
    all_observers_test_fold[fold] = generate_list_of_test_sj(all_observers_training_fold[fold])


weights_training_fold = {}
pi_training_fold = {}
weights_training_no_hist_fold = {}
pi_training_no_hist_fold = {}
for fold in range(1,7):
    weights_training_fold[fold], pi_training_fold[fold], weights_training_no_hist_fold[fold],\
    pi_training_no_hist_fold[fold] = get_weights_training(all_observers_training_fold[fold], num_lags)

test_design_matrices_test_fold = {}
response_test_fold = {}
subjects_test_fold = {}
for fold in range(1,7):
    test_design_matrices_test_fold[fold], response_test_fold[fold], subjects_test_fold[fold]\
    = get_params_test(all_observers_test_fold[fold])

all_subjects_test = ['P1_', 'P2_', 'P3_', 'P4_', 'P5_', 'P7_', 'P8_', 'P9_', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15',\
                    'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29',\
                    'P30', 'P31', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37', 'P38', 'P39', 'P40', 'P41', 'P42']


likelihood_test_fold = {}
likelihood_test_no_hist_fold = {}
for fold in range(1,7):
    likelihood_test_fold[fold], likelihood_test_no_hist_fold[fold] = get_likelihood(test_design_matrices_test_fold[fold],\
    weights_training_fold[fold], weights_training_no_hist_fold[fold], \
    response_test_fold[fold], pi_training_fold[fold], pi_training_no_hist_fold[fold], subjects_test_fold[fold])


for k, sj in enumerate(all_subjects_test):
    for fold in range(1,7):
        if sj not in subjects_test_fold[fold]:
            likelihood_test_fold[fold].insert(k, float('nan'))
            likelihood_test_no_hist_fold[fold].insert(k, float('nan'))


df_no_hist = pd.DataFrame({'subjects': all_subjects_test,
                   'likelihood_test_no_hist_fold_1': likelihood_test_no_hist_fold[1],
                   'likelihood_test_no_hist_fold_2': likelihood_test_no_hist_fold[2],
                   'likelihood_test_no_hist_fold_3': likelihood_test_no_hist_fold[3],
                   'likelihood_test_no_hist_fold_4': likelihood_test_no_hist_fold[4],
                   'likelihood_test_no_hist_fold_5': likelihood_test_no_hist_fold[5],
                   'likelihood_test_no_hist_fold_6': likelihood_test_no_hist_fold[6]})

df_no_hist['avg'] = df_no_hist.mean(axis=1)
df_no_hist = df_no_hist[['subjects', 'likelihood_test_no_hist_fold_1', 'likelihood_test_no_hist_fold_2',\
            'likelihood_test_no_hist_fold_3', 'likelihood_test_no_hist_fold_4', 'likelihood_test_no_hist_fold_5',\
            'likelihood_test_no_hist_fold_6', 'avg']]

df_no_hist.to_csv('likelihood_repetitive_no_hist.csv', sep = '\t', header = True, encoding = 'utf-8')


df = pd.DataFrame({'subjects': all_subjects_test,
                   'likelihood_test_fold_1': likelihood_test_fold[1],
                   'likelihood_test_fold_2': likelihood_test_fold[2],
                   'likelihood_test_fold_3': likelihood_test_fold[3],
                   'likelihood_test_fold_4': likelihood_test_fold[4],
                   'likelihood_test_fold_5': likelihood_test_fold[5],
                   'likelihood_test_fold_6': likelihood_test_fold[6]})

df['avg'] = df.mean(axis=1)
df = df[['subjects', 'likelihood_test_fold_1', 'likelihood_test_fold_2', 'likelihood_test_fold_3',\
         'likelihood_test_fold_4', 'likelihood_test_fold_5', 'likelihood_test_fold_6', 'avg']]

df.to_csv('likelihood_repetitive_' + str(num_lags) + '_lags.csv', sep = '\t', header = True, encoding = 'utf-8')
