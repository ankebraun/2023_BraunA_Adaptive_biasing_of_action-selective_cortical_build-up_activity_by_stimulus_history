#!/usr/bin/env python
# coding: utf8
__doc__ = """
Get history weights from logistic regression model. In order to run this file, you will need to first run the "analysis.py" script in this folder. 
This will generate backup files in the folder sim_backup.

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
import cPickle
import os
import pdb
import sys
from optparse import OptionParser

import numpy as np
import pandas as pd
import pylab as pl
import scipy
import scipy.io
import seaborn as sns


os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

all_observers = ['P2_neutral_meg_behav_unique_blocks.csv_1_lags', 'P3_neutral_meg_behav_unique_blocks.csv_1_lags', \
'P5_neutral_meg_behav_unique_blocks.csv_1_lags', 'P7_neutral_meg_behav_unique_blocks.csv_4_lags', \
'P9_neutral_meg_behav_unique_blocks.csv_1_lags', 'P10_neutral_meg_behav_unique_blocks.csv_1_lags', \
'P11_neutral_meg_behav_unique_blocks.csv_1_lags', 'P12_neutral_meg_behav_unique_blocks.csv_1_lags', 'P13_neutral_meg_behav_unique_blocks.csv_1_lags', \
'P14_neutral_meg_behav_unique_blocks.csv_5_lags', 'P15_neutral_meg_behav_unique_blocks.csv_1_lags', \
'P16_neutral_meg_behav_unique_blocks.csv_3_lags', 'P17_neutral_meg_behav_unique_blocks.csv_1_lags', 'P18_neutral_meg_behav_unique_blocks.csv_1_lags', \
'P19_neutral_meg_behav_unique_blocks.csv_1_lags', 'P20_neutral_meg_behav_unique_blocks.csv_1_lags', 'P21_neutral_meg_behav_unique_blocks.csv_1_lags', \
'P22_neutral_meg_behav_unique_blocks.csv_2_lags', 'P23_neutral_meg_behav_unique_blocks.csv_1_lags', 'P24_neutral_meg_behav_unique_blocks.csv_1_lags', \
'P25_neutral_meg_behav_unique_blocks.csv_1_lags', 'P26_neutral_meg_behav_unique_blocks.csv_2_lags', 'P27_neutral_meg_behav_unique_blocks.csv_1_lags', \
'P28_neutral_meg_behav_unique_blocks.csv_7_lags', 'P29_neutral_meg_behav_unique_blocks.csv_4_lags', \
'P30_neutral_meg_behav_unique_blocks.csv_2_lags', 'P31_neutral_meg_behav_unique_blocks.csv_1_lags', 'P32_neutral_meg_behav_unique_blocks.csv_2_lags', \
'P33_neutral_meg_behav_unique_blocks.csv_2_lags', 'P34_neutral_meg_behav_unique_blocks.csv_2_lags', \
'P37_neutral_meg_behav_unique_blocks.csv_1_lags', 'P38_neutral_meg_behav_unique_blocks.csv_2_lags', 'P39_neutral_meg_behav_unique_blocks.csv_4_lags', 'P40_neutral_meg_behav_unique_blocks.csv_2_lags', \
'P41_neutral_meg_behav_unique_blocks.csv_1_lags', 'P42_neutral_meg_behav_unique_blocks.csv_4_lags', \
'P2_repetitive_meg_behav_unique_blocks.csv_4_lags', 'P3_repetitive_meg_behav_unique_blocks.csv_2_lags', \
'P5_repetitive_meg_behav_unique_blocks.csv_1_lags', 'P7_repetitive_meg_behav_unique_blocks.csv_1_lags', \
'P9_repetitive_meg_behav_unique_blocks.csv_1_lags', 'P10_repetitive_meg_behav_unique_blocks.csv_2_lags', \
'P11_repetitive_meg_behav_unique_blocks.csv_2_lags', 'P12_repetitive_meg_behav_unique_blocks.csv_1_lags','P13_repetitive_meg_behav_unique_blocks.csv_4_lags', \
'P14_repetitive_meg_behav_unique_blocks.csv_4_lags', 'P15_repetitive_meg_behav_unique_blocks.csv_3_lags', \
'P16_repetitive_meg_behav_unique_blocks.csv_1_lags', 'P17_repetitive_meg_behav_unique_blocks.csv_3_lags', 'P18_repetitive_meg_behav_unique_blocks.csv_2_lags', \
'P19_repetitive_meg_behav_unique_blocks.csv_7_lags', 'P20_repetitive_meg_behav_unique_blocks.csv_3_lags', 'P21_repetitive_meg_behav_unique_blocks.csv_1_lags', \
'P22_repetitive_meg_behav_unique_blocks.csv_3_lags', 'P23_repetitive_meg_behav_unique_blocks.csv_1_lags', 'P24_repetitive_meg_behav_unique_blocks.csv_1_lags', \
'P25_repetitive_meg_behav_unique_blocks.csv_2_lags', 'P26_repetitive_meg_behav_unique_blocks.csv_2_lags', 'P27_repetitive_meg_behav_unique_blocks.csv_2_lags', \
'P28_repetitive_meg_behav_unique_blocks.csv_1_lags', 'P29_repetitive_meg_behav_unique_blocks.csv_1_lags', \
'P30_repetitive_meg_behav_unique_blocks.csv_1_lags', 'P31_repetitive_meg_behav_unique_blocks.csv_3_lags', 'P32_repetitive_meg_behav_unique_blocks.csv_1_lags', \
'P33_repetitive_meg_behav_unique_blocks.csv_7_lags', 'P34_repetitive_meg_behav_unique_blocks.csv_2_lags', \
'P37_repetitive_meg_behav_unique_blocks.csv_2_lags', 'P38_repetitive_meg_behav_unique_blocks.csv_2_lags', 'P39_repetitive_meg_behav_unique_blocks.csv_2_lags', \
'P40_repetitive_meg_behav_unique_blocks.csv_2_lags', 'P41_repetitive_meg_behav_unique_blocks.csv_1_lags', 'P42_repetitive_meg_behav_unique_blocks.csv_3_lags',\
'P2_alternating_meg_behav_unique_blocks.csv_1_lags', 'P3_alternating_meg_behav_unique_blocks.csv_1_lags', \
'P5_alternating_meg_behav_unique_blocks.csv_2_lags', 'P7_alternating_meg_behav_unique_blocks.csv_5_lags', \
'P9_alternating_meg_behav_unique_blocks.csv_3_lags', 'P10_alternating_meg_behav_unique_blocks.csv_2_lags', \
'P11_alternating_meg_behav_unique_blocks.csv_1_lags', 'P12_alternating_meg_behav_unique_blocks.csv_2_lags', 'P13_alternating_meg_behav_unique_blocks.csv_1_lags', \
'P14_alternating_meg_behav_unique_blocks.csv_2_lags', 'P15_alternating_meg_behav_unique_blocks.csv_1_lags', \
'P16_alternating_meg_behav_unique_blocks.csv_1_lags', 'P17_alternating_meg_behav_unique_blocks.csv_1_lags', 'P18_alternating_meg_behav_unique_blocks.csv_1_lags', \
'P19_alternating_meg_behav_unique_blocks.csv_1_lags', 'P20_alternating_meg_behav_unique_blocks.csv_1_lags', 'P21_alternating_meg_behav_unique_blocks.csv_1_lags', \
'P22_alternating_meg_behav_unique_blocks.csv_1_lags', 'P23_alternating_meg_behav_unique_blocks.csv_1_lags', 'P24_alternating_meg_behav_unique_blocks.csv_1_lags', \
'P25_alternating_meg_behav_unique_blocks.csv_1_lags', 'P26_alternating_meg_behav_unique_blocks.csv_2_lags', 'P27_alternating_meg_behav_unique_blocks.csv_1_lags', \
'P28_alternating_meg_behav_unique_blocks.csv_1_lags', 'P29_alternating_meg_behav_unique_blocks.csv_1_lags', \
'P30_alternating_meg_behav_unique_blocks.csv_2_lags', 'P31_alternating_meg_behav_unique_blocks.csv_1_lags', 'P32_alternating_meg_behav_unique_blocks.csv_1_lags', \
'P33_alternating_meg_behav_unique_blocks.csv_3_lags', 'P34_alternating_meg_behav_unique_blocks.csv_3_lags', \
'P37_alternating_meg_behav_unique_blocks.csv_1_lags', 'P38_alternating_meg_behav_unique_blocks.csv_1_lags', 'P39_alternating_meg_behav_unique_blocks.csv_1_lags', \
'P40_alternating_meg_behav_unique_blocks.csv_4_lags', 'P41_alternating_meg_behav_unique_blocks.csv_1_lags', 'P42_alternating_meg_behav_unique_blocks.csv_2_lags']


observer={}
for j,o in enumerate ( all_observers ):
    observer["observer{0}".format(j)] = all_observers[j] 
backup_file = {}
for j,o in enumerate ( all_observers ):
    backup_file["backup_file{0}".format(j)] = os.path.join ( "sim_backup",observer["observer{0}".format(j)] +".pcl" )
results = {}
for j,o in enumerate ( all_observers ):
   results["results{0}".format(j)] = cPickle.load ( open ( backup_file["backup_file{0}".format(j)], 'r' ) )
data = {}
w0 = {}
plotinfo = {}
for j, o in enumerate ( all_observers ):
    if all_observers[j][-6:] == "1_lags":
        from intertrial_no_bootstrap_1_lags import threshold,history,statistics,graphics,model,util
    elif all_observers[j][-6:] == "2_lags":
        from intertrial_no_bootstrap_2_lags import threshold,history,statistics,graphics,model,util
    elif all_observers[j][-6:] == "3_lags":
        from intertrial_no_bootstrap_3_lags import threshold,history,statistics,graphics,model,util
    elif all_observers[j][-6:] == "4_lags":
        from intertrial_no_bootstrap_4_lags import threshold,history,statistics,graphics,model,util
    elif all_observers[j][-6:] == "5_lags":
        from intertrial_no_bootstrap_5_lags import threshold,history,statistics,graphics,model,util
    elif all_observers[j][-6:] == "6_lags":
        from intertrial_no_bootstrap_6_lags import threshold,history,statistics,graphics,model,util
    elif all_observers[j][-6:] == "7_lags":
        from intertrial_no_bootstrap_7_lags import threshold,history,statistics,graphics,model,util
    data["data{0}".format(j)],w0["w0{0}".format(j)],plotinfo["plotinfo{0}".format(j)] = util.load_data_file (all_observers[j][:-7])

if os.path.exists ( 'sim_backup/all_kernels_wo_nohist_new.pcl'):
    print "Loading kernels"
    kernels = cPickle.load ( open ('sim_backup/all_kernels_wo_nohist_new.pcl', 'r' ) )

    print "kernels:", kernels
else:
    kernels = []
    for j, o in enumerate(all_observers):
        if all_observers[j][-6:] == "1_lags":
            from intertrial_no_bootstrap_1_lags import threshold,history,statistics,graphics,model,util
        elif all_observers[j][-6:] == "2_lags":
            from intertrial_no_bootstrap_2_lags import threshold,history,statistics,graphics,model,util
        elif all_observers[j][-6:] == "3_lags":
            from intertrial_no_bootstrap_3_lags import threshold,history,statistics,graphics,model,util
        elif all_observers[j][-6:] == "4_lags":
            from intertrial_no_bootstrap_4_lags import threshold,history,statistics,graphics,model,util
        elif all_observers[j][-6:] == "5_lags":
            from intertrial_no_bootstrap_5_lags import threshold,history,statistics,graphics,model,util
        elif all_observers[j][-6:] == "6_lags":
            from intertrial_no_bootstrap_6_lags import threshold,history,statistics,graphics,model,util
        elif all_observers[j][-6:] == "7_lags":
            from intertrial_no_bootstrap_7_lags import threshold,history,statistics,graphics,model,util

        data["data{0}".format(j)],w0["w0{0}".format(j)],plotinfo["plotinfo{0}".format(j)] = util.load_data_file (all_observers[j][:-7])
        print 'j', j
        print 'o', o
        hf = data["data{0}".format(j)].h
        backup_file = os.path.join ( "sim_backup",o+".pcl" )
        results_ =  cPickle.load ( open ( backup_file, 'r' ) )
        M = results_['model_w_hist']
        C = statistics.Kernel_and_Slope_Collector (
                hf, M.hf0, range(1,M.hf0) )
        print 'C(M)', C(M)
        kernels.append ( C(M) )
    kernels = pl.array ( kernels ).T
    cPickle.dump ( kernels, open ( 'sim_backup/all_kernels_wo_nohist_new.pcl', 'w' ) )


kr_neutral =[]
kr_repetitive =[]
kr_alternating =[]
kz_neutral =[]
kz_repetitive =[]
kz_alternating =[]
slope_neutral = []
slope_repetitive = []
slope_alternating = []
observer_neutral = []
observer_repetitive = []
observer_alternating = []
stim_kernels_rep = []
stim_kernels_alt = []
stim_kernels_neutr = []
stim_kernels_rep_nan = []
stim_kernels_alt_nan = []
stim_kernels_neutr_nan = []
correct_all_sj_lag_1_neutral = []
incorrect_all_sj_lag_1_neutral = []
correct_all_sj_lag_1_repetitive = []
incorrect_all_sj_lag_1_repetitive = []
correct_all_sj_lag_1_alternating = []
incorrect_all_sj_lag_1_alternating = []
for j,o in enumerate ( all_observers ):
    al = kernels[j][-2]
    kr = kernels[j][:int((len(kernels[j]) - 2)/2)]*al
    kz = kernels[j][int((len(kernels[j]) - 2)/2):int((len(kernels[j])) - 2)]*al

    if observer["observer{0}".format(j)].find('neutral') != -1:
        kr_neutral.append(kernels[j][0]*al)
        kz_neutral.append(kernels[j][int((len(kernels[j]) - 2)/2)]*al)
        stim_kernels_neutr.append(kernels[j][int((len(kernels[j]) - 2)/2):-2]*al)
        stim_kernels_neutr_nan.append(np.append(kernels[j][int((len(kernels[j]) - 2)/2):-2]*al, np.zeros(7-len(kernels[j][int((len(kernels[j]) - 2)/2):-2]*al))+ np.nan))
        correct_all_sj_lag_1_neutral.append ( kr[0]+kz[0])
        incorrect_all_sj_lag_1_neutral.append ( -kz[0]+kr[0])
        if '_' in observer["observer{0}".format(j)][0:3]:
            observer_neutral.append(observer["observer{0}".format(j)][1:2])
        else:
            observer_neutral.append(observer["observer{0}".format(j)][1:3])
        slope_neutral.append(al)
    elif observer["observer{0}".format(j)].find('repetitive') != -1:
        kr_repetitive.append(kernels[j][0]*al)
        kz_repetitive.append(kernels[j][int((len(kernels[j]) - 2)/2)]*al)
        stim_kernels_rep.append(kernels[j][int((len(kernels[j]) - 2)/2):-2]*al)
        stim_kernels_rep_nan.append(np.append(kernels[j][int((len(kernels[j]) - 2)/2):-2]*al, np.zeros(7-len(kernels[j][int((len(kernels[j]) - 2)/2):-2]*al))+ np.nan))
        correct_all_sj_lag_1_repetitive.append ( kr[0]+kz[0])
        incorrect_all_sj_lag_1_repetitive.append ( -kz[0]+kr[0])
        if '_' in observer["observer{0}".format(j)][0:3]:
            observer_repetitive.append(observer["observer{0}".format(j)][1:2])
        else:
            observer_repetitive.append(observer["observer{0}".format(j)][1:3])
        slope_repetitive.append(al)
    elif observer["observer{0}".format(j)].find('alternating') != -1:
        kr_alternating.append(kernels[j][0]*al)
        kz_alternating.append(kernels[j][int((len(kernels[j]) - 2)/2)]*al)
        stim_kernels_alt.append(kernels[j][int((len(kernels[j]) - 2)/2):-2]*al)
        stim_kernels_alt_nan.append(np.append(kernels[j][int((len(kernels[j]) - 2)/2):-2]*al, np.zeros(7-len(kernels[j][int((len(kernels[j]) - 2)/2):-2]*al))+ np.nan))
        correct_all_sj_lag_1_alternating.append ( kr[0]+kz[0])
        incorrect_all_sj_lag_1_alternating.append ( -kz[0]+kr[0])
        if '_' in observer["observer{0}".format(j)][0:3]:
            observer_alternating.append(observer["observer{0}".format(j)][1:2])
        else:
            observer_alternating.append(observer["observer{0}".format(j)][1:3])
        slope_alternating.append(al)


df1 = pd.DataFrame({'subject': observer_neutral,
                    'kr_neutral': kr_neutral,
                    'kz_neutral': kz_neutral,
                    'correct_neutral': correct_all_sj_lag_1_neutral,
                    'incorrect_neutral': incorrect_all_sj_lag_1_neutral
                })
df1 = df1[['subject', 'kr_neutral', 'kz_neutral', 'correct_neutral', 'incorrect_neutral']]
df2 = pd.DataFrame({'subject': observer_repetitive,
                    'kr_repetitive': kr_repetitive,
                    'kz_repetitive': kz_repetitive,
                    'correct_repetitive': correct_all_sj_lag_1_repetitive,
                    'incorrect_repetitive': incorrect_all_sj_lag_1_repetitive
                })
df2 = df2[['subject', 'kr_repetitive', 'kz_repetitive', 'correct_repetitive', 'incorrect_repetitive']]
df3 = pd.DataFrame({'subject': observer_alternating,
                    'kr_alternating': kr_alternating,
                    'kz_alternating': kz_alternating,
                    'correct_alternating': correct_all_sj_lag_1_alternating,
                    'incorrect_alternating': incorrect_all_sj_lag_1_alternating
                })
df3 = df3[['subject', 'kr_alternating', 'kz_alternating', 'correct_alternating', 'incorrect_alternating']]
filename = 'Weights_neutral_meg_behav.csv'
df1.to_csv(filename, sep = '\t', index = True, encoding = 'utf-8')
filename = 'Weights_repetitive_meg_behav.csv'
df2.to_csv(filename, sep = '\t', index = True, encoding = 'utf-8')
filename = 'Weights_alternating_meg_behav.csv'
df3.to_csv(filename, sep = '\t', index = True, encoding = 'utf-8')
