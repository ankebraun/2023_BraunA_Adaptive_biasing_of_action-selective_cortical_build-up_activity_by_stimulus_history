import os
import pandas as pd
import numpy as np
import sys
#sys.path.insert(0, "/Users/anke/cluster_home/pymeg")
sys.path.insert(0, "/mnt/homes/home024/abraun/pymeg")
sys.path.insert(0, "/Users/anke/cluster_home/anaconda/envs/py36/bin/mne")
import pymeg
from pymeg.contrast_tfr_baseline_per_session_subsample_test import Cache, compute_contrast, augment_data
from pymeg import contrast_tfr_baseline_per_session_subsample_test
from pymeg import parallel
from joblib import Memory
import logging
import glob
import pdb

from optparse import OptionParser
from os.path import join
usage = "srtfr_bias_bins.py [options] <sj_number>"

long_help = """
This script extracts the time frequency responses for three single-bias bins for one session.

This script was adapted from Niklas Wilming et al. Nature Communications 2020
https://github.com/DonnerLab/2020_Large-scale-Dynamics-of-Perceptual-Decision-Information-across-Human-Cortex
"""

parser = OptionParser ( usage, epilog=long_help )


opts,args = parser.parse_args ()

logging.getLogger().setLevel(logging.INFO)
logger = contrast_tfr_baseline_per_session_subsample_test.logging.getLogger()
logger.setLevel(logging.INFO)


meta_path = '/home/abraun/meg_data'
data_path = '/home/abraun/meg_data/sr_labeled/sf600/'
fig_folder = '/home/abraun/meg_data/sr_labeled/figures'

memory = Memory(cachedir=os.environ['PYMEG_CACHE_DIR'], verbose=0)
path = '/home/abraun/meg_data/sr_contrasts'


contrasts_lhipsi = {
    'bias_low_tertile_minus_neutral_hand': (['bias_low_tertile_left_is_up_minus_neutral', 'bias_low_tertile_right_is_up_minus_neutral'], (-0.5, 0.5)),
    'bias_medium_tertile_minus_neutral_hand': (['bias_medium_tertile_left_is_up_minus_neutral', 'bias_medium_tertile_right_is_up_minus_neutral'], (-0.5, 0.5)),
    'bias_high_tertile_minus_neutral_hand': (['bias_high_tertile_left_is_up_minus_neutral', 'bias_high_tertile_right_is_up_minus_neutral'], (-0.5, 0.5)),
    'bias_low_tertile_hand': (['bias_low_tertile_left_is_up', 'bias_low_tertile_right_is_up'], (-0.5, 0.5)),
    'bias_medium_tertile_hand': (['bias_medium_tertile_left_is_up', 'bias_medium_tertile_right_is_up'], (-0.5, 0.5)),
    'bias_high_tertile_hand': (['bias_high_tertile_left_is_up', 'bias_high_tertile_right_is_up'], (-0.5, 0.5)),
    'Repetitive_bias_low_tertile_minus_neutral_hand': (['Repetitive_bias_low_tertile_left_is_up_minus_neutral', 'Repetitive_bias_low_tertile_right_is_up_minus_neutral'], (-0.5, 0.5)),
    'Repetitive_bias_medium_tertile_minus_neutral_hand': (['Repetitive_bias_medium_tertile_left_is_up_minus_neutral', 'Repetitive_bias_medium_tertile_right_is_up_minus_neutral'], (-0.5, 0.5)),
    'Repetitive_bias_high_tertile_minus_neutral_hand': (['Repetitive_bias_high_tertile_left_is_up_minus_neutral', 'Repetitive_bias_high_tertile_right_is_up_minus_neutral'], (-0.5, 0.5)),
    'Repetitive_bias_low_tertile_hand': (['Repetitive_bias_low_tertile_left_is_up', 'Repetitive_bias_low_tertile_right_is_up'], (-0.5, 0.5)),
    'Repetitive_bias_medium_tertile_hand': (['Repetitive_bias_medium_tertile_left_is_up', 'Repetitive_bias_medium_tertile_right_is_up'], (-0.5, 0.5)),
    'Repetitive_bias_high_tertile_hand': (['Repetitive_bias_high_tertile_left_is_up', 'Repetitive_bias_high_tertile_right_is_up'], (-0.5, 0.5)),
    'Alternating_bias_low_tertile_minus_neutral_hand': (['Alternating_bias_low_tertile_left_is_up_minus_neutral', 'Alternating_bias_low_tertile_right_is_up_minus_neutral'], (-0.5, 0.5)),
    'Alternating_bias_medium_tertile_minus_neutral_hand': (['Alternating_bias_medium_tertile_left_is_up_minus_neutral', 'Alternating_bias_medium_tertile_right_is_up_minus_neutral'], (-0.5, 0.5)),
    'Alternating_bias_high_tertile_minus_neutral_hand': (['Alternating_bias_high_tertile_left_is_up_minus_neutral', 'Alternating_bias_high_tertile_right_is_up_minus_neutral'], (-0.5, 0.5)),
    'Alternating_bias_low_tertile_hand': (['Alternating_bias_low_tertile_left_is_up', 'Alternating_bias_low_tertile_right_is_up'], (-0.5, 0.5)),
    'Alternating_bias_medium_tertile_hand': (['Alternating_bias_medium_tertile_left_is_up', 'Alternating_bias_medium_tertile_right_is_up'], (-0.5, 0.5)),
    'Alternating_bias_high_tertile_hand': (['Alternating_bias_high_tertile_left_is_up', 'Alternating_bias_high_tertile_right_is_up'], (-0.5, 0.5)),
    'Neutral_bias_low_tertile_hand': (['Neutral_bias_low_tertile_left_is_up', 'Neutral_bias_low_tertile_right_is_up'], (-0.5, 0.5)),
    'Neutral_bias_medium_tertile_hand': (['Neutral_bias_medium_tertile_left_is_up', 'Neutral_bias_medium_tertile_right_is_up'], (-0.5, 0.5)),
    'Neutral_bias_high_tertile_hand': (['Neutral_bias_high_tertile_left_is_up', 'Neutral_bias_high_tertile_right_is_up'], (-0.5, 0.5)),
}


def srtfrfilename(contrasts, subject, session):
    try:
        makedirs(path)
    except:
        pass
    subject = int(subject)
    filename = f'S{subject}-{contrasts}_baseline_per_session_session_{session}_bin_on_bias_incl_general_bias_bins_subsample_10k.hdf' 
    return join(path, filename)


@memory.cache()
def get_contrasts(contrasts, subject, session, hemi, baseline_per_condition=False):

    stim, resp, meta_data_filenames, meta_data_filenames_before_artf_reject = [], [], [], []
    

    stim.append(join(data_path, f'S{subject}-SESS{session}-REC*-stimulus-*-lcmv.hdf' ))

    resp.append(join(data_path, f'S{subject}-SESS{session}-REC*-response-*-lcmv.hdf'))

    sessions = range(session, session + 1)
    
    for session in sessions: 
        if int(subject) == 5 and session == 1:
            recs = [2]
        else:
            recs = [1]

        for rec in recs:
            if int(subject) < 10:
               meta_data_filenames.append(join(meta_path, 'P0{}/MEG/Locked/P0{}-S{}_rec{}_stim_new.hdf'.format(subject, subject, session, rec)))
               meta_data_filenames_before_artf_reject.append(join(meta_path, 'P0{}/MEG/Preproc/P0{}-S{}_rec{}_data.hdf'.format(subject, subject, session, rec)))
            elif int(subject) >= 10:
               meta_data_filenames.append(join(meta_path, 'P{}/MEG/Locked/P{}-S{}_rec{}_stim_new.hdf'.format(subject, subject, session, rec)))
               meta_data_filenames_before_artf_reject.append(join(meta_path, 'P{}/MEG/Preproc/P{}-S{}_rec{}_data.hdf'.format(subject, subject, session, rec)))
         
###### for contrasts based on previous trials take data from all trials before artifact rejection
###### and match with trials after artifact rejection   

    meta_data = pd.concat(
        [pd.read_hdf(meta_data_filename, 'df')
            for meta_data_filename in meta_data_filenames])
    meta_data_before_artf_reject = pd.concat(
        [pd.read_hdf(meta_data_filename_before_artf_reject, 'df')
            for meta_data_filename_before_artf_reject in meta_data_filenames_before_artf_reject])
            

    idx_after_artf_reject =  meta_data['idx']
    meta_data_all_trls = meta_data_before_artf_reject[(meta_data_before_artf_reject['idx']).isin(idx_after_artf_reject)]

    meta_data["all"] = 1
    print(np.sum(meta_data["all"]))
    meta_data['respbutton'][meta_data['respbutton'] == 75] = 12 # fix mixed codings for respbuttons
    meta_data['respbutton'][meta_data['respbutton'] == 77] = 18 # fix mixed codings for respbuttons

    meta_data['respbutton'][meta_data['respbutton'] == 8] = 12 # fix mixed codings for respbuttons
    meta_data['respbutton'][meta_data['respbutton'] == 1] = 18 # fix mixed codings for respbuttons

    meta_data['respbutton_up_hand'] = 1
    for block in np.unique(meta_data.blockcnt):
        up_hand_this_block = np.unique(meta_data.query("blockcnt == {} & resptype == 1".format(block)).respbutton.values)   
        meta_data["respbutton_up_hand"][meta_data["blockcnt"] == block] = up_hand_this_block*np.ones((len(meta_data["blockcnt"] == block)))    
        
  
    meta_data_all_trls['respbutton'][meta_data_all_trls['respbutton'] == 75] = 12 # fix mixed codings for respbuttons
    meta_data_all_trls['respbutton'][meta_data_all_trls['respbutton'] == 77] = 18 # fix mixed codings for respbuttons

    meta_data_all_trls['prev_respbutton'][meta_data_all_trls['prev_respbutton'] == 75] = 12 # fix mixed codings for respbuttons
    meta_data_all_trls['prev_respbutton'][meta_data_all_trls['prev_respbutton'] == 77] = 18 # fix mixed codings for respbuttons

    meta_data_all_trls['respbutton'][meta_data_all_trls['respbutton'] == 8] = 12 # fix mixed codings for respbuttons
    meta_data_all_trls['respbutton'][meta_data_all_trls['respbutton'] == 1] = 18 # fix mixed codings for respbuttons

    meta_data_all_trls['prev_respbutton'][meta_data_all_trls['prev_respbutton'] == 8] = 12 # fix mixed codings for respbuttons
    meta_data_all_trls['prev_respbutton'][meta_data_all_trls['prev_respbutton'] == 1] = 18 # fix mixed codings for respbuttons


    meta_data["Repetitive"] = (meta_data["alternation_prob"] == 0.2).astype(float)
    meta_data["Neutral"] = (meta_data["alternation_prob"] == 0.5).astype(float)
    meta_data["Alternating"] = (meta_data["alternation_prob"] == 0.8).astype(float)
    
    rep_blocks = np.unique(meta_data[meta_data["alternation_prob"] == 0.2].blockcnt)
    alt_blocks = np.unique(meta_data[meta_data["alternation_prob"] == 0.8].blockcnt)    
    neutr_blocks = np.unique(meta_data[meta_data["alternation_prob"] == 0.5].blockcnt)     
    

    meta_data_rep = meta_data[(meta_data.alternation_prob == 0.2)]

    meta_data_alt = meta_data[(meta_data.alternation_prob == 0.8)]
    
    meta_data_neutr = meta_data[(meta_data.alternation_prob == 0.5)]

    
    def get_single_trial_bias(subject, condition, meta_data_condition):        
        ################# get trials conditioned on current response #############################################################
        idx_up_resp = meta_data_condition.query('resptype == "1"')['idx'] 
        idx_down_resp = meta_data_condition.query('resptype == "-1"')['idx']  

        sj_condition_up = meta_data_condition[(meta_data_condition['idx']).isin(idx_up_resp)]
        sj_condition_down = meta_data_condition[(meta_data_condition['idx']).isin(idx_down_resp)]
    
        ################# load single trial fründ biases #########################################################################
    
        single_trial_bias_condition = []
        def load_fold(subject, condition, fold):
            single_trial_bias_fold = pd.read_csv('/home/abraun/meg_anke_behavdata/crossvalidation/P' + str(subject) + '_' + condition.lower() + '_meg_behav_unique_blocks_test_fold_' + str(fold) + '.single_trial_bias_incl_general_bias.csv', sep = '\t')
            return single_trial_bias_fold
        for i in range(1,7):
            if condition == "all": 
                if os.path.isfile(os.path.join('/home/abraun/meg_anke_behavdata/crossvalidation/P' + str(subject) + '_' + "Repetitive".lower() + '_meg_behav_unique_blocks_test_fold_' + str(i) + '.single_trial_bias_incl_general_bias.csv')):
                    single_trial_bias_fold_rep = load_fold(subject, "Repetitive", i) 
                    single_trial_bias_fold_rep.loc[:, 'single_trial_bias'] = single_trial_bias_fold_rep['single_trial_bias_' + "Repetitive".lower()]
                    single_trial_bias_fold_rep.loc[:, 'single_trial_bias_prev_trial'] = np.roll(single_trial_bias_fold_rep['single_trial_bias_' + "Repetitive".lower()],1)
                    single_trial_bias_fold_rep.loc[0, 'single_trial_bias_prev_trial'] = 0
                    single_trial_bias_fold_rep.loc[:, 'single_trial_bias_next_trial'] = np.roll(single_trial_bias_fold_rep['single_trial_bias_' + "Repetitive".lower()],-1)
                    single_trial_bias_condition.append(single_trial_bias_fold_rep)  
                if os.path.isfile(os.path.join('/home/abraun/meg_anke_behavdata/crossvalidation/P' + str(subject) + '_' + "Alternating".lower() + '_meg_behav_unique_blocks_test_fold_' + str(i) + '.single_trial_bias_incl_general_bias.csv')):
                    single_trial_bias_fold_alt = load_fold(subject, "Alternating", i) 
                    single_trial_bias_fold_alt.loc[:, 'single_trial_bias'] = single_trial_bias_fold_alt['single_trial_bias_' + "Alternating".lower()]
                    single_trial_bias_fold_alt.loc[:, 'single_trial_bias_prev_trial'] = np.roll(single_trial_bias_fold_alt['single_trial_bias_' + "Alternating".lower()],1)
                    single_trial_bias_fold_alt.loc[0, 'single_trial_bias_prev_trial'] = 0
                    single_trial_bias_fold_alt.loc[:, 'single_trial_bias_next_trial'] = np.roll(single_trial_bias_fold_alt['single_trial_bias_' + "Alternating".lower()],-1)
                    single_trial_bias_condition.append(single_trial_bias_fold_alt)  
                if os.path.isfile(os.path.join('/home/abraun/meg_anke_behavdata/crossvalidation/P' + str(subject) + '_' + "Neutral".lower() + '_meg_behav_unique_blocks_test_fold_' + str(i) + '.single_trial_bias_incl_general_bias.csv')):
                    single_trial_bias_fold_neutr = load_fold(subject, "Neutral", i) 
                    single_trial_bias_fold_neutr.loc[:, 'single_trial_bias'] = single_trial_bias_fold_neutr['single_trial_bias_' + "Neutral".lower()]
                    single_trial_bias_fold_neutr.loc[:, 'single_trial_bias_prev_trial'] = np.roll(single_trial_bias_fold_neutr['single_trial_bias_' + "Neutral".lower()],1)
                    single_trial_bias_fold_neutr.loc[0, 'single_trial_bias_prev_trial'] = 0
                    single_trial_bias_fold_neutr.loc[:, 'single_trial_bias_next_trial'] = np.roll(single_trial_bias_fold_neutr['single_trial_bias_' + "Neutral".lower()],-1)
                    single_trial_bias_condition.append(single_trial_bias_fold_neutr)  
                    
            else:                    
                if os.path.isfile(os.path.join('/home/abraun/meg_anke_behavdata/crossvalidation/P' + str(subject) + '_' + condition.lower() + '_meg_behav_unique_blocks_test_fold_' + str(i) + '.single_trial_bias_incl_general_bias.csv')):
                    single_trial_bias_fold = load_fold(subject, condition, i) 
                    single_trial_bias_fold.loc[:, 'single_trial_bias'] = single_trial_bias_fold['single_trial_bias_' + condition.lower()]
                    single_trial_bias_fold.loc[:, 'single_trial_bias_prev_trial'] = np.roll(single_trial_bias_fold['single_trial_bias_' + condition.lower()],1)
                    single_trial_bias_fold.loc[0, 'single_trial_bias_prev_trial'] = 0
                    single_trial_bias_fold.loc[:, 'single_trial_bias_next_trial'] = np.roll(single_trial_bias_fold['single_trial_bias_' + condition.lower()],-1)
                    single_trial_bias_condition.append(single_trial_bias_fold)  
        

        single_trial_bias_condition = pd.concat(single_trial_bias_condition).sort_values('idx')  
        idx = single_trial_bias_condition['idx'] 
        meta_data_idx = meta_data_condition[(meta_data_condition['idx']).isin(idx)]
        trials =  meta_data_idx['idx']
        single_trial_bias_condition = single_trial_bias_condition[(single_trial_bias_condition['idx']).isin(trials)]
        single_trial_bias_condition = single_trial_bias_condition.sort_values('idx')
        max_bias = max(single_trial_bias_condition['single_trial_bias'])
        min_bias = min(single_trial_bias_condition['single_trial_bias'])

        binsize = (max_bias - min_bias)/3

        single_trial_bias_positive = single_trial_bias_condition.query('single_trial_bias > 0')
        single_trial_bias_negative = single_trial_bias_condition.query('single_trial_bias < 0')
        
        single_trial_bias_low_tertile = single_trial_bias_condition.query('single_trial_bias <= {}'.format(min_bias + binsize))
        single_trial_bias_medium_tertile = single_trial_bias_condition.query('{} < single_trial_bias < {}'.format(min_bias + binsize, min_bias + 2*binsize))
        single_trial_bias_high_tertile = single_trial_bias_condition.query('single_trial_bias >= {}'.format(min_bias + 2*binsize))

        single_trial_bias_lower_half = single_trial_bias_condition.query('single_trial_bias < {}'.format(min_bias + (max_bias - min_bias)/2))
        single_trial_bias_upper_half = single_trial_bias_condition.query('single_trial_bias >= {}'.format(min_bias + (max_bias - min_bias)/2))
        
        sj_condition_up_positive = sj_condition_up[(sj_condition_up['idx']).isin(single_trial_bias_positive.idx)]
        sj_condition_up_negative = sj_condition_up[(sj_condition_up['idx']).isin(single_trial_bias_negative.idx)]

        sj_condition_down_positive = sj_condition_down[(sj_condition_down['idx']).isin(single_trial_bias_positive.idx)]
        sj_condition_down_negative = sj_condition_down[(sj_condition_down['idx']).isin(single_trial_bias_negative.idx)]


        ######## collapse across choices ##########################

        ####### bias congurent with choice #####################
        sj_condition_eq = []
        sj_condition_eq.append(sj_condition_up_positive)
        sj_condition_eq.append(sj_condition_down_negative)  
        sj_condition_eq = pd.concat(sj_condition_eq) 

        ####### bias incongurent with choice #####################
        sj_condition_opp = []
        sj_condition_opp.append(sj_condition_up_negative)
        sj_condition_opp.append(sj_condition_down_positive)  
        sj_condition_opp = pd.concat(sj_condition_opp)
        return sj_condition_eq, sj_condition_opp, single_trial_bias_positive, single_trial_bias_negative, single_trial_bias_low_tertile, single_trial_bias_medium_tertile, single_trial_bias_high_tertile, single_trial_bias_lower_half, single_trial_bias_upper_half


    def get_single_trial_bias_minus_neutral(subject, condition, meta_data_condition):        
        ################# get trials conditioned on current response #############################################################
        idx_up_resp = meta_data_condition.query('resptype == "1"')['idx'] 
        idx_down_resp = meta_data_condition.query('resptype == "-1"')['idx']  

        sj_condition_up = meta_data_condition[(meta_data_condition['idx']).isin(idx_up_resp)]
        sj_condition_down = meta_data_condition[(meta_data_condition['idx']).isin(idx_down_resp)]
    
        ################# load single trial fründ biases #########################################################################
    
        single_trial_bias_condition = []
        def load_fold(subject, condition, fold):
            single_trial_bias_fold = pd.read_csv('/home/abraun/meg_anke_behavdata/crossvalidation/P' + str(subject) + '_' + condition.lower() + '_meg_behav_unique_blocks_test_fold_' + str(fold) + '.single_trial_bias_incl_general_bias_minus_neutral.csv', sep = '\t')
            return single_trial_bias_fold
        for i in range(1,7):
            if condition == "all": 
                if os.path.isfile(os.path.join('/home/abraun/meg_anke_behavdata/crossvalidation/P' + str(subject) + '_' + "Repetitive".lower() + '_meg_behav_unique_blocks_test_fold_' + str(i) + '.single_trial_bias_incl_general_bias_minus_neutral.csv')):
                    single_trial_bias_fold_rep = load_fold(subject, "Repetitive", i) 
                    single_trial_bias_fold_rep.loc[:, 'single_trial_bias'] = single_trial_bias_fold_rep['single_trial_bias_' + "Repetitive".lower()]
                    single_trial_bias_fold_rep.loc[:, 'single_trial_bias_prev_trial'] = np.roll(single_trial_bias_fold_rep['single_trial_bias_' + "Repetitive".lower()],1)
                    single_trial_bias_fold_rep.loc[0, 'single_trial_bias_prev_trial'] = 0
                    single_trial_bias_fold_rep.loc[:, 'single_trial_bias_next_trial'] = np.roll(single_trial_bias_fold_rep['single_trial_bias_' + "Repetitive".lower()],-1)
                    single_trial_bias_condition.append(single_trial_bias_fold_rep)  
                if os.path.isfile(os.path.join('/home/abraun/meg_anke_behavdata/crossvalidation/P' + str(subject) + '_' + "Alternating".lower() + '_meg_behav_unique_blocks_test_fold_' + str(i) + '.single_trial_bias_incl_general_bias_minus_neutral.csv')):
                    single_trial_bias_fold_alt = load_fold(subject, "Alternating", i) 
                    single_trial_bias_fold_alt.loc[:, 'single_trial_bias'] = single_trial_bias_fold_alt['single_trial_bias_' + "Alternating".lower()]
                    single_trial_bias_fold_alt.loc[:, 'single_trial_bias_prev_trial'] = np.roll(single_trial_bias_fold_alt['single_trial_bias_' + "Alternating".lower()],1)
                    single_trial_bias_fold_alt.loc[0, 'single_trial_bias_prev_trial'] = 0
                    single_trial_bias_fold_alt.loc[:, 'single_trial_bias_next_trial'] = np.roll(single_trial_bias_fold_alt['single_trial_bias_' + "Alternating".lower()],-1)
                    single_trial_bias_condition.append(single_trial_bias_fold_alt)                      
            else:                                
                if os.path.isfile(os.path.join('/home/abraun/meg_anke_behavdata/crossvalidation/P' + str(subject) + '_' + condition.lower() + '_meg_behav_unique_blocks_test_fold_' + str(i) + '.single_trial_bias_incl_general_bias_minus_neutral.csv')):
                    single_trial_bias_fold = load_fold(subject, condition, i) 
                    single_trial_bias_fold.loc[:, 'single_trial_bias'] = single_trial_bias_fold['single_trial_bias_' + condition.lower()]
                    single_trial_bias_fold.loc[:, 'single_trial_bias_prev_trial'] = np.roll(single_trial_bias_fold['single_trial_bias_' + condition.lower()],1)
                    single_trial_bias_fold.loc[0, 'single_trial_bias_prev_trial'] = 0
                    single_trial_bias_fold.loc[:, 'single_trial_bias_next_trial'] = np.roll(single_trial_bias_fold['single_trial_bias_' + condition.lower()],-1)
                    single_trial_bias_condition.append(single_trial_bias_fold)  
        

        single_trial_bias_condition = pd.concat(single_trial_bias_condition).sort_values('idx')  
        idx = single_trial_bias_condition['idx'] 
        meta_data_idx = meta_data_condition[(meta_data_condition['idx']).isin(idx)]
        trials =  meta_data_idx['idx']
        single_trial_bias_condition = single_trial_bias_condition[(single_trial_bias_condition['idx']).isin(trials)]
        single_trial_bias_condition = single_trial_bias_condition.sort_values('idx')
        max_bias = max(single_trial_bias_condition['single_trial_bias'])
        min_bias = min(single_trial_bias_condition['single_trial_bias'])
        print('single_trial_bias_condition', single_trial_bias_condition)
        print('max_bias', max_bias)
        print('min_bias', min_bias)
        binsize = (max_bias - min_bias)/3
        
        single_trial_bias_positive = single_trial_bias_condition.query('single_trial_bias > 0')
        single_trial_bias_negative = single_trial_bias_condition.query('single_trial_bias < 0')

        print('single_trial_bias_positive', single_trial_bias_positive)
        print('single_trial_bias_negative', single_trial_bias_negative)

        single_trial_bias_low_tertile = single_trial_bias_condition.query('single_trial_bias <= {}'.format(min_bias + binsize))
        single_trial_bias_medium_tertile = single_trial_bias_condition.query('{} < single_trial_bias < {}'.format(min_bias + binsize, min_bias + 2*binsize))
        single_trial_bias_high_tertile = single_trial_bias_condition.query('single_trial_bias >= {}'.format(min_bias + 2*binsize))

        single_trial_bias_lower_half = single_trial_bias_condition.query('single_trial_bias < {}'.format(min_bias + (max_bias - min_bias)/2))
        single_trial_bias_upper_half = single_trial_bias_condition.query('single_trial_bias >= {}'.format(min_bias + (max_bias - min_bias)/2))


        sj_condition_up_positive = sj_condition_up[(sj_condition_up['idx']).isin(single_trial_bias_positive.idx)]
        sj_condition_up_negative = sj_condition_up[(sj_condition_up['idx']).isin(single_trial_bias_negative.idx)]

        sj_condition_down_positive = sj_condition_down[(sj_condition_down['idx']).isin(single_trial_bias_positive.idx)]
        sj_condition_down_negative = sj_condition_down[(sj_condition_down['idx']).isin(single_trial_bias_negative.idx)]

        ######## collapse across choices ##########################

        ####### bias congurent with choice #####################
        sj_condition_eq = []
        sj_condition_eq.append(sj_condition_up_positive)
        sj_condition_eq.append(sj_condition_down_negative)  
        sj_condition_eq = pd.concat(sj_condition_eq) 

        ####### bias incongurent with choice #####################
        sj_condition_opp = []
        sj_condition_opp.append(sj_condition_up_negative)
        sj_condition_opp.append(sj_condition_down_positive)  
        sj_condition_opp = pd.concat(sj_condition_opp)
        return sj_condition_eq, sj_condition_opp, single_trial_bias_positive, single_trial_bias_negative, single_trial_bias_low_tertile, single_trial_bias_medium_tertile, single_trial_bias_high_tertile, single_trial_bias_lower_half, single_trial_bias_upper_half


    eq, opp, positive, negative, low_tertile,\
     medium_tertile, high_tertile, lower_half, upper_half = get_single_trial_bias(subject, "all", meta_data)

    eq_minus_neutral, opp_minus_neutral, positive_minus_neutral, negative_minus_neutral, low_tertile_minus_neutral,\
     medium_tertile_minus_neutral, high_tertile_minus_neutral, lower_half_minus_neutral, upper_half_minus_neutral = get_single_trial_bias_minus_neutral(subject, "all", meta_data)

    meta_data["bias_low_tertile_left_is_up"] = (meta_data.idx.isin(low_tertile.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["bias_low_tertile_right_is_up"] = (meta_data.idx.isin(low_tertile.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 
    meta_data["bias_medium_tertile_left_is_up"] = (meta_data.idx.isin(medium_tertile.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["bias_medium_tertile_right_is_up"] = (meta_data.idx.isin(medium_tertile.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 
    meta_data["bias_high_tertile_left_is_up"] = (meta_data.idx.isin(high_tertile.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["bias_high_tertile_right_is_up"] = (meta_data.idx.isin(high_tertile.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 

    meta_data["bias_low_tertile_left_is_up_minus_neutral"] = (meta_data.idx.isin(low_tertile_minus_neutral.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["bias_low_tertile_right_is_up_minus_neutral"] = (meta_data.idx.isin(low_tertile_minus_neutral.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 
    meta_data["bias_medium_tertile_left_is_up_minus_neutral"] = (meta_data.idx.isin(medium_tertile_minus_neutral.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["bias_medium_tertile_right_is_up_minus_neutral"] = (meta_data.idx.isin(medium_tertile_minus_neutral.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 
    meta_data["bias_high_tertile_left_is_up_minus_neutral"] = (meta_data.idx.isin(high_tertile_minus_neutral.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["bias_high_tertile_right_is_up_minus_neutral"] = (meta_data.idx.isin(high_tertile_minus_neutral.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 

    rep_eq_minus_neutral, rep_opp_minus_neutral, rep_positive_minus_neutral, rep_negative_minus_neutral, rep_low_tertile_minus_neutral,\
     rep_medium_tertile_minus_neutral, rep_high_tertile_minus_neutral, rep_lower_half_minus_neutral, rep_upper_half_minus_neutral = get_single_trial_bias_minus_neutral(subject, "Repetitive", meta_data_rep)

    meta_data["Repetitive_bias_low_tertile_left_is_up_minus_neutral"] = (meta_data.idx.isin(rep_low_tertile_minus_neutral.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["Repetitive_bias_low_tertile_right_is_up_minus_neutral"] = (meta_data.idx.isin(rep_low_tertile_minus_neutral.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 
    meta_data["Repetitive_bias_medium_tertile_left_is_up_minus_neutral"] = (meta_data.idx.isin(rep_medium_tertile_minus_neutral.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["Repetitive_bias_medium_tertile_right_is_up_minus_neutral"] = (meta_data.idx.isin(rep_medium_tertile_minus_neutral.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 
    meta_data["Repetitive_bias_high_tertile_left_is_up_minus_neutral"] = (meta_data.idx.isin(rep_high_tertile_minus_neutral.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["Repetitive_bias_high_tertile_right_is_up_minus_neutral"] = (meta_data.idx.isin(rep_high_tertile_minus_neutral.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 
    
    alt_eq_minus_neutral, alt_opp_minus_neutral, alt_positive_minus_neutral, alt_negative_minus_neutral, alt_low_tertile_minus_neutral,\
     alt_medium_tertile_minus_neutral, alt_high_tertile_minus_neutral, alt_lower_half_minus_neutral, alt_upper_half_minus_neutral = get_single_trial_bias_minus_neutral(subject, "Alternating", meta_data_alt)
     
    meta_data["Alternating_bias_low_tertile_left_is_up_minus_neutral"] = (meta_data.idx.isin(alt_low_tertile_minus_neutral.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["Alternating_bias_low_tertile_right_is_up_minus_neutral"] = (meta_data.idx.isin(alt_low_tertile_minus_neutral.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 
    meta_data["Alternating_bias_medium_tertile_left_is_up_minus_neutral"] = (meta_data.idx.isin(alt_medium_tertile_minus_neutral.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["Alternating_bias_medium_tertile_right_is_up_minus_neutral"] = (meta_data.idx.isin(alt_medium_tertile_minus_neutral.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 
    meta_data["Alternating_bias_high_tertile_left_is_up_minus_neutral"] = (meta_data.idx.isin(alt_high_tertile_minus_neutral.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["Alternating_bias_high_tertile_right_is_up_minus_neutral"] = (meta_data.idx.isin(alt_high_tertile_minus_neutral.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 

#############################################        
    rep_eq, rep_opp, rep_positive, rep_negative, rep_low_tertile,\
     rep_medium_tertile, rep_high_tertile, rep_lower_half, rep_upper_half = get_single_trial_bias(subject, "Repetitive", meta_data_rep)

    meta_data["Repetitive_bias_low_tertile_left_is_up"] = (meta_data.idx.isin(rep_low_tertile.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["Repetitive_bias_low_tertile_right_is_up"] = (meta_data.idx.isin(rep_low_tertile.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 
    meta_data["Repetitive_bias_medium_tertile_left_is_up"] = (meta_data.idx.isin(rep_medium_tertile.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["Repetitive_bias_medium_tertile_right_is_up"] = (meta_data.idx.isin(rep_medium_tertile.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 
    meta_data["Repetitive_bias_high_tertile_left_is_up"] = (meta_data.idx.isin(rep_high_tertile.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["Repetitive_bias_high_tertile_right_is_up"] = (meta_data.idx.isin(rep_high_tertile.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 

    alt_eq, alt_opp, alt_positive, alt_negative, alt_low_tertile,\
     alt_medium_tertile, alt_high_tertile, alt_lower_half, alt_upper_half = get_single_trial_bias(subject, "Alternating", meta_data_alt)

    meta_data["Alternating_bias_low_tertile_left_is_up"] = (meta_data.idx.isin(alt_low_tertile.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["Alternating_bias_low_tertile_right_is_up"] = (meta_data.idx.isin(alt_low_tertile.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 
    meta_data["Alternating_bias_medium_tertile_left_is_up"] = (meta_data.idx.isin(alt_medium_tertile.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["Alternating_bias_medium_tertile_right_is_up"] = (meta_data.idx.isin(alt_medium_tertile.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 
    meta_data["Alternating_bias_high_tertile_left_is_up"] = (meta_data.idx.isin(alt_high_tertile.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["Alternating_bias_high_tertile_right_is_up"] = (meta_data.idx.isin(alt_high_tertile.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 
    neutr_eq, neutr_opp, neutr_positive, neutr_negative, neutr_low_tertile,\
     neutr_medium_tertile, neutr_high_tertile, neutr_lower_half, neutr_upper_half = get_single_trial_bias(subject, "Neutral", meta_data_neutr)
    meta_data["Neutral_bias_low_tertile_left_is_up"] = (meta_data.idx.isin(neutr_low_tertile.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["Neutral_bias_low_tertile_right_is_up"] = (meta_data.idx.isin(neutr_low_tertile.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 
    meta_data["Neutral_bias_medium_tertile_left_is_up"] = (meta_data.idx.isin(neutr_medium_tertile.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["Neutral_bias_medium_tertile_right_is_up"] = (meta_data.idx.isin(neutr_medium_tertile.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 
    meta_data["Neutral_bias_high_tertile_left_is_up"] = (meta_data.idx.isin(neutr_high_tertile.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["Neutral_bias_high_tertile_right_is_up"] = (meta_data.idx.isin(neutr_high_tertile.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 

        
    meta_data["left"] = (meta_data["respbutton"] == 12).astype(bool)
    meta_data["right"] = (meta_data["respbutton"] == 18).astype(bool)


    for col in ['Repetitive', 'Neutral', 'Alternating']:
        meta_data.loc[meta_data["left"], col + '_left'] = meta_data.loc[meta_data["left"], col]
        meta_data.loc[meta_data["right"], col + '_right'] = meta_data.loc[meta_data["right"], col]
        print(np.sum(meta_data.loc[meta_data["left"], col + '_left']))      

    meta_data["hand_map"] = (((meta_data["respbutton"] == 12) & (meta_data["resptype"] == 1)) | ((meta_data["respbutton"] == 18) & (meta_data["resptype"] == -1))).astype(bool)
    
    hemis = [hemi, 'avg']
    meta_data.loc[:, 'hash'] = meta_data.loc[:, 'idx']

    cps = []
    new_contrasts = dict()
    print(contrasts)
    contrasts_pre_check = contrasts
    contrasts = dict()
    for key, value in contrasts_pre_check.items():
        print(key)
        print(value)
        print(np.sum(meta_data[value[0]]))
        if np.sum(meta_data[value[0]])[0] == 0:
            value = ([value[0][1]], [value[1][1]*2]) 
        elif np.sum(meta_data[value[0]])[1] == 0:
            value = ([value[0][0]], [value[1][0]*2])     
        contrasts[key] = value                     

    contrast_file = srtfrfilename(hemi, subject, session)
    if os.path.isfile(contrast_file): #load already existing contrasts
        contrast_data = pd.read_hdf(contrast_file)
        print(contrast_data.index.get_level_values('contrast'))
        for key, value in contrasts.items(): # check which contrasts already exist in that file
            print('key', key)
            print('value', value)
            if key not in contrast_data.index.get_level_values('contrast'):
                new_contrasts[key] = value
        if bool(new_contrasts):
            with Cache() as cache: # compute non existing contrasts
                contrast = compute_contrast(
                    new_contrasts, hemis, stim, stim,
                    meta_data, (-0.35, -0.1),
                    baseline_per_condition=False,
                    n_jobs=1, cache=cache)
                contrast.loc[:, 'epoch'] = 'stimulus'
                cps.append(contrast)
                contrast = compute_contrast(
                    new_contrasts, hemis, resp, stim,
                    meta_data, (-0.35, -0.1),
                    baseline_per_condition=False,
                    n_jobs=1, cache=cache)
                contrast.loc[:, 'epoch'] = 'response'
                cps.append(contrast)
            contrast = pd.concat(cps)
            del cps
            contrast.loc[:, 'subject'] = subject
            #contrast.loc[:, 'session'] = session
            contrast.set_index(['subject',  'contrast',
                                'hemi', 'epoch', 'cluster'], append=True, inplace=True)
                                
            contrast = pd.concat([contrast, contrast_data]) # concatenate new and old contrasts                    
            filename = srtfrfilename(hemi, subject, session)
            #filename = srtfrfilename_single_contrast(hemi, subject, contrasts)
            contrast.to_hdf(filename, 'epochs')
            return contrast
    else: # if no contrasts exist already
        with Cache() as cache:
            contrast = compute_contrast(
                contrasts, hemis, stim, stim,
                meta_data, (-0.35, -0.1),
                baseline_per_condition=False,
                n_jobs=1, cache=cache)
            contrast.loc[:, 'epoch'] = 'stimulus'
            cps.append(contrast)
            contrast = compute_contrast(
                contrasts, hemis, resp, stim,
                meta_data, (-0.35, -0.1),
                baseline_per_condition= False,
                n_jobs=1, cache=cache)
            contrast.loc[:, 'epoch'] = 'response'
            cps.append(contrast)
        contrast = pd.concat(cps)

        del cps
        contrast.loc[:, 'subject'] = subject
        #contrast.loc[:, 'session'] = session
        contrast.set_index(['subject',  'contrast',
                            'hemi', 'epoch', 'cluster'], append=True, inplace=True)
        filename = srtfrfilename(hemi, subject, session)
        #filename = srtfrfilename_single_contrast(hemi, subject, contrasts)
        contrast.to_hdf(filename, 'epochs')
        return contrast
        

subject = str(args[0])
session = 1
get_contrasts(contrasts_lhipsi, subject, session, 'lh_is_ipsi', baseline_per_condition=False)

