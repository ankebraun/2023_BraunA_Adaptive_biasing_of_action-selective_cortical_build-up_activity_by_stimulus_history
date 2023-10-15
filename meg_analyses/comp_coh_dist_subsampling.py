import os
import pandas as pd
import numpy as np
from joblib import Memory
import pdb
import random
from scipy import stats
from os.path import join
meta_path = '/home/abraun/meg_data'
fig_folder = '/home/abraun/meg_data/sr_labeled/figures'
memory = Memory(cachedir=os.environ['PYMEG_CACHE_DIR'], verbose=0)
path = '/home/abraun/meg_data/sr_contrasts'

contrasts_lhipsi = {
    'bias_low_tertile_hand': (['bias_low_tertile_left_is_up', 'bias_low_tertile_right_is_up'], (-0.5, 0.5)),
    'bias_high_tertile_hand': (['bias_high_tertile_left_is_up', 'bias_high_tertile_right_is_up'], (-0.5, 0.5)),
}


@memory.cache()
def get_contrasts(contrasts, subject, session, cond):
    stim, resp, meta_data_filenames, meta_data_filenames_before_artf_reject = [], [], [], []
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

    meta_data["bias_low_tertile_left_is_up"] = (meta_data.idx.isin(low_tertile.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["bias_low_tertile_right_is_up"] = (meta_data.idx.isin(low_tertile.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 

    meta_data["bias_high_tertile_left_is_up"] = (meta_data.idx.isin(high_tertile.idx)&(meta_data.respbutton_up_hand == 12)).astype(bool) 
    meta_data["bias_high_tertile_right_is_up"] = (meta_data.idx.isin(high_tertile.idx)&(meta_data.respbutton_up_hand == 18)).astype(bool) 
    meta_data[meta_data["bias_high_tertile_right_is_up"]].query('resptype == -1')
    condition_ind = meta_data.loc[meta_data[cond]== 1,  "idx"]

    condition_data_up_resp = meta_data[meta_data.idx.isin(condition_ind)].query('resptype == 1')
    condition_data_down_resp = meta_data[meta_data.idx.isin(condition_ind)].query('resptype == -1')    

    num_trials_up_resp = np.shape(condition_data_up_resp )[0]          
    num_trials_down_resp = np.shape( condition_data_down_resp )[0]   

    sample_idx_up_resp_all_samples = []
    sample_idx_down_resp_all_samples = []
    motion_coherence_up_resp_all_samples = []
    motion_coherence_down_resp_all_samples = []
    for samples in range(0,1000):
        sample_idx = [] 
        if num_trials_up_resp > num_trials_down_resp:
            sample_idx_up_resp = random.sample(condition_data_up_resp.idx.values.tolist(), num_trials_down_resp)
            sample_idx_down_resp = condition_data_down_resp.idx.values.tolist()
            # sample_idx_up_resp_all_samples.extend(sample_idx_up_resp)
            # sample_idx_down_resp_all_samples.extend(sample_idx_down_resp)
            motion_coherence_up_resp_all_samples.extend(meta_data.query('idx ==  @sample_idx_up_resp').motion_coherence)
            motion_coherence_down_resp_all_samples.extend(meta_data.query('idx ==  @sample_idx_down_resp').motion_coherence)
        elif num_trials_down_resp > num_trials_up_resp:
            sample_idx_down_resp = random.sample(condition_data_down_resp.idx.values.tolist(), num_trials_up_resp)
            sample_idx_up_resp = condition_data_up_resp.idx.values.tolist()
            # sample_idx_up_resp_all_samples.extend(sample_idx_up_resp)
            # sample_idx_down_resp_all_samples.extend(sample_idx_down_resp)
            motion_coherence_up_resp_all_samples.extend(meta_data.query('idx ==  @sample_idx_up_resp').motion_coherence)
            motion_coherence_down_resp_all_samples.extend(meta_data.query('idx ==  @sample_idx_down_resp').motion_coherence)
        elif num_trials_down_resp == num_trials_up_resp:    
            sample_idx_down_resp = condition_data_down_resp.idx.values.tolist()
            sample_idx_up_resp = condition_data_up_resp.idx.values.tolist()
            # sample_idx_up_resp_all_samples.extend(sample_idx_up_resp)
            # sample_idx_down_resp_all_samples.extend(sample_idx_down_resp)
            motion_coherence_up_resp_all_samples.extend(meta_data.query('idx ==  @sample_idx_up_resp').motion_coherence)
            motion_coherence_down_resp_all_samples.extend(meta_data.query('idx ==  @sample_idx_down_resp').motion_coherence)
                
    prop_zero_coh_down_resp = len(np.where(np.array(motion_coherence_down_resp_all_samples) == 0.0)[0])/len((np.array(motion_coherence_down_resp_all_samples)))
    prop_zero_three_coh_down_resp = len(np.where(np.array(motion_coherence_down_resp_all_samples) == 0.03)[0])/len((np.array(motion_coherence_down_resp_all_samples)))
    prop_zero_nine_coh_down_resp = len(np.where(np.array(motion_coherence_down_resp_all_samples) == 0.09)[0])/len((np.array(motion_coherence_down_resp_all_samples)))
    prop_two_seven_coh_down_resp = len(np.where(np.array(motion_coherence_down_resp_all_samples) == 0.27)[0])/len((np.array(motion_coherence_down_resp_all_samples)))
    prop_eight_one_coh_down_resp = len(np.where(np.array(motion_coherence_down_resp_all_samples) == 0.81)[0])/len((np.array(motion_coherence_down_resp_all_samples)))

    prop_zero_coh_up_resp = len(np.where(np.array(motion_coherence_up_resp_all_samples) == 0.0)[0])/len((np.array(motion_coherence_up_resp_all_samples)))
    prop_zero_three_coh_up_resp = len(np.where(np.array(motion_coherence_up_resp_all_samples) == 0.03)[0])/len((np.array(motion_coherence_up_resp_all_samples)))
    prop_zero_nine_coh_up_resp = len(np.where(np.array(motion_coherence_up_resp_all_samples) == 0.09)[0])/len((np.array(motion_coherence_up_resp_all_samples)))
    prop_two_seven_coh_up_resp = len(np.where(np.array(motion_coherence_up_resp_all_samples) == 0.27)[0])/len((np.array(motion_coherence_up_resp_all_samples)))
    prop_eight_one_coh_up_resp = len(np.where(np.array(motion_coherence_up_resp_all_samples) == 0.81)[0])/len((np.array(motion_coherence_up_resp_all_samples)))

    prop_zero_coh_down_resp_orig = len(np.where(np.array(condition_data_down_resp.motion_coherence) == 0.0)[0])/len((np.array(condition_data_down_resp)))
    prop_zero_three_coh_down_resp_orig = len(np.where(np.array(condition_data_down_resp.motion_coherence) == 0.03)[0])/len((np.array(condition_data_down_resp.motion_coherence)))
    prop_zero_nine_coh_down_resp_orig = len(np.where(np.array(condition_data_down_resp.motion_coherence) == 0.09)[0])/len((np.array(condition_data_down_resp.motion_coherence)))
    prop_two_seven_coh_down_resp_orig = len(np.where(np.array(condition_data_down_resp.motion_coherence) == 0.27)[0])/len((np.array(condition_data_down_resp.motion_coherence)))
    prop_eight_one_coh_down_resp_orig = len(np.where(np.array(condition_data_down_resp.motion_coherence) == 0.81)[0])/len((np.array(condition_data_down_resp.motion_coherence)))

    prop_zero_coh_up_resp_orig = len(np.where(np.array(condition_data_up_resp.motion_coherence) == 0.0)[0])/len((np.array(condition_data_up_resp)))
    prop_zero_three_coh_up_resp_orig = len(np.where(np.array(condition_data_up_resp.motion_coherence) == 0.03)[0])/len((np.array(condition_data_up_resp.motion_coherence)))
    prop_zero_nine_coh_up_resp_orig = len(np.where(np.array(condition_data_up_resp.motion_coherence) == 0.09)[0])/len((np.array(condition_data_up_resp.motion_coherence)))
    prop_two_seven_coh_up_resp_orig = len(np.where(np.array(condition_data_up_resp.motion_coherence) == 0.27)[0])/len((np.array(condition_data_up_resp.motion_coherence)))
    prop_eight_one_coh_up_resp_orig = len(np.where(np.array(condition_data_up_resp.motion_coherence) == 0.81)[0])/len((np.array(condition_data_up_resp.motion_coherence)))
    
    prop_cohs_subsampled_up_resp = []
    prop_cohs_subsampled_up_resp.append(prop_zero_coh_up_resp)
    prop_cohs_subsampled_up_resp.append(prop_zero_three_coh_up_resp)
    prop_cohs_subsampled_up_resp.append(prop_zero_nine_coh_up_resp)
    prop_cohs_subsampled_up_resp.append(prop_two_seven_coh_up_resp)
    prop_cohs_subsampled_up_resp.append(prop_eight_one_coh_up_resp)
    
    prop_cohs_subsampled_down_resp = []
    prop_cohs_subsampled_down_resp.append(prop_zero_coh_down_resp)
    prop_cohs_subsampled_down_resp.append(prop_zero_three_coh_down_resp)
    prop_cohs_subsampled_down_resp.append(prop_zero_nine_coh_down_resp)
    prop_cohs_subsampled_down_resp.append(prop_two_seven_coh_down_resp)
    prop_cohs_subsampled_down_resp.append(prop_eight_one_coh_down_resp)    
    
    prop_cohs_subsampled_up_resp_orig = []
    prop_cohs_subsampled_up_resp_orig.append(prop_zero_coh_up_resp_orig)
    prop_cohs_subsampled_up_resp_orig.append(prop_zero_three_coh_up_resp_orig)
    prop_cohs_subsampled_up_resp_orig.append(prop_zero_nine_coh_up_resp_orig)
    prop_cohs_subsampled_up_resp_orig.append(prop_two_seven_coh_up_resp_orig)
    prop_cohs_subsampled_up_resp_orig.append(prop_eight_one_coh_up_resp_orig)
    
    prop_cohs_subsampled_down_resp_orig = []
    prop_cohs_subsampled_down_resp_orig.append(prop_zero_coh_down_resp_orig)
    prop_cohs_subsampled_down_resp_orig.append(prop_zero_three_coh_down_resp_orig)
    prop_cohs_subsampled_down_resp_orig.append(prop_zero_nine_coh_down_resp_orig)
    prop_cohs_subsampled_down_resp_orig.append(prop_two_seven_coh_down_resp_orig)
    prop_cohs_subsampled_down_resp_orig.append(prop_eight_one_coh_down_resp_orig)      
    
    return prop_cohs_subsampled_up_resp, prop_cohs_subsampled_down_resp, prop_cohs_subsampled_up_resp_orig, prop_cohs_subsampled_down_resp_orig


subjects = [2, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 42]

prop_cohs_subsampled_up_resp_low_tert_all_sess_all_sjs = []
prop_cohs_subsampled_down_resp_low_tert_all_sess_all_sjs = []
prop_cohs_subsampled_up_resp_high_tert_all_sess_all_sjs = []
prop_cohs_subsampled_down_resp_high_tert_all_sess_all_sjs = []

prop_cohs_subsampled_up_resp_low_tert_orig_all_sess_all_sjs = []
prop_cohs_subsampled_down_resp_low_tert_orig_all_sess_all_sjs = []
prop_cohs_subsampled_up_resp_high_tert_orig_all_sess_all_sjs = []
prop_cohs_subsampled_down_resp_high_tert_orig_all_sess_all_sjs = []

for sj in subjects: 
    
    prop_cohs_subsampled_up_resp_low_tert_all_sess = []
    prop_cohs_subsampled_down_resp_low_tert_all_sess = []
    prop_cohs_subsampled_up_resp_high_tert_all_sess = []
    prop_cohs_subsampled_down_resp_high_tert_all_sess = []

    prop_cohs_subsampled_up_resp_low_tert_orig_all_sess = []
    prop_cohs_subsampled_down_resp_low_tert_orig_all_sess = []
    prop_cohs_subsampled_up_resp_high_tert_orig_all_sess = []
    prop_cohs_subsampled_down_resp_high_tert_orig_all_sess = []
    if sj in [10, 11]:
        sessions = range(2,4)
    elif sj == 12:
        sessions = [1]
    else: 
        sessions = range(1,4)
    for sess in sessions:
        prop_cohs_subsampled_up_resp_right_is_up_low_tert, prop_cohs_subsampled_down_resp_right_is_up_low_tert, prop_cohs_subsampled_up_resp_right_is_up_low_tert_orig, prop_cohs_subsampled_down_resp_right_is_up_low_tert_orig = get_contrasts(contrasts = contrasts_lhipsi, subject = sj, session = sess, cond= "bias_low_tertile_right_is_up")

        prop_cohs_subsampled_up_resp_left_is_up_low_tert, prop_cohs_subsampled_down_resp_left_is_up_low_tert, prop_cohs_subsampled_up_resp_left_is_up_low_tert_orig, prop_cohs_subsampled_down_resp_left_is_up_low_tert_orig = get_contrasts(contrasts = contrasts_lhipsi, subject = sj, session = sess, cond= "bias_low_tertile_left_is_up")


        prop_cohs_subsampled_up_resp_right_is_up_high_tert, prop_cohs_subsampled_down_resp_right_is_up_high_tert, prop_cohs_subsampled_up_resp_right_is_up_high_tert_orig, prop_cohs_subsampled_down_resp_right_is_up_high_tert_orig = get_contrasts(contrasts = contrasts_lhipsi, subject = sj, session = sess, cond= "bias_high_tertile_right_is_up")

        prop_cohs_subsampled_up_resp_left_is_up_high_tert, prop_cohs_subsampled_down_resp_left_is_up_high_tert, prop_cohs_subsampled_up_resp_left_is_up_high_tert_orig, prop_cohs_subsampled_down_resp_left_is_up_high_tert_orig = get_contrasts(contrasts = contrasts_lhipsi, subject = sj, session = sess, cond= "bias_high_tertile_left_is_up")
    
        prop_cohs_subsampled_up_resp_low_tert_all_sess.append(prop_cohs_subsampled_up_resp_right_is_up_low_tert)
        prop_cohs_subsampled_up_resp_low_tert_all_sess.append(prop_cohs_subsampled_up_resp_left_is_up_low_tert)

        prop_cohs_subsampled_down_resp_low_tert_all_sess.append(prop_cohs_subsampled_down_resp_right_is_up_low_tert)
        prop_cohs_subsampled_down_resp_low_tert_all_sess.append(prop_cohs_subsampled_down_resp_left_is_up_low_tert)

        prop_cohs_subsampled_up_resp_high_tert_all_sess.append(prop_cohs_subsampled_up_resp_right_is_up_high_tert)
        prop_cohs_subsampled_up_resp_high_tert_all_sess.append(prop_cohs_subsampled_up_resp_left_is_up_high_tert)

        prop_cohs_subsampled_down_resp_high_tert_all_sess.append(prop_cohs_subsampled_down_resp_right_is_up_high_tert)
        prop_cohs_subsampled_down_resp_high_tert_all_sess.append(prop_cohs_subsampled_down_resp_left_is_up_high_tert)
    

        prop_cohs_subsampled_up_resp_low_tert_orig_all_sess.append(prop_cohs_subsampled_up_resp_right_is_up_low_tert_orig)
        prop_cohs_subsampled_up_resp_low_tert_orig_all_sess.append(prop_cohs_subsampled_up_resp_left_is_up_low_tert_orig)

        prop_cohs_subsampled_down_resp_low_tert_orig_all_sess.append(prop_cohs_subsampled_down_resp_right_is_up_low_tert_orig)
        prop_cohs_subsampled_down_resp_low_tert_orig_all_sess.append(prop_cohs_subsampled_down_resp_left_is_up_low_tert_orig)

        prop_cohs_subsampled_up_resp_high_tert_orig_all_sess.append(prop_cohs_subsampled_up_resp_right_is_up_high_tert_orig)
        prop_cohs_subsampled_up_resp_high_tert_orig_all_sess.append(prop_cohs_subsampled_up_resp_left_is_up_high_tert_orig)

        prop_cohs_subsampled_down_resp_high_tert_orig_all_sess.append(prop_cohs_subsampled_down_resp_right_is_up_high_tert_orig)
        prop_cohs_subsampled_down_resp_high_tert_orig_all_sess.append(prop_cohs_subsampled_down_resp_left_is_up_high_tert_orig)
    
    mean_prop_cohs_subsampled_up_resp_low_tert_all_sess = np.nanmean(prop_cohs_subsampled_up_resp_low_tert_all_sess, axis = 0)
    mean_prop_cohs_subsampled_down_resp_low_tert_all_sess = np.nanmean(prop_cohs_subsampled_down_resp_low_tert_all_sess, axis = 0)
    mean_prop_cohs_subsampled_up_resp_high_tert_all_sess = np.nanmean(prop_cohs_subsampled_up_resp_high_tert_all_sess, axis = 0)
    mean_prop_cohs_subsampled_down_resp_high_tert_all_sess = np.nanmean(prop_cohs_subsampled_down_resp_high_tert_all_sess, axis = 0)

    mean_prop_cohs_subsampled_up_resp_low_tert_orig_all_sess = np.nanmean(prop_cohs_subsampled_up_resp_low_tert_orig_all_sess, axis = 0)
    mean_prop_cohs_subsampled_down_resp_low_tert_orig_all_sess = np.nanmean(prop_cohs_subsampled_down_resp_low_tert_orig_all_sess, axis = 0)
    mean_prop_cohs_subsampled_up_resp_high_tert_orig_all_sess = np.nanmean(prop_cohs_subsampled_up_resp_high_tert_orig_all_sess, axis = 0)
    mean_prop_cohs_subsampled_down_resp_high_tert_orig_all_sess = np.nanmean(prop_cohs_subsampled_down_resp_high_tert_orig_all_sess, axis = 0)

    prop_cohs_subsampled_up_resp_low_tert_all_sess_all_sjs.append(mean_prop_cohs_subsampled_up_resp_low_tert_all_sess)
    prop_cohs_subsampled_down_resp_low_tert_all_sess_all_sjs.append(mean_prop_cohs_subsampled_down_resp_low_tert_all_sess)
    prop_cohs_subsampled_up_resp_high_tert_all_sess_all_sjs.append(mean_prop_cohs_subsampled_up_resp_high_tert_all_sess)
    prop_cohs_subsampled_down_resp_high_tert_all_sess_all_sjs.append(mean_prop_cohs_subsampled_down_resp_high_tert_all_sess)
    
    prop_cohs_subsampled_up_resp_low_tert_orig_all_sess_all_sjs.append(mean_prop_cohs_subsampled_up_resp_low_tert_orig_all_sess)
    prop_cohs_subsampled_down_resp_low_tert_orig_all_sess_all_sjs.append(mean_prop_cohs_subsampled_down_resp_low_tert_orig_all_sess)
    prop_cohs_subsampled_up_resp_high_tert_orig_all_sess_all_sjs.append(mean_prop_cohs_subsampled_up_resp_high_tert_orig_all_sess)
    prop_cohs_subsampled_down_resp_high_tert_orig_all_sess_all_sjs.append(mean_prop_cohs_subsampled_down_resp_high_tert_orig_all_sess)

df_up_resp_low_tert = pd.DataFrame(prop_cohs_subsampled_up_resp_low_tert_all_sess_all_sjs, columns = ['zero', 'three', 'nine', 'two_seven', 'eight_one'])#.to_csv('prop_cohs_subsampled_up_resp_low_tert_all_sess_all_sjs.csv', sep = '\t')
df_down_resp_low_tert = pd.DataFrame(prop_cohs_subsampled_down_resp_low_tert_all_sess_all_sjs, columns = ['zero', 'three', 'nine', 'two_seven', 'eight_one'])#.to_csv('prop_cohs_subsampled_down_resp_low_tert_all_sess_all_sjs.csv', sep = '\t')
df_up_resp_high_tert = pd.DataFrame(prop_cohs_subsampled_up_resp_high_tert_all_sess_all_sjs, columns = ['zero', 'three', 'nine', 'two_seven', 'eight_one'])#.to_csv('prop_cohs_subsampled_up_resp_high_tert_all_sess_all_sjs.csv', sep = '\t')
df_down_resp_high_tert = pd.DataFrame(prop_cohs_subsampled_down_resp_high_tert_all_sess_all_sjs, columns = ['zero', 'three', 'nine', 'two_seven', 'eight_one'])#.to_csv('prop_cohs_subsampled_down_resp_high_tert_all_sess_all_sjs.csv', sep = '\t')

df_up_resp_low_tert_orig = pd.DataFrame(prop_cohs_subsampled_up_resp_low_tert_orig_all_sess_all_sjs, columns = ['zero', 'three', 'nine', 'two_seven', 'eight_one'])#.to_csv('prop_cohs_subsampled_up_resp_low_tert_orig_all_sess_all_sjs.csv', sep = '\t')
df_down_resp_low_tert_orig = pd.DataFrame(prop_cohs_subsampled_down_resp_low_tert_orig_all_sess_all_sjs, columns = ['zero', 'three', 'nine', 'two_seven', 'eight_one'])#.to_csv('prop_cohs_subsampled_down_resp_low_tert_orig_all_sess_all_sjs.csv', sep = '\t')
df_up_resp_high_tert_orig = pd.DataFrame(prop_cohs_subsampled_up_resp_high_tert_orig_all_sess_all_sjs, columns = ['zero', 'three', 'nine', 'two_seven', 'eight_one'])#.to_csv('prop_cohs_subsampled_up_resp_high_tert_orig_all_sess_all_sjs.csv', sep = '\t')
df_down_resp_high_tert_orig = pd.DataFrame(prop_cohs_subsampled_down_resp_high_tert_orig_all_sess_all_sjs, columns = ['zero', 'three', 'nine', 'two_seven', 'eight_one'])#.to_csv('prop_cohs_subsampled_down_resp_high_tert_orig_all_sess_all_sjs.csv', sep = '\t')
 
df_up_resp_low_tert.to_csv('prop_cohs_subsampled_up_resp_low_tert_all_sess_all_sjs.csv', sep = '\t')
df_down_resp_low_tert.to_csv('prop_cohs_subsampled_down_resp_low_tert_all_sess_all_sjs.csv', sep = '\t')
df_up_resp_high_tert.to_csv('prop_cohs_subsampled_up_resp_high_tert_all_sess_all_sjs.csv', sep = '\t')
df_down_resp_high_tert.to_csv('prop_cohs_subsampled_down_resp_high_tert_all_sess_all_sjs.csv', sep = '\t')
df_up_resp_low_tert_orig.to_csv('prop_cohs_subsampled_up_resp_low_tert_orig_all_sess_all_sjs.csv', sep = '\t')
df_down_resp_low_tert_orig.to_csv('prop_cohs_subsampled_down_resp_low_tert_orig_all_sess_all_sjs.csv', sep = '\t')
df_up_resp_high_tert_orig.to_csv('prop_cohs_subsampled_up_resp_high_tert_orig_all_sess_all_sjs.csv', sep = '\t')
df_down_resp_high_tert_orig.to_csv('prop_cohs_subsampled_down_resp_high_tert_orig_all_sess_all_sjs.csv', sep = '\t')



df_down_high = pd.read_csv('prop_cohs_subsampled_down_resp_high_tert_all_sess_all_sjs.csv', sep = '\t', usecols=['zero', 'three', 'nine', 'two_seven', 'eight_one'],index_col = None)
df_up_high = pd.read_csv('prop_cohs_subsampled_up_resp_high_tert_all_sess_all_sjs.csv', sep = '\t', usecols=['zero', 'three', 'nine', 'two_seven', 'eight_one'],index_col = None)
df_down_low = pd.read_csv('prop_cohs_subsampled_down_resp_low_tert_all_sess_all_sjs.csv', sep = '\t', usecols=['zero', 'three', 'nine', 'two_seven', 'eight_one'],index_col = None)
df_up_low = pd.read_csv('prop_cohs_subsampled_up_resp_low_tert_all_sess_all_sjs.csv', sep = '\t', usecols=['zero', 'three', 'nine', 'two_seven', 'eight_one'],index_col = None)

df_orig_down_high = pd.read_csv('prop_cohs_subsampled_down_resp_high_tert_orig_all_sess_all_sjs.csv', sep = '\t', usecols=['zero', 'three', 'nine', 'two_seven', 'eight_one'],index_col = None)
df_orig_up_high = pd.read_csv('prop_cohs_subsampled_up_resp_high_tert_orig_all_sess_all_sjs.csv', sep = '\t', usecols=['zero', 'three', 'nine', 'two_seven', 'eight_one'],index_col = None)
df_orig_down_low = pd.read_csv('prop_cohs_subsampled_down_resp_low_tert_orig_all_sess_all_sjs.csv', sep = '\t', usecols=['zero', 'three', 'nine', 'two_seven', 'eight_one'],index_col = None)
df_orig_up_low = pd.read_csv('prop_cohs_subsampled_up_resp_low_tert_orig_all_sess_all_sjs.csv', sep = '\t', usecols=['zero', 'three', 'nine', 'two_seven', 'eight_one'],index_col = None)


fig = plt.figure(figsize=(10,10))
ax1 = plt.subplot2grid((2,2), (0, 0))
ax2 = plt.subplot2grid((2,2), (0, 1))
ax3 = plt.subplot2grid((2,2), (1, 0))
ax4 = plt.subplot2grid((2,2), (1, 1))

#    ax3 = plt.subplot2grid((2,3), (0, 2))    
axes = [ax1, ax2, ax3, ax4]

plt.subplots_adjust(left=0.1, bottom=0.15, top = 0.85, right=.9, wspace = .5, hspace = .7) 



contrasts = ['0.0', '0.03', '0.09', '0.27', '0.81']
ax1.bar(contrasts, df_up_high.mean().values, yerr = stats.sem(df_up_high), align='center', alpha=0.5, ecolor='black', capsize=10)
ax2.bar(contrasts, df_orig_up_high.mean().values, yerr = stats.sem(df_orig_up_high), align='center', alpha=0.5, ecolor='black', capsize=10)
ax3.bar(contrasts, df_down_high.mean().values, yerr = stats.sem(df_down_high), align='center', alpha=0.5, ecolor='black', capsize=10)
ax4.bar(contrasts, df_orig_down_high.mean().values, yerr = stats.sem(df_orig_down_high), align='center', alpha=0.5, ecolor='black', capsize=10)
ax1.set_title('Up resp. high bin subsampled')
ax2.set_title('Up resp. high bin orig. data')
ax3.set_title('Down resp. high bin subsampled')
ax4.set_title('Down resp. high bin orig. data')
for ax in axes:
    ax.set_ylim(0, 0.3)
    sns.despine(ax = ax, offset=10, right =True, left = False)
    ax.set_ylabel('Proportion of trials')
    ax.set_xlabel('Motion coherence')
plt.savefig('Proportion_coherences_subsamples_high_bin.pdf')
plt.show()



fig = plt.figure(figsize=(10,10))
ax1 = plt.subplot2grid((2,2), (0, 0))
ax2 = plt.subplot2grid((2,2), (0, 1))
ax3 = plt.subplot2grid((2,2), (1, 0))
ax4 = plt.subplot2grid((2,2), (1, 1))

#    ax3 = plt.subplot2grid((2,3), (0, 2))    
axes = [ax1, ax2, ax3, ax4]

plt.subplots_adjust(left=0.1, bottom=0.15, top = 0.85, right=.9, wspace = .5, hspace = .7) 



contrasts = ['0.0', '0.03', '0.09', '0.27', '0.81']
ax1.bar(contrasts, df_up_low.mean().values, yerr = stats.sem(df_up_low), align='center', alpha=0.5, ecolor='black', capsize=10)
ax2.bar(contrasts, df_orig_up_low.mean().values, yerr = stats.sem(df_orig_up_low), align='center', alpha=0.5, ecolor='black', capsize=10)
ax3.bar(contrasts, df_down_low.mean().values, yerr = stats.sem(df_down_low), align='center', alpha=0.5, ecolor='black', capsize=10)
ax4.bar(contrasts, df_orig_down_low.mean().values, yerr = stats.sem(df_orig_down_low), align='center', alpha=0.5, ecolor='black', capsize=10)
ax1.set_title('Up resp. low bin subsampled')
ax2.set_title('Up resp. low bin orig. data')
ax3.set_title('Down resp. low bin subsampled')
ax4.set_title('Down resp. low bin orig. data')
for ax in axes:
    ax.set_ylim(0, 0.3)
    sns.despine(ax = ax, offset=10, right =True, left = False)
    ax.set_ylabel('Proportion of trials')
    ax.set_xlabel('Motion coherence')
plt.savefig('Proportion_coherences_subsamples_low_bin.pdf')
plt.show()



fig = plt.figure(figsize=(10,10))
ax1 = plt.subplot2grid((2,2), (0, 0))
ax2 = plt.subplot2grid((2,2), (0, 1))
ax3 = plt.subplot2grid((2,2), (1, 0))
ax4 = plt.subplot2grid((2,2), (1, 1))

#    ax3 = plt.subplot2grid((2,3), (0, 2))    
axes = [ax1, ax2, ax3, ax4]

plt.subplots_adjust(left=0.1, bottom=0.15, top = 0.85, right=.9, wspace = .5, hspace = .7) 


df_concat_up = pd.concat((df_up_high, df_up_low))
by_row_index_up = df_concat_up.groupby(df_concat_up.index)
df_means_up = by_row_index_up.mean()

df_concat_down = pd.concat((df_down_high, df_down_low))
by_row_index_down = df_concat_down.groupby(df_concat_down.index)
df_means_down = by_row_index_down.mean()

df_orig_concat_up = pd.concat((df_orig_up_high, df_orig_up_low))
by_row_index_orig_up = df_orig_concat_up.groupby(df_orig_concat_up.index)
df_orig_means_up = by_row_index_orig_up.mean()

df_orig_concat_down = pd.concat((df_orig_down_high, df_orig_down_low))
by_row_index_orig_down = df_orig_concat_down.groupby(df_orig_concat_down.index)
df_orig_means_down = by_row_index_orig_down.mean()

contrasts = ['0.0', '0.03', '0.09', '0.27', '0.81']
ax1.bar(contrasts, df_means_up.mean().values, yerr = stats.sem(df_means_up), align='center', alpha=0.5, ecolor='black', capsize=10)
ax2.bar(contrasts, df_orig_means_up.mean().values, yerr = stats.sem(df_orig_means_up), align='center', alpha=0.5, ecolor='black', capsize=10)
ax3.bar(contrasts, df_means_down.mean().values, yerr = stats.sem(df_means_down), align='center', alpha=0.5, ecolor='black', capsize=10)
ax4.bar(contrasts, df_orig_means_down.mean().values, yerr = stats.sem(df_orig_means_down), align='center', alpha=0.5, ecolor='black', capsize=10)
ax1.set_title('Up resp. avg. across low and high bin subsampled')
ax2.set_title('Up resp. avg. across low and high bin orig. data')
ax3.set_title('Down resp. avg. across low and high bin subsampled')
ax4.set_title('Down resp. avg. across low and high bin orig. data')
for ax in axes:
    ax.set_ylim(0, 0.3)
    sns.despine(ax = ax, offset=10, right =True, left = False)
    ax.set_ylabel('Proportion of trials')
    ax.set_xlabel('Motion coherence')
plt.savefig('Proportion_coherences_subsamples_mean_low_high_bin.pdf')
plt.show()





