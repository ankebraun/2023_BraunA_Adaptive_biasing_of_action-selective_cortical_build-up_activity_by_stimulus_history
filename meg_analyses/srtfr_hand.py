import os
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/mnt/homes/home024/abraun/pymeg')
sys.path.insert(0, "/Users/anke/cluster_home/anaconda/envs/py36/bin/mne")
import pymeg
from pymeg.contrast_tfr_baseline_per_session import Cache, compute_contrast, augment_data
from pymeg import contrast_tfr_baseline_per_session
from pymeg import parallel
from joblib import Memory
import logging
import glob
import pdb

from optparse import OptionParser
from os.path import join

usage = "srtfr_hand.py [options] <sj_number>"

long_help = """
This script extracts the time frequency responses contra- vs. ipsilateral to the previous button-press
and contra- vs. ipsi to current button-press separately for current correct and error for one session.

This script was adapted from Niklas Wilming et al. Nature Communications 2020
https://github.com/DonnerLab/2020_Large-scale-Dynamics-of-Perceptual-Decision-Information-across-Human-Cortex
"""

parser = OptionParser(usage, epilog=long_help)


opts, args = parser.parse_args()

logging.getLogger().setLevel(logging.INFO)
logger = contrast_tfr_baseline_per_session.logging.getLogger()
logger.setLevel(logging.INFO)


meta_path = '/home/abraun/meg_data'
data_path = '/home/abraun/meg_data/sr_labeled/sf600/'
fig_folder = '/home/abraun/meg_data/sr_labeled/figures'

memory = Memory(cachedir=os.environ['PYMEG_CACHE_DIR'], verbose=0)
path = '/home/abraun/meg_data/sr_contrasts'


contrasts_lhipsi = {
    'prev_hand': (['prev_left_all_trls', 'prev_right_all_trls'], (-0.5, 0.5)),
    'Repetitive_prev_hand': (['Repetitive_prev_left_all_trls', 'Repetitive_prev_right_all_trls'], (-0.5, 0.5)),
    'Neutral_prev_hand': (['Neutral_prev_left_all_trls', 'Neutral_prev_right_all_trls'], (-0.5, 0.5)),
    'Alternating_prev_hand': (['Alternating_prev_left_all_trls', 'Alternating_prev_right_all_trls'], (-0.5, 0.5)),
    'hand_current_correct': (['left_current_correct', 'right_current_correct'], (-0.5, 0.5)),
    'hand_current_error': (['left_current_error', 'right_current_error'], (-0.5, 0.5)),
}


def srtfrfilename(contrasts, subject, session):
    try:
        makedirs(path)
    except:
        pass
    subject = int(subject)
    filename = f'S{subject}-{contrasts}_baseline_per_session_session_{session}_prev_hand.hdf'
    return join(path, filename)


@memory.cache()
def get_contrasts(contrasts, subject, session, hemi, baseline_per_condition=False):
    stim, resp, meta_data_filenames, meta_data_filenames_before_artf_reject = [], [], [], []

    stim.append(join(data_path, f'S{subject}-SESS{session}-REC*-stimulus-*-lcmv.hdf'))

    resp.append(join(data_path, f'S{subject}-SESS{session}-REC*-response-*-lcmv.hdf'))

    sessions = range(session, session+1)

    for session in sessions:
        if int(subject) == 5 and session == 1:
            recs = [2]
        else:
            recs = [1]

        for rec in recs:
            if int(subject) < 10:
                meta_data_filenames.append(
                    join(
                        meta_path, 'P0{}/MEG/Locked/P0{}-S{}_rec{}_stim_new.hdf'.format(subject, subject, session, rec)
                    )
                )
                meta_data_filenames_before_artf_reject.append(
                    join(meta_path, 'P0{}/MEG/Preproc/P0{}-S{}_rec{}_data.hdf'.format(subject, subject, session, rec))
                )
            elif int(subject) >= 10:
                meta_data_filenames.append(
                    join(meta_path, 'P{}/MEG/Locked/P{}-S{}_rec{}_stim_new.hdf'.format(subject, subject, session, rec))
                )
                meta_data_filenames_before_artf_reject.append(
                    join(meta_path, 'P{}/MEG/Preproc/P{}-S{}_rec{}_data.hdf'.format(subject, subject, session, rec))
                )

    ###### for contrasts based on previous trials take data from all trials before artifact rejection
    ###### and match with trials after artifact rejection

    meta_data = pd.concat([pd.read_hdf(meta_data_filename, 'df') for meta_data_filename in meta_data_filenames])
    meta_data_before_artf_reject = pd.concat(
        [
            pd.read_hdf(meta_data_filename_before_artf_reject, 'df')
            for meta_data_filename_before_artf_reject in meta_data_filenames_before_artf_reject
        ]
    )

    idx_after_artf_reject = meta_data['idx']
    meta_data_all_trls = meta_data_before_artf_reject[(meta_data_before_artf_reject['idx']).isin(idx_after_artf_reject)]
    meta_data["all"] = 1
    meta_data['respbutton'][meta_data['respbutton'] == 75] = 12  # fix mixed codings for respbuttons
    meta_data['respbutton'][meta_data['respbutton'] == 77] = 18  # fix mixed codings for respbuttons

    meta_data_all_trls['respbutton'][meta_data_all_trls['respbutton'] == 75] = 12  # fix mixed codings for respbuttons
    meta_data_all_trls['respbutton'][meta_data_all_trls['respbutton'] == 77] = 18  # fix mixed codings for respbuttons

    meta_data_all_trls['prev_respbutton'][
        meta_data_all_trls['prev_respbutton'] == 75
    ] = 12  # fix mixed codings for respbuttons
    meta_data_all_trls['prev_respbutton'][
        meta_data_all_trls['prev_respbutton'] == 77
    ] = 18  # fix mixed codings for respbuttons

    meta_data['respbutton'][meta_data['respbutton'] == 8] = 12  # fix mixed codings for respbuttons
    meta_data['respbutton'][meta_data['respbutton'] == 1] = 18  # fix mixed codings for respbuttons

    meta_data_all_trls['respbutton'][meta_data_all_trls['respbutton'] == 8] = 12  # fix mixed codings for respbuttons
    meta_data_all_trls['respbutton'][meta_data_all_trls['respbutton'] == 1] = 18  # fix mixed codings for respbuttons

    meta_data["Repetitive"] = (meta_data["alternation_prob"] == 0.2).astype(float)
    meta_data["Neutral"] = (meta_data["alternation_prob"] == 0.5).astype(float)
    meta_data["Alternating"] = (meta_data["alternation_prob"] == 0.8).astype(float)

    rep_blocks = np.unique(meta_data[meta_data["alternation_prob"] == 0.2].blockcnt)
    alt_blocks = np.unique(meta_data[meta_data["alternation_prob"] == 0.8].blockcnt)
    neutr_blocks = np.unique(meta_data[meta_data["alternation_prob"] == 0.5].blockcnt)

    meta_data_rep = meta_data[(meta_data.alternation_prob == 0.2)]

    meta_data_alt = meta_data[(meta_data.alternation_prob == 0.8)]

    meta_data_neutr = meta_data[(meta_data.alternation_prob == 0.5)]

    meta_data["stim_repetition"] = (np.roll(meta_data.stimtype, 1) == meta_data.stimtype).astype(bool)
    meta_data["left"] = (meta_data["respbutton"] == 12).astype(bool)
    meta_data["right"] = (meta_data["respbutton"] == 18).astype(bool)

    meta_data["prev_left_all_trls"] = (meta_data_all_trls["prev_respbutton"].values == 12).astype(bool)
    meta_data["prev_right_all_trls"] = (meta_data_all_trls["prev_respbutton"].values == 18).astype(bool)
    
    meta_data["left_current_correct"] = (meta_data["respbutton"] == 12)&(meta_data.respcorrect.values == 1)
    meta_data["right_current_correct"] = (meta_data["respbutton"] == 18)&(meta_data.respcorrect.values == 1)
    meta_data["left_current_error"] = (meta_data["respbutton"] == 12)&(meta_data.respcorrect.values != 1)
    meta_data["right_current_error"] = (meta_data["respbutton"] == 18)&(meta_data.respcorrect.values != 1)

    for col in ['Repetitive', 'Neutral', 'Alternating']:
        meta_data.loc[meta_data["prev_left_all_trls"], col + '_prev_left_all_trls'] = meta_data.loc[
            meta_data["prev_left_all_trls"], col
        ]
        meta_data.loc[meta_data["prev_right_all_trls"], col + '_prev_right_all_trls'] = meta_data.loc[
            meta_data["prev_right_all_trls"], col
        ]

    hemis = [hemi]
    meta_data.loc[:, 'hash'] = meta_data.loc[:, 'idx']

    cps = []
    new_contrasts = dict()
    for key, value in contrasts.items():
        print(key)
        print(np.sum(meta_data[value[0]]))

    contrast_file = srtfrfilename(hemi, subject, session)
    if os.path.isfile(contrast_file):  # load already existing contrasts
        contrast_data = pd.read_hdf(contrast_file)
        print(contrast_data.index.get_level_values('contrast'))
        for key, value in contrasts.items():  # check which contrasts already exist in that file
            print('key', key)
            print('value', value)
            if key not in contrast_data.index.get_level_values('contrast'):
                new_contrasts[key] = value

        if bool(new_contrasts):
            with Cache() as cache:  # compute non existing contrasts
                contrast = compute_contrast(
                    new_contrasts,
                    hemis,
                    stim,
                    stim,
                    meta_data,
                    (-0.35, -0.1),
                    baseline_per_condition=False,
                    n_jobs=1,
                    cache=cache,
                )
                contrast.loc[:, 'epoch'] = 'stimulus'
                cps.append(contrast)
                contrast = compute_contrast(
                    new_contrasts,
                    hemis,
                    resp,
                    stim,
                    meta_data,
                    (-0.35, -0.1),
                    baseline_per_condition=False,
                    n_jobs=1,
                    cache=cache,
                )
                contrast.loc[:, 'epoch'] = 'response'
                cps.append(contrast)
            contrast = pd.concat(cps)
            del cps
            contrast.loc[:, 'subject'] = subject
            contrast.set_index(['subject', 'contrast', 'hemi', 'epoch', 'cluster'], append=True, inplace=True)

            contrast = pd.concat([contrast, contrast_data])  # concatenate new and old contrasts
            filename = srtfrfilename(hemi, subject, session)
            contrast.to_hdf(filename, 'epochs')
            return contrast

    else:  # if no contrasts exist already
        with Cache() as cache:
            contrast = compute_contrast(
                contrasts,
                hemis,
                stim,
                stim,
                meta_data,
                (-0.35, -0.1),
                baseline_per_condition=False,
                n_jobs=1,
                cache=cache,
            )
            contrast.loc[:, 'epoch'] = 'stimulus'
            cps.append(contrast)
            contrast = compute_contrast(
                contrasts,
                hemis,
                resp,
                stim,
                meta_data,
                (-0.35, -0.1),
                baseline_per_condition=False,
                n_jobs=1,
                cache=cache,
            )
            contrast.loc[:, 'epoch'] = 'response'
            cps.append(contrast)
        contrast = pd.concat(cps)

        del cps
        contrast.loc[:, 'subject'] = subject
        contrast.set_index(['subject', 'contrast', 'hemi', 'epoch', 'cluster'], append=True, inplace=True)
        filename = srtfrfilename(hemi, subject, session)
        contrast.to_hdf(filename, 'epochs')
        return contrast


subject = str(args[0])
session = 1
get_contrasts(contrasts_lhipsi, subject, session, 'lh_is_ipsi', baseline_per_condition=False)

