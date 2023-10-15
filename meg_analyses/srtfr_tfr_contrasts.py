import os
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "/home/abraun/pymeg")
import pymeg
from pymeg.contrast_tfr import Cache, compute_contrast, augment_data
from pymeg import contrast_tfr
from pymeg import parallel
from joblib import Memory
import logging
import glob
import pdb

from optparse import OptionParser
from os.path import join
usage = "srtfr_anke.py [options] <sj_number>"

long_help = """
This script computes the time frequency responses for all trials, for high coherence trials, 
for zero coherence trials and contra- vs. ipsilateral to current button-press.

This script was adapted from Niklas Wilming et al. Nature Communications 2020
https://github.com/DonnerLab/2020_Large-scale-Dynamics-of-Perceptual-Decision-Information-across-Human-Cortex
"""

parser = OptionParser ( usage, epilog=long_help )


opts,args = parser.parse_args ()

logging.getLogger().setLevel(logging.INFO)
logger = contrast_tfr.logging.getLogger()
logger.setLevel(logging.INFO)


meta_path = '/home/abraun/meg_data'
data_path = '/home/abraun/meg_data/sr_labeled/'
fig_folder = '/home/abraun/meg_data/sr_labeled/figures'

memory = Memory(cachedir=os.environ['PYMEG_CACHE_DIR'], verbose=0)
path = '/home/abraun/meg_data/sr_contrasts'

contrasts_lhipsi = {
    'all': (['all'], [1]),
    '0.81_coh': (['0.81_coh'], [1]),
    '0_coh': (['0_coh'], [1]),
    'hand': (['left', 'right'], (-0.5, 0.5)),
}


def srtfrfilename(contrasts, subject, session):
    try:
        makedirs(path)
    except:
        pass
    subject = int(subject)
    filename = f'S{subject}-{contrasts}_session_{session}.hdf' 
    return join(path, filename)


@memory.cache()
def get_contrasts(contrasts, subject, session, hemi, baseline_per_condition=False):

    stim, resp, meta_data_filenames = [], [], []
    

    stim.append(join(data_path, f'S{subject}-SESS{session}-REC*-stimulus-*-lcmv.hdf' ))

    resp.append(join(data_path, f'S{subject}-SESS{session}-REC*-response-*-lcmv.hdf'))
    
    sessions = range(session,session+1)

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
    
    meta_data = pd.concat(
        [pd.read_hdf(meta_data_filename, 'df')
            for meta_data_filename in meta_data_filenames])

    meta_data['respbutton'][meta_data['respbutton'] == 75] = 12 # fix mixed codings for respbuttons
    meta_data['respbutton'][meta_data['respbutton'] == 77] = 18 # fix mixed codings for respbuttons
    meta_data['respbutton'][meta_data['respbutton'] == 8] = 12 # fix mixed codings for respbuttons
    meta_data['respbutton'][meta_data['respbutton'] == 1] = 18 # fix mixed codings for respbuttons
    meta_data["all"] = 1
    meta_data["0.81_coh"] = (meta_data["motion_coherence"] == 0.81).astype(float)
    meta_data["0_coh"] = (meta_data["motion_coherence"] == 0.00).astype(float)
    meta_data["Repetitive"] = (meta_data["alternation_prob"] == 0.2).astype(float)
    meta_data["Neutral"] = (meta_data["alternation_prob"] == 0.5).astype(float)
    meta_data["Alternating"] = (meta_data["alternation_prob"] == 0.8).astype(float) 
    meta_data["left"] = (meta_data["respbutton"] == 12).astype(bool)
    meta_data["right"] = (meta_data["respbutton"] == 18).astype(bool)
        


    for col in ['Repetitive', 'Neutral', 'Alternating']:
        meta_data.loc[meta_data["left"], col + '_left'] = meta_data.loc[meta_data["left"], col]
        meta_data.loc[meta_data["right"], col + '_right'] = meta_data.loc[meta_data["right"], col]
        print(np.sum(meta_data.loc[meta_data["left"], col + '_left']))      


    hemis = [hemi, 'avg']
    meta_data.loc[:, 'hash'] = meta_data.loc[:, 'idx']

    cps = []
    new_contrasts = dict()
    for key, value in contrasts.items():
 #       print(key)
        print(np.sum(meta_data[value[0]]))
    contrast_file = srtfrfilename(hemi, subject, session)
    if os.path.isfile(contrast_file): #load already existing contrasts
        contrast_data = pd.read_hdf(contrast_file)
        print(contrast_data.index.get_level_values('contrast'))
        for key, value in contrasts.items(): # check which contrasts already exist in that file
            print('key', key)
            print('value', value)
            if key not in contrast_data.index.get_level_values('contrast'):
                new_contrasts[key] = value
            # elif key in ['Repetitive_hand', 'Alternating_hand', 'Neutral_hand']:
            #     new_contrasts[key] = value
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
        filename = srtfrfilename(hemi, subject, session)
        #filename = srtfrfilename_single_contrast(hemi, subject, contrasts)
        contrast.to_hdf(filename, 'epochs')
        return contrast


subject = str(args[0])
session = 1
get_contrasts(contrasts_lhipsi, subject, session, 'lh_is_ipsi', baseline_per_condition=False)
