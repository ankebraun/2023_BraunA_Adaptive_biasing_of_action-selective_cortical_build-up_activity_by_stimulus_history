import logging
import os
import sys

import numpy as np

import mne

sys.path.insert(0, '/mnt/homes/home024/abraun/pymeg')
sys.path.insert(0, "/mnt/homes/home024/abraun/conf_meg")
from optparse import OptionParser
from os import makedirs
from os.path import join

import pandas as pd

from joblib import Memory
from pymeg import lcmv as pymeglcmv
from pymeg import source_reconstruction as sr


usage = "lcmv_2.py [options] <sj_number>"

long_help = """
Compute lcmv beamforming.

This script was adapted from Niklas Wilming et al. Nature Communications 2020
https://github.com/DonnerLab/2020_Large-scale-Dynamics-of-Perceptual-Decision-Information-across-Human-Cortex
"""

parser = OptionParser(usage, epilog=long_help)


opts, args = parser.parse_args()


memory = Memory(cachedir=os.environ['PYMEG_CACHE_DIR'])
path = '/home/abraun/meg_data/sr_labeled/'

data_files = {
    'S1': [
        'MH-1_SequentialEffects_20160810_01.ds',
        'MH-2_SequentialEffects_20160811_01.ds',
        #                      'MH-3_SequentialEffects_20160815_01.ds',
        'MH-3_SequentialEffects_20160815_02.ds',
    ],
    'S2': [
        'SE-1_SequentialEffects_20160822_01.ds',
        'SE-2_SequentialEffects_20160907_01.ds',
        'SE-3_SequentialEffects_20160921_01.ds',
    ],
    'S3': [
        'JW-1_SequentialEffects_20160822_01.ds',
        'JW-2_SequentialEffects_20160829_01.ds',
        'JW-3_SequentialEffects_20160831_01.ds',
    ],
    'S5': [
        'DL-1_SequentialEffects_20160829_02.ds',
        'DL-2_SequentialEffects_20160907_01.ds',
        'DL-3_SequentialEffects_20160908_01.ds',
    ],
    'S7': [
        'CK-1_SequentialEffects_20160902_01.ds',
        'CK-2_SequentialEffects_20161014_01.ds',
        # 					  'CK-3_SequentialEffects_20161028_01.ds',
        'CK-3_SequentialEffects_20161028_02.ds',
    ],
    'S8': [
        'MS-1_SequentialEffects_20160902_01.ds',
        'MS-2_SequentialEffects_20160905_01.ds',
        # 'MS-2_SequentialEffects_20160907_02.ds',
        'MS-3_SequentialEffects_20160907_01.ds',
    ],
    'S9': [
        'AW-1_SequentialEffects_20160908_01.ds',
        'AW-2_SequentialEffects_20160909_01.ds',
        'AW-3_SequentialEffects_20161019_01.ds',
    ],
    'S10': [
        'KJ-1_SequentialEffects_20160919_01.ds',
        'KJ-2_SequentialEffects_20160921_01.ds',
        'KJ-3_SequentialEffects_20161004_01.ds',
    ],
    'S11': [
        'AD-1_SequentialEffects_20160919_01.ds',
        'AD-2_SequentialEffects_20160929_01.ds',
        'AD-3_SequentialEffects_20161020_01.ds',
    ],
    'S12': ['LF-1_SequentialEffects_20160928_01.ds'],
    'S13': [
        'VR-1_SequentialEffects_20161010_01.ds',
        'VR-2_SequentialEffects_20161012_01.ds',
        'VR-3_SequentialEffects_20161013_01.ds',
    ],
    'S14': [
        'LA-1_SequentialEffects_20161010_01.ds',
        'LA-2_SequentialEffects_20161011_01.ds',
        'LA-3_SequentialEffects_20161012_01.ds',
    ],
    'S15': [
        'MM-1_SequentialEffects_20161013_01.ds',
        'MM-2_SequentialEffects_20161018_01.ds',
        'MM-3_SequentialEffects_20161028_01.ds',
    ],
    'S16': [
        'NK16-01_SequentEffects_20180222_01.ds',
        'NK16-02_SequentEffects_20180308_01.ds',
        'NK16-03_SequentEffects_20180309_01.ds',
    ],
    'S17': [
        'AZ17-01_SequentEffects_20180226_01.ds',
        'AZ17-02_SequentEffects_20180313_01.ds',
        'AZ17-03_SequentEffects_20180319_01.ds',
    ],
    'S18': [
        'GB18-01_SequentEffects_20180223_01.ds',
        'GB18-02_SequentEffects_20180226_01.ds',
        'GB18-03_SequentEffects_20180307_01.ds',
    ],
    'S19': [
        'KJ19-01_SequentEffects_20180308_01.ds',
        'KJ19-02_SequentEffects_20180321_01.ds',
        'KJ19-03_SequentEffects_20180322_01.ds',
    ],
    'S20': [
        'MB20-01_SequentEffects_20180227_01.ds',
        'MB20-02_SequentEffects_20180308_01.ds',
        'MB20-03_SequentEffects_20180327_01.ds',
    ],
    'S21': [
        'AK21-01_SequentEffects_20180222_01.ds',
        'AK21-02_SequentEffects_20180308_01.ds',
        'AK21-03_SequentEffects_20180424_01.ds',
    ],
    'S22': [
        'GA22-01_SequentEffects_20180219_01.ds',
        'GA22-02_SequentEffects_20180222_01.ds',
        'GA22-03_SequentEffects_20180226_01.ds',
    ],
    'S23': [
        'DB23-01_SequentEffects_20180219_01.ds',
        'DB23-02_SequentEffects_20180220_01.ds',
        'DB23-03_SequentEffects_20180221_01.ds',
    ],
    'S24': [
        'JR24-01_SequentEffects_20180313_01.ds',
        'JR24-02_SequentEffects_20180327_01.ds',
        'JR24-03_SequentEffects_20180405_01.ds',
    ],
    'S25': [
        'RB25-01_SequentEffects_20180307_01.ds',
        'RB25-02_SequentEffects_20180309_01.ds',
        'RB25-03_SequentEffects_20180313_01.ds',
    ],
    'S26': [
        'JF26-01_SequentEffects_20180406_01.ds',
        'JF26-02_SequentEffects_20180410_01.ds',
        'JF26-03_SequentEffects_20180413_01.ds',
    ],
    'S27': [
        'SM27-01_SequentEffects_20180406_01.ds',
        'SM27-02_SequentEffects_20180410_01.ds',
        'SM27-03_SequentEffects_20180412_01.ds',
    ],
    'S28': [
        'HN28-01_SequentEffects_20180423_01.ds',
        'HN28-02_SequentEffects_20180424_01.ds',
        'HN28-03_SequentEffects_20180427_01.ds',
    ],
    'S29': [
        'SS29-01_SequentEffects_20180406_01.ds',
        'SS29-02_SequentEffects_20180423_01.ds',
        'SS29-03_SequentEffects_20180424_01.ds',
    ],
    'S30': [
        'JB30-01_SequentEffects_20180410_01.ds',
        'JB30-02_SequentEffects_20180413_01.ds',
        'JB30-03_SequentEffects_20180427_01.ds',
    ],
    'S31': [
        'CL31-01_SequentEffects_20180405_01.ds',
        'CL31-02_SequentEffects_20180406_01.ds',
        'CL31-03_SequentEffects_20180503_01.ds',
    ],
    'S32': [
        'JR32-01_SequentEffects_20180412_01.ds',
        'JR32-02_SequentEffects_20180413_01.ds',
        'JR32-03_SequentEffects_20180424_01.ds',
    ],
    'S33': [
        'TS33-01_SequentEffects_20180409_01.ds',
        'TS33-02_SequentEffects_20180410_01.ds',
        'TS33-03_SequentEffects_20180413_01.ds',
    ],
    'S34': [
        'ML-34-1_SequentialEffects_20191021_01.ds',
        'ML-34-2_SequentialEffects_20191028_01.ds',
        'ML-34-3_SequentialEffects_20191104_01.ds',
    ],
    'S35': [
        'ML-35-1_SequentialEffects_20191210_01.ds',
        'ML-35-2_SequentialEffects_20191212_01.ds',
        'ML-35-3_SequentialEffects_20200116_01.ds',
    ],
    'S36': [
        'GL-36-1_SequentialEffects_20191118_01.ds',
        'GL-36-2_SequentialEffects_20191203_01.ds',
        'GL-36-3_SequentialEffects_20191205_01.ds',
    ],
    'S37': [
        'OE-37-1_SequentialEffects_20191205_01.ds',
        'OE-37-2_SequentialEffects_20191211_01.ds',
        'OE-37-3_SequentialEffects_20191212_01.ds',
    ],
    'S38': [
        'IA-38-1_SequentialEffects_20191125_01.ds',
        'IA-38-2_SequentialEffects_20191129_01.ds',
        'IA-38-3_SequentialEffects_20191203_01.ds',
    ],
    'S39': [
        'IF-39-1_SequentialEffects_20191202_01.ds',
        'IF-39-2_SequentialEffects_20191203_01.ds',
        'IF-39-3_SequentialEffects_20191204_01.ds',
    ],
    'S40': [
        'ZB-40-1_SequentialEffects_20191202_01.ds',
        'ZB-40-2_SequentialEffects_20191203_01.ds',
        'ZB-40-3_SequentialEffects_20191204_01.ds',
    ],
    'S41': [
        'UT-41-1_SequentialEffects_20191209_01.ds',
        'UT-41-2_SequentialEffects_20191210_01.ds',
        'UT-41-3_SequentialEffects_20191212_01.ds',
    ],
    'S42': [
        'QH-42-1_SequentialEffects_20191216_01.ds',
        'QH-42-2_SequentialEffects_20191218_01.ds',
        'QH-42-3_SequentialEffects_20191219_01.ds',
    ],
}


def set_n_threads(n):
    import os

    os.environ['OPENBLAS_NUM_THREADS'] = str(n)
    os.environ['MKL_NUM_THREADS'] = str(n)
    os.environ['OMP_NUM_THREADS'] = str(n)


def submit():
    from itertools import product

    from pymeg import parallel

    for subject, session, epoch, signal in product(range(3, 11), range(1, 4), ['stimulus'], ['F']):
        #    for subject, session, epoch, signal in product(range(1,34), range(1, 4), ['stimulus', 'response'],
        #            ['F']):
        parallel.pmap(
            extract,
            [(subject, session, epoch, signal)],
            walltime=10,
            memory=40,
            nodes=1,
            tasks=4,
            name='P0' + str(subject) + '-S' + str(session) + '_rec1_' + epoch,
            ssh_to=None,
        )


def lcmvfilename(subject, session, rec, signal, epoch_type, chunk=None):
    try:
        makedirs(path)
    except:
        pass
    if chunk is None:
        filename = 'S%i-SESS%i-REC%i-%s-%s-lcmv_sf400.hdf' % (subject, session, rec, epoch_type, signal)
    else:
        filename = 'S%i-SESS%i-REC%i-%s-%s-chunk%i-lcmv_sf400.hdf' % (subject, session, rec, epoch_type, signal, chunk)
    return join(path, filename)


def get_stim_epoch(subject, session):
    if subject < 10:
        raw_filename = (
            '/home/abraun/meg_data/P0' + str(subject) + '/MEG/Raw/' + data_files['S' + str(subject)][session - 1]
        )
    elif subject >= 10:
        raw_filename = (
            '/home/abraun/meg_data/P' + str(subject) + '/MEG/Raw/' + data_files['S' + str(subject)][session - 1]
        )

    if raw_filename[-5:] == '01.ds':
        rec = 1
    elif raw_filename[-5:] == '02.ds':
        rec = 2

    if subject < 16:
        session = int(raw_filename[-34])
    elif 34 > subject >= 16:
        session = int(raw_filename[-31])
    elif subject >= 34:
        session = int(raw_filename[-34])

    if subject < 10:
        fif_filename = (
            '/home/abraun/meg_data/P0'
            + str(subject)
            + '/MEG/Locked/P0'
            + str(subject)
            + '-S'
            + str(session)
            + '_rec'
            + str(rec)
            + '_stim_new-epo.fif'
        )

    elif subject >= 10:
        fif_filename = (
            '/home/abraun/meg_data/P'
            + str(subject)
            + '/MEG/Locked/P'
            + str(subject)
            + '-S'
            + str(session)
            + '_rec'
            + str(rec)
            + '_stim_new-epo.fif'
        )

    epochs = mne.read_epochs(fif_filename)
    epochs = epochs.pick_channels([x for x in epochs.ch_names if x.startswith('M')])
    id_time = (-0.35 <= epochs.times) & (epochs.times <= -0.1)
    means = epochs._data[:, :, id_time].mean(-1)
    epochs._data -= means[:, :, np.newaxis]
    data_cov = pymeglcmv.get_cov(epochs, tmin=-0.35, tmax=1.25)
    return data_cov, epochs


def get_response_epoch(subject, session):
    if subject < 10:
        raw_filename = (
            '/home/abraun/meg_data/P0' + str(subject) + '/MEG/Raw/' + data_files['S' + str(subject)][session - 1]
        )
    elif subject >= 10:
        raw_filename = (
            '/home/abraun/meg_data/P' + str(subject) + '/MEG/Raw/' + data_files['S' + str(subject)][session - 1]
        )

    if raw_filename[-5:] == '01.ds':
        rec = 1
    elif raw_filename[-5:] == '02.ds':
        rec = 2

    if subject < 16:
        session = int(raw_filename[-34])
    elif 34 > subject >= 16:
        session = int(raw_filename[-31])
    elif subject >= 34:
        session = int(raw_filename[-34])

    if subject < 10:
        fif_filename = (
            '/home/abraun/meg_data/P0'
            + str(subject)
            + '/MEG/Locked/P0'
            + str(subject)
            + '-S'
            + str(session)
            + '_rec'
            + str(rec)
            + '_stim_new-epo.fif'
        )
        meta_filename = (
            '/home/abraun/meg_data/P0'
            + str(subject)
            + '/MEG/Locked/P0'
            + str(subject)
            + '-S'
            + str(session)
            + '_rec'
            + str(rec)
            + '_stim_new.hdf'
        )
        resp_fif_filename = (
            '/home/abraun/meg_data/P0'
            + str(subject)
            + '/MEG/Locked/P0'
            + str(subject)
            + '-S'
            + str(session)
            + '_rec'
            + str(rec)
            + '_resp_new-epo.fif'
        )

    elif subject >= 10:
        fif_filename = (
            '/home/abraun/meg_data/P'
            + str(subject)
            + '/MEG/Locked/P'
            + str(subject)
            + '-S'
            + str(session)
            + '_rec'
            + str(rec)
            + '_stim_new-epo.fif'
        )
        meta_filename = (
            '/home/abraun/meg_data/P'
            + str(subject)
            + '/MEG/Locked/P'
            + str(subject)
            + '-S'
            + str(session)
            + '_rec'
            + str(rec)
            + '_stim_new.hdf'
        )
        resp_fif_filename = (
            '/home/abraun/meg_data/P'
            + str(subject)
            + '/MEG/Locked/P'
            + str(subject)
            + '-S'
            + str(session)
            + '_rec'
            + str(rec)
            + '_resp_new-epo.fif'
        )

    epochs = mne.read_epochs(fif_filename)
    meta = pd.read_hdf(meta_filename)
    epochs = epochs.pick_channels([x for x in epochs.ch_names if x.startswith('M')])
    response = mne.read_epochs(resp_fif_filename)
    response = response.pick_channels([x for x in response.ch_names if x.startswith('M')])
    #   Find trials that are present in both time periods
    overlap = list(set(epochs.events[:, 2]).intersection(set(response.events[:, 2])))
    epochs = epochs[[str(l) for l in overlap]]
    response = response[[str(l) for l in overlap]]
    id_time = (-0.35 <= epochs.times) & (epochs.times <= -0.1)
    means = epochs._data[:, :, id_time].mean(-1)
    epochs._data -= means[:, :, np.newaxis]
    response._data = response._data - means[:, :, np.newaxis]

    # Now also baseline stimulus period
    if subject < 10:
        fif_filename = (
            '/home/abraun/meg_data/P0'
            + str(subject)
            + '/MEG/Locked/P0'
            + str(subject)
            + '-S'
            + str(session)
            + '_rec'
            + str(rec)
            + '_stim_new-epo.fif'
        )
        meta_filename = (
            '/home/abraun/meg_data/P0'
            + str(subject)
            + '/MEG/Locked/P0'
            + str(subject)
            + '-S'
            + str(session)
            + '_rec'
            + str(rec)
            + '_stim_new.hdf'
        )

    elif subject >= 10:
        fif_filename = (
            '/home/abraun/meg_data/P'
            + str(subject)
            + '/MEG/Locked/P'
            + str(subject)
            + '-S'
            + str(session)
            + '_rec'
            + str(rec)
            + '_stim_new-epo.fif'
        )
        meta_filename = (
            '/home/abraun/meg_data/P'
            + str(subject)
            + '/MEG/Locked/P'
            + str(subject)
            + '-S'
            + str(session)
            + '_rec'
            + str(rec)
            + '_stim_new.hdf'
        )

    epochs = mne.read_epochs(fif_filename)
    meta = pd.read_hdf(meta_filename)
    epochs = epochs.pick_channels([x for x in epochs.ch_names if x.startswith('M')])
    id_time = (-0.35 <= epochs.times) & (epochs.times <= -0.1)
    means = epochs._data[:, :, id_time].mean(-1)
    epochs._data -= means[:, :, np.newaxis]
    data_cov = pymeglcmv.get_cov(epochs, tmin=-0.35, tmax=1.25)
    return data_cov, response


def extract(
    subject, session, epoch_type='stimulus', signal_type='F', BEM='three_layer', debug=False, chunks=100, njobs=4
):
    mne.set_log_level('WARNING')
    pymeglcmv.logging.getLogger().setLevel(logging.INFO)
    set_n_threads(1)
    logging.info('Reading stimulus data')
    if epoch_type == 'stimulus':
        data_cov, epochs = get_stim_epoch(subject, session)

    elif epoch_type == 'response':
        data_cov, epochs = get_response_epoch(subject, session)
    else:
        raise RuntimeError('Did not recognize epoch')

    logging.info('Setting up source space and forward model')

    if subject < 10:
        raw_filename = (
            '/home/abraun/meg_data/P0' + str(subject) + '/MEG/Raw/' + data_files['S' + str(subject)][session - 1]
        )
    elif subject >= 10:
        raw_filename = (
            '/home/abraun/meg_data/P' + str(subject) + '/MEG/Raw/' + data_files['S' + str(subject)][session - 1]
        )
    if raw_filename[-5:] == '01.ds':
        rec = 1
    elif raw_filename[-5:] == '02.ds':
        rec = 2

    if subject < 16:
        session = int(raw_filename[-34])
    elif 34 > subject >= 16:
        session = int(raw_filename[-31])
    elif subject >= 34:
        session = int(raw_filename[-34])

    if subject < 10:
        if epoch_type == 'stimulus':
            epochs_filename = (
                '/home/abraun/meg_data/P0'
                + str(subject)
                + '/MEG/Locked/P0'
                + str(subject)
                + '-S'
                + str(session)
                + '_rec'
                + str(rec)
                + '_stim_new-epo.fif'
            )
        elif epoch_type == 'response':
            epochs_filename = (
                '/home/abraun/meg_data/P0'
                + str(subject)
                + '/MEG/Locked/P0'
                + str(subject)
                + '-S'
                + str(session)
                + '_rec'
                + str(rec)
                + '_resp_new-epo.fif'
            )
        trans_filename = (
            '/home/abraun/meg_data/P0'
            + str(subject)
            + '/MEG/P0'
            + str(subject)
            + '-S'
            + str(session)
            + '_rec'
            + str(rec)
            + '_trans.fif'
        )
        if subject == 9:
            forward, bem, source = sr.get_leadfield(
                "sj0" + str(subject),
                raw_filename,
                epochs_filename,
                trans_filename,
                conductivity=np.array([0.3]),
                njobs=4,
                bem_sub_path='bem_ft',
            )
        else:
            forward, bem, source = sr.get_leadfield(
                "sj0" + str(subject),
                raw_filename,
                epochs_filename,
                trans_filename,
                conductivity=(0.3, 0.006, 0.3),
                njobs=4,
                bem_sub_path='bem_ft',
            )

    elif subject >= 10:
        if epoch_type == 'stimulus':
            epochs_filename = (
                '/home/abraun/meg_data/P'
                + str(subject)
                + '/MEG/Locked/P'
                + str(subject)
                + '-S'
                + str(session)
                + '_rec'
                + str(rec)
                + '_stim_new-epo.fif'
            )
        elif epoch_type == 'response':
            epochs_filename = (
                '/home/abraun/meg_data/P'
                + str(subject)
                + '/MEG/Locked/P'
                + str(subject)
                + '-S'
                + str(session)
                + '_rec'
                + str(rec)
                + '_resp_new-epo.fif'
            )
        trans_filename = (
            '/home/abraun/meg_data/P'
            + str(subject)
            + '/MEG/P'
            + str(subject)
            + '-S'
            + str(session)
            + '_rec'
            + str(rec)
            + '_trans.fif'
        )
        if subject == 23 or subject == 31 or subject == 37:
            forward, bem, source = sr.get_leadfield(
                "sj" + str(subject),
                raw_filename,
                epochs_filename,
                trans_filename,
                conductivity=np.array([0.3]),
                njobs=4,
                bem_sub_path='bem_ft',
            )
        else:
            forward, bem, source = sr.get_leadfield(
                "sj" + str(subject),
                raw_filename,
                epochs_filename,
                trans_filename,
                conductivity=(0.3, 0.006, 0.3),
                njobs=4,
                bem_sub_path='bem_ft',
            )

    labels = sr.get_labels('sj%02i' % subject)  # Check that all labels are present

    labels = sr.labels_exclude(
        labels, exclude_filters=['wang2015atlas.IPS4', 'wang2015atlas.IPS5', 'wang2015atlas.SPL', 'JWG_lat_Unknown']
    )
    labels = sr.labels_remove_overlap(labels, priority_filters=['wang', 'JWG'])

    # Now chunk Reconstruction into blocks of ~100 trials to save Memory
    # fois = np.arange(36, 162, 4)
    # lfois = np.arange(2, 36, 2)
    # tfr_params = {
    #     'F': {'foi': fois, 'cycles': fois * 0.25, 'time_bandwidth': 6 + 1,
    #           'n_jobs': 1, 'est_val': fois, 'est_key': 'F', 'sf': 400,
    #           'decim': 10},
    #     'LF': {'foi': lfois, 'cycles': lfois * 0.5, 'time_bandwidth': 1 + 1,
    #            'n_jobs': 1, 'est_val': lfois, 'est_key': 'LF', 'sf': 400,
    #            'decim': 10}
    # }

    # fois = np.arange(36, 162, 4)
    # lfois = np.arange(2, 36, 2)
    # tfr_params = {
    #     'F': {'foi': fois, 'cycles': fois * 0.25, 'time_bandwidth': 6 + 1,
    #           'n_jobs': 1, 'est_val': fois, 'est_key': 'F', 'sf': 400,
    #           'decim': 10},
    #     'LF': {'foi': lfois, 'cycles': lfois * 0.5, 'time_bandwidth': 1 + 1,
    #            'n_jobs': 1, 'est_val': lfois, 'est_key': 'LF', 'sf': 400,
    #            'decim': 10}
    # }

    # fois = np.arange(42, 162, 4)
    # lfois = np.arange(2, 40, 2)
    fois = np.arange(37, 161, 4)
    lfois = np.arange(3, 37, 2)
    tfr_params = {
        'F': {
            'foi': fois,
            'cycles': fois * 0.4,
            'time_bandwidth': 8,
            'n_jobs': 1,
            'est_val': fois,
            'est_key': 'F',
            'sf': 400,
        },  # 7 taper, fsmooth = 20
        'LF': {
            'foi': lfois,
            'cycles': lfois * 0.4,
            'time_bandwidth': 2,  # 1 taper, fsmooth = 5
            'n_jobs': 1,
            'est_val': lfois,
            'est_key': 'LF',
            'sf': 400,
        },
    }  # fsmooth = time_bandwidth/time

    events = epochs.events[:, 2]
    data = []
    filters = pymeglcmv.setup_filters(epochs.info, forward, data_cov, None, labels)
    set_n_threads(1)

    for i in range(0, len(events), chunks):
        filename = lcmvfilename(subject, session, rec, signal_type, epoch_type, chunk=i)
        if os.path.isfile(filename):
            continue
        if signal_type == 'BB':
            logging.info('Starting reconstruction of BB signal')
            M = pymeglcmv.reconstruct_broadband(
                filters, epochs.info, epochs._data[i : i + chunks], events[i : i + chunks], epochs.times, njobs=1
            )
        else:
            logging.info('Starting reconstruction of TFR signal')
            M = pymeglcmv.reconstruct_tfr(
                filters,
                epochs.info,
                epochs._data[i : i + chunks],
                events[i : i + chunks],
                epochs.times,
                est_args=tfr_params[signal_type],
                njobs=4,
            )
        M.to_hdf(filename, 'epochs')
    set_n_threads(njobs)


sessions = range(1, 4)
subject = int(args[0])

for session in sessions:
    extract(
        subject, session, epoch_type='stimulus', signal_type='LF', BEM='three_layer', debug=False, chunks=100, njobs=4
    )
    extract(
        subject, session, epoch_type='response', signal_type='LF', BEM='three_layer', debug=False, chunks=100, njobs=4
    )
    extract(
        subject, session, epoch_type='stimulus', signal_type='F', BEM='three_layer', debug=False, chunks=100, njobs=4
    )
    extract(
        subject, session, epoch_type='response', signal_type='F', BEM='three_layer', debug=False, chunks=100, njobs=4
    )
