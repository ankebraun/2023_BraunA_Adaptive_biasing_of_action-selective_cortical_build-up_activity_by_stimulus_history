import logging
import mne
import numpy as np
import os
import glob
#from conf_meg import conf_analysis
import sys
sys.path.insert(0, '/mnt/homes/home024/abraun/pymeg')
sys.path.insert(0, "/mnt/homes/home024/abraun/conf_meg")
import conf_analysis
from pymeg import source_reconstruction as sr
from pymeg import lcmv as pymeglcmv
from joblib import Memory
import preprocessing
import pandas as pd
import pdb
import scipy.io as sio
from os import makedirs
from os.path import join
from optparse import OptionParser

usage = "compute_lcmv_filters.py [options] <sj_number>"

long_help = """
This script was adapted from Alessandro Toso.

Compute lcmv filter weights for Figure 2-Figure Supplement 1.
"""

parser = OptionParser ( usage, epilog=long_help )


opts,args = parser.parse_args ()

import pdb
from pymeg import atlas_glasser
all_clusters, _, _, _ = atlas_glasser.get_clusters()



memory = Memory(cachedir=os.environ['PYMEG_CACHE_DIR']) 
path = '/home/abraun/meg_data/sr_labeled/'

data_files = {'S1': ['MH-1_SequentialEffects_20160810_01.ds',
                      'MH-2_SequentialEffects_20160811_01.ds',
                      'MH-3_SequentialEffects_20160815_02.ds'],
              'S2': ['SE-1_SequentialEffects_20160822_01.ds',
			          'SE-2_SequentialEffects_20160907_01.ds',
					  'SE-3_SequentialEffects_20160921_01.ds'],
              'S3': ['JW-1_SequentialEffects_20160822_01.ds',
                      'JW-2_SequentialEffects_20160829_01.ds',
					  'JW-3_SequentialEffects_20160831_01.ds'],
              'S5': ['DL-1_SequentialEffects_20160829_02.ds',
			          'DL-2_SequentialEffects_20160907_01.ds',
					  'DL-3_SequentialEffects_20160908_01.ds'],
              'S7': ['CK-1_SequentialEffects_20160902_01.ds',
			          'CK-2_SequentialEffects_20161014_01.ds',
					  'CK-3_SequentialEffects_20161028_02.ds'],
              'S8': ['MS-1_SequentialEffects_20160902_01.ds',
                      'MS-2_SequentialEffects_20160905_01.ds',
					  'MS-3_SequentialEffects_20160907_01.ds'],
              'S9': ['AW-1_SequentialEffects_20160908_01.ds',
			          'AW-2_SequentialEffects_20160909_01.ds',
					  'AW-3_SequentialEffects_20161019_01.ds'],
              'S10': ['KJ-1_SequentialEffects_20160919_01.ds',
                      'KJ-2_SequentialEffects_20160921_01.ds',
					  'KJ-3_SequentialEffects_20161004_01.ds'], 
              'S11': ['AD-1_SequentialEffects_20160919_01.ds',
			          'AD-2_SequentialEffects_20160929_01.ds',
					  'AD-3_SequentialEffects_20161020_01.ds'],
              'S12': ['LF-1_SequentialEffects_20160928_01.ds'],
              'S13': ['VR-1_SequentialEffects_20161010_01.ds',
			          'VR-2_SequentialEffects_20161012_01.ds',
					  'VR-3_SequentialEffects_20161013_01.ds'],
              'S14': ['LA-1_SequentialEffects_20161010_01.ds',
                      'LA-2_SequentialEffects_20161011_01.ds',
                      'LA-3_SequentialEffects_20161012_01.ds'],
              'S15': ['MM-1_SequentialEffects_20161013_01.ds',
			          'MM-2_SequentialEffects_20161018_01.ds',
					  'MM-3_SequentialEffects_20161028_01.ds'],
              'S16': ['NK16-01_SequentEffects_20180222_01.ds',
			          'NK16-02_SequentEffects_20180308_01.ds',
					  'NK16-03_SequentEffects_20180309_01.ds'],
              'S17': ['AZ17-01_SequentEffects_20180226_01.ds',
			          'AZ17-02_SequentEffects_20180313_01.ds',
					  'AZ17-03_SequentEffects_20180319_01.ds'],
              'S18': ['GB18-01_SequentEffects_20180223_01.ds',
			          'GB18-02_SequentEffects_20180226_01.ds',
					  'GB18-03_SequentEffects_20180307_01.ds'],
		      'S19': ['KJ19-01_SequentEffects_20180308_01.ds',
			          'KJ19-02_SequentEffects_20180321_01.ds',
					  'KJ19-03_SequentEffects_20180322_01.ds'],
			  'S20': ['MB20-01_SequentEffects_20180227_01.ds',
			          'MB20-02_SequentEffects_20180308_01.ds',
					  'MB20-03_SequentEffects_20180327_01.ds'],
			  'S21': ['AK21-01_SequentEffects_20180222_01.ds',
			          'AK21-02_SequentEffects_20180308_01.ds',
					  'AK21-03_SequentEffects_20180424_01.ds'],
			  'S22': ['GA22-01_SequentEffects_20180219_01.ds',
                      'GA22-02_SequentEffects_20180222_01.ds',
					  'GA22-03_SequentEffects_20180226_01.ds'],
			  'S23': ['DB23-01_SequentEffects_20180219_01.ds',
			          'DB23-02_SequentEffects_20180220_01.ds',
					  'DB23-03_SequentEffects_20180221_01.ds'],
			  'S24': ['JR24-01_SequentEffects_20180313_01.ds',
			          'JR24-02_SequentEffects_20180327_01.ds',
					  'JR24-03_SequentEffects_20180405_01.ds'],
			  'S25': ['RB25-01_SequentEffects_20180307_01.ds',
			          'RB25-02_SequentEffects_20180309_01.ds',
					  'RB25-03_SequentEffects_20180313_01.ds'],
			  'S26': ['JF26-01_SequentEffects_20180406_01.ds',
			          'JF26-02_SequentEffects_20180410_01.ds',
					  'JF26-03_SequentEffects_20180413_01.ds'],
			  'S27': ['SM27-01_SequentEffects_20180406_01.ds',
			          'SM27-02_SequentEffects_20180410_01.ds',
					  'SM27-03_SequentEffects_20180412_01.ds'],
			  'S28': ['HN28-01_SequentEffects_20180423_01.ds',
			          'HN28-02_SequentEffects_20180424_01.ds',
					  'HN28-03_SequentEffects_20180427_01.ds'],
			  'S29': ['SS29-01_SequentEffects_20180406_01.ds',
			          'SS29-02_SequentEffects_20180423_01.ds',
					  'SS29-03_SequentEffects_20180424_01.ds'],				
			  'S30': ['JB30-01_SequentEffects_20180410_01.ds',
			          'JB30-02_SequentEffects_20180413_01.ds',
					  'JB30-03_SequentEffects_20180427_01.ds'],
			  'S31': ['CL31-01_SequentEffects_20180405_01.ds',
			          'CL31-02_SequentEffects_20180406_01.ds',
					  'CL31-03_SequentEffects_20180503_01.ds'],
			  'S32': ['JR32-01_SequentEffects_20180412_01.ds',
			          'JR32-02_SequentEffects_20180413_01.ds',
					  'JR32-03_SequentEffects_20180424_01.ds'],
			  'S33': ['TS33-01_SequentEffects_20180409_01.ds',
			          'TS33-02_SequentEffects_20180410_01.ds',
					  'TS33-03_SequentEffects_20180413_01.ds'],
                      
              'S34': ['ML-34-1_SequentialEffects_20191021_01.ds',
                                  'ML-34-2_SequentialEffects_20191028_01.ds',
                                  'ML-34-3_SequentialEffects_20191104_01.ds'],
                                  
              'S35': ['ML-35-1_SequentialEffects_20191210_01.ds',
                                  'ML-35-2_SequentialEffects_20191212_01.ds',
                                  'ML-35-3_SequentialEffects_20200116_01.ds'],
                                  
              'S36': ['GL-36-1_SequentialEffects_20191118_01.ds',
                                  'GL-36-2_SequentialEffects_20191203_01.ds',
                                  'GL-36-3_SequentialEffects_20191205_01.ds'],
                                  
              'S37': ['OE-37-1_SequentialEffects_20191205_01.ds',
                                  'OE-37-2_SequentialEffects_20191211_01.ds',
                                  'OE-37-3_SequentialEffects_20191212_01.ds'],
                                  
              'S38': ['IA-38-1_SequentialEffects_20191125_01.ds',
                                  'IA-38-2_SequentialEffects_20191129_01.ds',
                                  'IA-38-3_SequentialEffects_20191203_01.ds'],
                                  
              'S39': ['IF-39-1_SequentialEffects_20191202_01.ds',
                                  'IF-39-2_SequentialEffects_20191203_01.ds',
                                  'IF-39-3_SequentialEffects_20191204_01.ds'],
                                  
              'S40': ['ZB-40-1_SequentialEffects_20191202_01.ds',
                                  'ZB-40-2_SequentialEffects_20191203_01.ds',
                                  'ZB-40-3_SequentialEffects_20191204_01.ds'],
                                  
              'S41': ['UT-41-1_SequentialEffects_20191209_01.ds',
                                  'UT-41-2_SequentialEffects_20191210_01.ds',
                                  'UT-41-3_SequentialEffects_20191212_01.ds'],
                                  
              'S42': ['QH-42-1_SequentialEffects_20191216_01.ds',
                                  'QH-42-2_SequentialEffects_20191218_01.ds',
                                  'QH-42-3_SequentialEffects_20191219_01.ds']                                                                                                                                                                                                                                              
              }


def get_all_glasser_clusters(side,Side):   
    labels_l = {'vfcPrimary': ['lh.wang2015atlas.V1d-lh', 'lh.wang2015atlas.V1v-lh'], 
                  'vfcEarly': ['lh.wang2015atlas.V2d-lh', 'lh.wang2015atlas.V2v-lh', 'lh.wang2015atlas.V3d-lh', 'lh.wang2015atlas.V3v-lh', 'lh.wang2015atlas.hV4-lh'],
                  'vfcVO': ['lh.wang2015atlas.VO1-lh','lh.wang2015atlas.VO2-lh'],
                  'vfcPHC': ['lh.wang2015atlas.PHC1-lh', 'lh.wang2015atlas.PHC2-lh'],
                  'vfcV3ab': ['lh.wang2015atlas.V3A-lh', 'lh.wang2015atlas.V3B-lh'],
                  'vfcTO': ['lh.wang2015atlas.TO1-lh', 'lh.wang2015atlas.TO2-lh'], 
                  'vfcIPS01': ['lh.wang2015atlas.IPS0-lh', 'lh.wang2015atlas.IPS1-lh'],
                  'vfcIPS23': ['lh.wang2015atlas.IPS2-lh', 'lh.wang2015atlas.IPS3-lh'],
                  'vfcLO': ['lh.wang2015atlas.LO1-lh', 'lh.wang2015atlas.LO2-lh'],
                  'vfcPHC': ['lh.wang2015atlas.PHC1-lh', 'lh.wang2015atlas.PHC2-lh'],
                  'JWG_aIPS': ['lh.JWDG.lr_aIPS1-lh'],
                  'JWG_IPS_PCeS': ['lh.JWDG.lr_IPS_PCes-lh'],
                  'JWG_M1': ['lh.JWDG.lr_M1-lh']}
                
    labels_r = {'vfcPrimary': ['rh.wang2015atlas.V1d-rh', 'rh.wang2015atlas.V1v-rh'], 
                  'vfcEarly': ['rh.wang2015atlas.V2d-rh', 'rh.wang2015atlas.V2v-rh', 'rh.wang2015atlas.V3d-rh', 'rh.wang2015atlas.V3v-rh', 'rh.wang2015atlas.hV4-rh'],
                  'vfcVO': ['rh.wang2015atlas.VO1-rh','rh.wang2015atlas.VO2-rh'],
                  'vfcPHC': ['rh.wang2015atlas.PHC1-rh', 'rh.wang2015atlas.PHC2-rh'],
                  'vfcV3ab': ['rh.wang2015atlas.V3A-rh', 'rh.wang2015atlas.V3B-rh'],
                  'vfcTO': ['rh.wang2015atlas.TO1-rh', 'rh.wang2015atlas.TO2-rh'], 
                  'vfcIPS01': ['rh.wang2015atlas.IPS0-rh', 'rh.wang2015atlas.IPS1-rh'],
                  'vfcIPS23': ['rh.wang2015atlas.IPS2-rh', 'rh.wang2015atlas.IPS3-rh'],
                  'vfcLO': ['rh.wang2015atlas.LO1-rh', 'rh.wang2015atlas.LO2-rh'],
                  'vfcPHC': ['rh.wang2015atlas.PHC1-rh', 'rh.wang2015atlas.PHC2-rh'],
                  'JWG_aIPS': ['rh.JWDG.lr_aIPS1-rh'],
                  'JWG_IPS_PCeS': ['rh.JWDG.lr_IPS_PCeS-rh'],
                  'JWG_M1': ['rh.JWDG.lr_M1-rh']}
              
             
    if side == 'l':
        areas_to_labels = labels_l
    elif side == 'r':
        areas_to_labels = labels_r
    return areas_to_labels
    
def set_n_threads(n):
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = str(n)
    os.environ['MKL_NUM_THREADS'] = str(n)
    os.environ['OMP_NUM_THREADS'] = str(n)


def submit():
    from pymeg import parallel
    from itertools import product
    for subject, session, epoch, signal in product(range(3,11), range(1, 4), ['stimulus'],
            ['F']):
        parallel.pmap(
            extract, [(subject, session, epoch, signal)],
            walltime=10, memory=40, nodes=1, tasks=4,
            name='P0' + str(subject) + '-S' + str(session) + '_rec1_' + epoch,
            ssh_to=None)


def lcmvfilename(subject, session, rec, signal, epoch_type, chunk=None):
    try:
        makedirs(path)
    except:
        pass
    if chunk is None:
        filename = 'S%i-SESS%i-REC%i-%s-%s-lcmv_sf400.hdf' % (
            subject, session, rec, epoch_type, signal)
    else:
        filename = 'S%i-SESS%i-REC%i-%s-%s-chunk%i-lcmv_sf400.hdf' % (
            subject, session, rec, epoch_type, signal, chunk)
    return join(path, filename)


def get_stim_epoch(subject, session):
    if subject < 10:
        raw_filename = '/home/abraun/meg_data/P0' + str(subject) + '/MEG/Raw/' + data_files['S' + str(subject)][session - 1]
    elif subject >= 10:
        raw_filename = '/home/abraun/meg_data/P' + str(subject) + '/MEG/Raw/' + data_files['S' + str(subject)][session - 1]  
          
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
        fif_filename = '/home/abraun/meg_data/P0' + str(subject) + '/MEG/Locked/P0' + str(subject) + '-S' + str(session) + '_rec' + str(rec) + '_stim_new-epo.fif'

    elif subject >= 10:
        fif_filename = '/home/abraun/meg_data/P' + str(subject) + '/MEG/Locked/P' + str(subject) + '-S' + str(session) + '_rec' + str(rec) + '_stim_new-epo.fif'

    epochs = mne.read_epochs(fif_filename)
    print(fif_filename)

    epochs = epochs.pick_channels([x for x in epochs.ch_names if x.startswith('M')])

    id_time = (-0.35 <= epochs.times) & (epochs.times <= -0.1)
    means = epochs._data[:, :, id_time].mean(-1)
    epochs._data -= means[:, :, np.newaxis]
    data_cov = pymeglcmv.get_cov(epochs, tmin=-0.35, tmax=1.25)
    return data_cov, epochs


def get_response_epoch(subject, session):
    if subject < 10:
        raw_filename = '/home/abraun/meg_data/P0' + str(subject) + '/MEG/Raw/' + data_files['S' + str(subject)][session - 1]
    elif subject >= 10:
        raw_filename = '/home/abraun/meg_data/P' + str(subject) + '/MEG/Raw/' + data_files['S' + str(subject)][session - 1]  

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
        fif_filename = '/home/abraun/meg_data/P0' + str(subject) + '/MEG/Locked/P0' + str(subject) + '-S' + str(session) + '_rec' + str(rec) + '_stim_new-epo.fif'
        meta_filename = '/home/abraun/meg_data/P0' + str(subject) + '/MEG/Locked/P0' + str(subject) + '-S' + str(session) + '_rec' + str(rec) + '_stim_new.hdf'
        resp_fif_filename = '/home/abraun/meg_data/P0' + str(subject) + '/MEG/Locked/P0' + str(subject) + '-S' + str(session) + '_rec' + str(rec) + '_resp_new-epo.fif'

    elif subject >= 10:   
        fif_filename = '/home/abraun/meg_data/P' + str(subject) + '/MEG/Locked/P' + str(subject) + '-S' + str(session) + '_rec' + str(rec) + '_stim_new-epo.fif'
        meta_filename = '/home/abraun/meg_data/P' + str(subject) + '/MEG/Locked/P' + str(subject) + '-S' + str(session) + '_rec' + str(rec) + '_stim_new.hdf'
        resp_fif_filename = '/home/abraun/meg_data/P' + str(subject) + '/MEG/Locked/P' + str(subject) + '-S' + str(session) + '_rec' + str(rec) + '_resp_new-epo.fif'


    epochs = mne.read_epochs(fif_filename)

    meta = pd.read_hdf(meta_filename)

    epochs = epochs.pick_channels(
        [x for x in epochs.ch_names if x.startswith('M')])
            
    response = mne.read_epochs(resp_fif_filename)
    response = response.pick_channels(
        [x for x in response.ch_names if x.startswith('M')])

 #   Find trials that are present in both time periods
    overlap = list(
        set(epochs.events[:, 2]).intersection(
            set(response.events[:, 2])))
    
    epochs = epochs[[str(l) for l in overlap]]
    response = response[[str(l) for l in overlap]]
    id_time = (-0.35 <= epochs.times) & (epochs.times <= -0.1)    
    means = epochs._data[:, :, id_time].mean(-1)
    epochs._data -= means[:, :, np.newaxis]

    response._data = (
        response._data - means[:, :, np.newaxis])

    # Now also baseline stimulus period
    if subject < 10:       
        fif_filename = '/home/abraun/meg_data/P0' + str(subject) + '/MEG/Locked/P0' + str(subject) + '-S' + str(session) + '_rec' + str(rec) + '_stim_new-epo.fif'
        meta_filename = '/home/abraun/meg_data/P0' + str(subject) + '/MEG/Locked/P0' + str(subject) + '-S' + str(session) + '_rec' + str(rec) + '_stim_new.hdf'

    elif subject >= 10:
            fif_filename = '/home/abraun/meg_data/P' + str(subject) + '/MEG/Locked/P' + str(subject) + '-S' + str(session) + '_rec' + str(rec) + '_stim_new-epo.fif'
            meta_filename = '/home/abraun/meg_data/P' + str(subject) + '/MEG/Locked/P' + str(subject) + '-S' + str(session) + '_rec' + str(rec) + '_stim_new.hdf'

    epochs = mne.read_epochs(fif_filename)

    meta = pd.read_hdf(meta_filename)
    epochs = epochs.pick_channels(
        [x for x in epochs.ch_names if x.startswith('M')])
    id_time = (-0.35 <= epochs.times) & (epochs.times <= -0.1)        
    means = epochs._data[:, :, id_time].mean(-1)
    epochs._data -= means[:, :, np.newaxis]
    data_cov = pymeglcmv.get_cov(epochs, tmin=-0.35, tmax=1.25)
    return data_cov, response


def extract(subject, session, area, side, epoch_type='stimulus', signal_type='F',
            BEM='three_layer', debug=False, chunks=100, njobs=4):
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
        raw_filename = '/home/abraun/meg_data/P0' + str(subject) + '/MEG/Raw/' + data_files['S' + str(subject)][session - 1]
    elif subject >= 10:
        raw_filename = '/home/abraun/meg_data/P' + str(subject) + '/MEG/Raw/' + data_files['S' + str(subject)][session - 1]  
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
            epochs_filename = '/home/abraun/meg_data/P0' + str(subject) + '/MEG/Locked/P0' + str(subject) + '-S' + str(session) + '_rec' + str(rec) + '_stim_new-epo.fif'
        elif epoch_type == 'response':    
            epochs_filename = '/home/abraun/meg_data/P0' + str(subject) + '/MEG/Locked/P0' + str(subject) + '-S' + str(session) + '_rec' + str(rec) + '_resp_new-epo.fif'
        trans_filename = '/home/abraun/meg_data/P0' + str(subject) + '/MEG/P0' + str(subject) + '-S' + str(session) + '_rec' + str(rec) + '_trans.fif'
        if subject == 9:
            forward, bem, source = sr.get_leadfield("sj0" + str(subject), raw_filename, epochs_filename, trans_filename, conductivity=np.array([0.3]), njobs = 4, bem_sub_path = 'bem_ft')
        else:
            forward, bem, source = sr.get_leadfield("sj0" + str(subject), raw_filename, epochs_filename, trans_filename, conductivity=(0.3, 0.006, 0.3), njobs = 4, bem_sub_path = 'bem_ft')

    elif subject >= 10:
        if epoch_type == 'stimulus':    
            epochs_filename = '/home/abraun/meg_data/P' + str(subject) + '/MEG/Locked/P' + str(subject) + '-S' + str(session) + '_rec' + str(rec) + '_stim_new-epo.fif'
        elif epoch_type == 'response':    
            epochs_filename = '/home/abraun/meg_data/P' + str(subject) + '/MEG/Locked/P' + str(subject) + '-S' + str(session) + '_rec' + str(rec) + '_resp_new-epo.fif'  
        trans_filename = '/home/abraun/meg_data/P' + str(subject) + '/MEG/P' + str(subject) + '-S' + str(session) + '_rec' + str(rec) + '_trans.fif'
        if subject == 11 or subject == 23 or subject == 31 or subject == 37:
            forward, bem, source = sr.get_leadfield("sj" + str(subject), raw_filename, epochs_filename, trans_filename, conductivity=np.array([0.3]), njobs = 4, bem_sub_path = 'bem_ft')
        else:
            forward, bem, source = sr.get_leadfield("sj" + str(subject), raw_filename, epochs_filename, trans_filename, conductivity=(0.3, 0.006, 0.3), njobs = 4, bem_sub_path = 'bem_ft')
 
    if  str(side) in ['l']:
       # save distance between sources in this hemifield
        df = pd.DataFrame(source[0]['rr'])
        df.to_csv('source_vx_dists_sj_' + str(subject) + '_sess_' + str(session) + '_L.csv', sep = '\t' )

        # save  vertex id for each source in this hemifield
        df = pd.DataFrame(source[0]['vertno'])
        df.to_csv('source_vx_id_sj_' + str(subject) + '_sess_' + str(session) + '_L.csv', sep = '\t' )

    elif  str(side) in ['r']:

        # save distance between sources in this hemifield
        df = pd.DataFrame(source[1]['rr'])
        df.to_csv('source_vx_dists_sj_' + str(subject) + '_sess_' + str(session) + '_R.csv', sep = '\t' )

        # save  vertex id for each source in this hemifield
        df = pd.DataFrame(source[1]['vertno'])
        df.to_csv('source_vx_id_sj_' + str(subject) + '_sess_' + str(session) + '_R.csv', sep = '\t' )

   
    # save leadfield matrix
    lead=forward['sol']
    fname = str(subject) + '_sess_' + str(session) +'_leadfield.mat'
    sio.savemat(fname, lead)


    labels = sr.get_labels('sj%02i' % subject) # Check that all labels are present
    labels = sr.labels_exclude(labels,
    exclude_filters=['wang2015atlas.IPS4',
    'wang2015atlas.IPS5',
    'wang2015atlas.SPL',
    'JWG_lat_Unknown'])
    labels = sr.labels_remove_overlap(
    labels, priority_filters=['wang', 'JWG'],)
    labels_new = []
    for x in labels:
        for cl in areas_to_labels[area]:
            if cl in x.name:
                labels_new.append(x)
    labels = labels_new
    print(labels)
    print(area)
    label = labels.pop()
    for l in labels:
        label += l
    print('Selecting this label for area %s:' % area, label)

    filters = []
    filters.append(pymeglcmv.setup_filters(epochs.info, forward, data_cov, None, [label]))
    subsmp = 400
    print(label)

    
    filters_this_area = pymeglcmv.setup_filters(epochs.info, forward, data_cov,
                                  None, [label])

    filters_all_source=mne.beamformer.make_lcmv(
        epochs.info, forward, data_cov,
        noise_cov=None, label=None, reg=0.05,
        pick_ori='max-power')
  
    this_area=areas_to_labels[area]
    
    this_area = ''.join(this_area)
    
    # extract lcmv filter weights for all sources
    filter_weights_all=(filters_all_source['weights'])
    df = pd.DataFrame(filter_weights_all)
    df.to_csv('All_source_filter_weights_sj_' + str(subject)+ '_sess_' + str(session) + '.csv', sep = '\t' )
    # extract vertex id for this label area
    def getList(dict):
        list = []
        for key in dict.keys():
            list.append(key)
         
        return list
    this_area = getList(filters_this_area)   
   # this_area = tuple(this_area)
    filter_vx_area=(filters_this_area[this_area[0]]['vertices'])
    df = pd.DataFrame(filter_vx_area)
    df.to_csv('id_VX__sj_' + str(subject)+ '_sess_' + str(session) + '_area_' + str(area) +'_'+ str(side)+'.csv', sep = '\t' )

sessions = range(1,3)
subject = int(args[0])

for session in sessions:   
    area = ['vfcPrimary', 'vfcEarly', 'vfcVO', 'vfcPHC', 'vfcV3ab', 'vfcTO', 'vfcIPS01', 'vfcIPS23', 'vfcLO', 'vfcPHC', 'JWG_aIPS', 'JWG_M1', 'JWG_IPS_PCeS']
    all_sides = [{'l': 'L' }, {'r': 'R' }]  
    for sides in all_sides: 
        for side,Side in sides.items():
            areas_to_labels = get_all_glasser_clusters(side,Side)
            for area_extract in area:
                print(area_extract)
                extract(subject, session, area_extract, side, epoch_type='stimulus', signal_type='LF',
                BEM='three_layer', debug=False, chunks=100, njobs=4)

