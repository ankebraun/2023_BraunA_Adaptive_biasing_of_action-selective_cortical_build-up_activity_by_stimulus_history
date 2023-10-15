"""
This script was adapted from Niklas Wilming et al. Nature Communications 2020
https://github.com/DonnerLab/2020_Large-scale-Dynamics-of-Perceptual-Decision-Information-across-Human-Cortex
"""
import os
import pandas as pd
from glob import glob
import numpy as np
from joblib import Parallel, delayed
from joblib import Memory
import logging
import pdb
from pymeg import atlas_glasser

memory = Memory(location=os.environ['PYMEG_CACHE_DIR'], verbose=0)

backend = 'loky'


class Cache(object):
    """A cache that can prevent reloading from disk.

    Can be used as a context manager.
    """

    def __init__(self, cache=True):
        self.store = {}
        self.cache = cache

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.clear()

    def get(self, globstring):
        if self.cache:
            if globstring not in self.store:
                self.store[globstring] = self._load_tfr_data(globstring)
            else:
                logging.info('Returning cached object: %s' % globstring)
            return self.store[globstring]
        else:
            return self._load_tfr_data(globstring)

    def clear(self):
        self.cache = {}

    def _load_tfr_data(self, globstring):
        """Load all files identified by glob string"""
        logging.info('Loading data for: %s' % globstring)
        tfr_data_filenames = glob(globstring)
        logging.info('This is these filenames: %s' % str(tfr_data_filenames))
        tfrs = []
        for f in tfr_data_filenames:
            logging.info('Now working on: %s' % str(f))
            tfr = pd.read_hdf(f)
            logging.info('Done loading, now pivoting.')
            tfr = pd.pivot_table(tfr.reset_index(), values=tfr.columns, index=[
                                 'trial', 'est_val'], columns='time').stack(-2)
            tfr.index.names = ['trial', 'freq', 'area']
            tfrs.append(tfr)
        logging.info('Concate now.')
        tfr = pd.concat(tfrs)
        logging.info('Done _load_tfr_data.')
        return tfr


def baseline_per_sensor_get(tfr, baseline_time=(-0.35, -0.1)):
    '''
    Get average baseline
    '''
    time = tfr.columns.get_level_values('time').values.astype(float)
    id_base = (time >= baseline_time[0]) & (time <= baseline_time[1])
    base = tfr.loc[:, id_base].groupby(['freq', 'area']).mean().mean(
        axis=1)  # This should be len(nr_freqs * nr_channels)
    return base


def baseline_per_sensor_apply(tfr, baseline):
    '''
    Baseline correction by dividing by average baseline
    '''
    def div(x):
        freqs = x.index.get_level_values('freq').values[0]
        areas = x.index.get_level_values('area').values[0]
        print("baseline.index.isin([freqs], level='freq')")
        print(baseline.index.isin([freqs], level='freq'))
        print("baseline.index.isin([areas], level='area')")
        print(baseline.index.isin([areas], level='area'))
        bval = float(baseline
                     .loc[
                         baseline.index.isin([freqs], level='freq')
                         & baseline.index.isin([areas], level='area')])
        return (x - bval) / bval * 100
    return tfr.groupby(['freq', 'area']).apply(div)


@memory.cache(ignore=['cache'])
def load_tfr_contrast(data_globstring, base_globstring, meta_data, conditions,
                      baseline_time, n_jobs=1, baseline_per_condition=False,
                      cache=Cache(cache=False)):
    """Load a set of data files and turn them into contrasts.
    """
    tfrs = []
    # load data:
    tfr_data = cache.get(data_globstring)
    print(data_globstring)
    print(tfr_data)
    # Make sure that meta_data and tfr_data overlap in trials
    tfr_trials = np.unique(tfr_data.index.get_level_values('trial').values)
    meta_trials = np.unique(meta_data.reset_index().loc[:, 'idx'].values)
    assert(any([t in meta_trials for t in tfr_trials]))
    # data to baseline:
    if not (data_globstring == base_globstring):
        tfr_data_to_baseline = cache.get(base_globstring)
    else:
        tfr_data_to_baseline = tfr_data

    # if baseline_per_condition:
    #     # apply condition ind, collapse across trials, and get baseline::
    #     tfr_data_to_baseline = tfr_data_to_baseline.groupby(
    #         ['freq', 'area']).mean()
    # compute contrasts
    tasks = []
    for condition in conditions:
        tasks.append((tfr_data, tfr_data_to_baseline, meta_data,
                      condition, baseline_time, baseline_per_condition))

    tfr_conditions = Parallel(n_jobs=n_jobs, verbose=1, backend=backend)(
        delayed(make_tfr_contrasts)(*task) for task in tasks)
    print(tfr_conditions)
    weight_dicts = [t[1] for t in tfr_conditions]
    weights = weight_dicts.pop()
    [weights.update(w) for w in weight_dicts]
    # weights = {(k, v) for k, v in [t[1] for t in tfr_conditions]}
    tfrs.append(pd.concat([t[0] for t in tfr_conditions if t[0] is not None]))
    tfrs = pd.concat(tfrs)
    return tfrs, weights


def make_tfr_contrasts(tfr_data, tfr_data_to_baseline, meta_data,
                       condition, baseline_time, baseline_per_condition=False):

    # unpack:
    condition_ind = meta_data.loc[meta_data[condition] == 1, "idx"]
    print(condition)
    print(condition_ind)
    if baseline_per_condition:
        # apply condition ind, collapse across trials, and get baseline::
        tfr_data_to_baseline = (tfr_data_to_baseline.loc[
            tfr_data_to_baseline.index.isin(condition_ind, level='trial'), :]
            .groupby(['freq', 'area']).mean())

    baseline = baseline_per_sensor_get(
        tfr_data_to_baseline, baseline_time=baseline_time)

    # apply condition ind, and collapse across trials:
    tfr_data_condition = (tfr_data.loc[
        tfr_data.index.isin(condition_ind, level='trial'), :])
    num_trials_in_condition = len(np.unique(
        tfr_data_condition.index.get_level_values('trial')))

    if num_trials_in_condition == 0:
        return None, {condition: num_trials_in_condition}
    tfr_data_condition = tfr_data_condition.groupby(['freq', 'area']).mean()
    #pdb.set_trace()
    # apply baseline, and collapse across sensors:
    tfr_data_condition = baseline_per_sensor_apply(
        tfr_data_condition, baseline=baseline).groupby(['freq', 'area']).mean()

    tfr_data_condition['condition'] = condition
    tfr_data_condition = tfr_data_condition.set_index(
        ['condition', ], append=True, inplace=False)
    tfr_data_condition = tfr_data_condition.reorder_levels(
        ['area', 'condition', 'freq'])
  #  pdb.set_trace()
    return tfr_data_condition, {condition: num_trials_in_condition}


@memory.cache(ignore=['cache'])
def single_conditions(conditions, data_glob, base_glob, meta_data,
                      baseline_time, baseline_per_condition=False,
                      n_jobs=1, cache=Cache(cache=False)):

    tfr_condition, weights = load_tfr_contrast(
        data_glob, base_glob, meta_data,
        list(conditions), baseline_time, n_jobs=n_jobs,
        baseline_per_condition=baseline_per_condition,
        cache=cache)
    return tfr_condition.groupby(
        ['area', 'condition', 'freq']).mean(), weights


@memory.cache(ignore=['cache'])
def pool_conditions(conditions, data_globs, base_globs, meta_data,
                    baseline_time, baseline_per_condition=False,
                    n_jobs=1, cache=Cache(cache=False)):
    weights = {}
    tfrs = {}
    for i, (data_glob, base_glob) in enumerate(
            zip(ensure_iter(data_globs), ensure_iter(base_globs))):
        print(i)
        print(data_glob)
        print(base_glob)
   #     pdb.set_trace()
        # tfr, weight = single_conditions(
        #    conditions, data_glob, base_glob, meta_data, baseline_time,
        #    n_jobs=n_jobs,
        #    cache=cache)
        tfr, weight = load_tfr_contrast(
            data_glob, base_glob, meta_data,
            list(conditions), baseline_time, n_jobs=n_jobs,
            baseline_per_condition=baseline_per_condition,
            cache=cache)
        print('tfr', tfr)
        print('weight', weight)  
        tfrs[i] = tfr
        weights[i] = weight
        # Compute total trials per condition
    total_weights = {}
    for i, w in weights.items():
        for k, v in w.items():
            if k not in total_weights:
                total_weights[k] = v
            else:
                total_weights[k] += v
    # Apply weights to each tfr
    ind_weights = {}
    for k in total_weights.keys():
        ind_weights[k] = []
    for key in tfrs.keys():
        tfr = tfrs[key]
        for condition in total_weights.keys():
            condition_ind = tfr.index.get_level_values(
                'condition') == condition
            if sum(condition_ind) == 0:
                continue
            w = weights[key][condition] / total_weights[condition]
            tfr.loc[condition_ind, :] *= w
            ind_weights[condition].append(w)
        tfrs[key] = tfr
    for condition, weights in ind_weights.items():
        logging.info("weights for %s -> %s, sum=%f" %
                     (condition, str(weights), sum(weights)))
   # pdb.set_trace()
    tfrs = pd.concat(tfrs.values()).groupby(
        ['area', 'condition', 'freq']).sum()
    return tfrs


@memory.cache(ignore=['cache'])
def compute_contrast(contrasts, hemis, data_globstring, base_globstring,
                     meta_data, baseline_time, baseline_per_condition=False,
                     n_jobs=1, cache=Cache(cache=False)):
    """Compute a single contrast from tfr data
    Args:
        contrast: dict
            Contains contrast names as keys and len==2 tuples as values. The
            tuples contain a list of condition names first and then a set of
            weights for each condition. Condition names identify columns in
            the meta data that are one for each trial that belongs to
            this condition.
        hemi: str
            Can be:
                'lh_is_ipsi' if contrast is ipsi-contra hemi and left hemi is
                    ipsi.
                'rh_is_ipsi' if contrast is ipis-contra and right hemi is ipsi
                'avg' if contrast should be averaged across hemispheres
        data_globstring: list
            Each string in data_globstring selects a set of filenames if
            passed through glob. Condition averages and baselines are then
            computed for each group of filenames identified by one entry
            in data_globstring. This is useful for, e.g. computing
            conditions per session first, then averaging them and then
            computing contrasts across sessions.
        base_globstring: string or list
            Same as data_globstring but selects data to use for baselining
        meta_data: data frame
            Meta data DataFrame with as many rows as trials.
        baseline_time: tuple

    """

    # load for all subjects:
    tfr_condition = []
    from functools import reduce
    from itertools import product

    conditions = contrasts
    conditions = set(
        reduce(lambda x, y: x + y, [x[0] for x in contrasts.values()]))

    # tfr_condition = pool_conditions(conditions, data_globstring,
    #                                 base_globstring, meta_data,
    #                                 baseline_time, n_jobs=n_jobs,
    #                                 cache=cache)
                                    
    print(conditions)        
    print(data_globstring) 
    print(base_globstring) 
    print(meta_data)    
    print(baseline_time)  
    print(baseline_per_condition)    
    print(n_jobs)        
    print(cache)
    tfr_condition = pool_conditions(conditions, data_globstring, 
                                    base_globstring, meta_data,
                                    baseline_time, 
                                    baseline_per_condition,
                                    n_jobs, cache)

    # Lower case all area names
    # FIXME: Set all area names to lower case!
   # pdb.set_trace()
    all_clusters, _, _, _ = atlas_glasser.get_clusters()
    tfr_areas = np.array([a for a in tfr_condition.index.levels[
        np.where(np.array(tfr_condition.index.names) == 'area')[0][0]]])
    tfr_areas_lower = np.array([area.lower() for area in tfr_areas])
    for cluster, areas in all_clusters.items():
        new_areas = []
        for area in areas:
            idx = np.where(tfr_areas_lower == area.lower())[0]
            if len(idx) == 1:
                new_areas.append(tfr_areas[idx[0]])
        all_clusters[cluster] = new_areas
    print(tfr_condition.groupby(
        ['area', 'condition', 'freq']))
    # mean across sessions:
    tfr_condition = tfr_condition.groupby(
        ['area', 'condition', 'freq']).mean()
    cluster_contrasts = []
   # pdb.set_trace()
    for cur_contrast, hemi, cluster in product(contrasts.items(), hemis,
                                               all_clusters.keys()):
        contrast, (conditions, weights) = cur_contrast
        logging.info('Start computing contrast %s for cluster %s' %
                     (contrast, cluster))
        right = []
        left = []
        for condition in conditions:
            tfrs_rh = []
            tfrs_lh = []
            for area in all_clusters[cluster]:
                area_idx = tfr_condition.index.isin([area], level='area')
                condition_idx = tfr_condition.index.isin(
                    [condition], level='condition')
                subset = tfr_condition.loc[area_idx & condition_idx].groupby(
                    ['freq']).mean()
                if 'rh' in area:
                    tfrs_rh.append(subset)
                else:
                    tfrs_lh.append(subset)
            # What happens when an area is not defined for both hemis?
            if (len(tfrs_lh) == 0) and (len(tfrs_rh) == 0):
                logging.warn('Skipping condition %s in cluster %s' %
                             (condition, cluster))
                continue
            try:
                left.append(pd.concat(tfrs_lh))
            except ValueError:
                pass
            try:
                right.append(pd.concat(tfrs_rh))
            except ValueError:
                pass
            print ('condition', condition)
       #     pdb.set_trace()
        if (len(left) == 0) and (len(right) == 0):
            logging.warn('Skipping cluster %s' % (cluster))
            continue
        if hemi == 'rh_is_ipsi':
            left, right = right, left
        if 'is_ipsi' in hemi:
            if not len(left) == len(right):
                logging.warn('Skipping cluster %s: does not have the same number of lh/rh rois' %
                             (cluster))
                continue
            tfrs = [left[i] - right[i]
                    for i in range(len(left))]
        else:
            if (len(right) == 0) and (len(left) == len(weights)):
                tfrs = left
            elif (len(left) == 0) and (len(right) == len(weights)):
                tfrs = right
            else:
                tfrs = [(right[i] + left[i]) / 2
                        for i in range(len(left))]
        assert(len(tfrs) == len(weights))
       # pdb.set_trace()
        tfrs = [tfr * weight for tfr, weight in zip(tfrs, weights)]
        # tfrs_ = [tfr * weight for tfr, weight in zip(tfrs, (-1, 1))]
        # tfrs__ = pd.concat(tfrs_) 
        #  tfrs__.groupby(['freq']).mean()
        # tfrs_[0].groupby(['freq']).mean()
       # pdb.set_trace()
        tfrs = reduce(lambda x, y: x + y, tfrs)
      #  pdb.set_trace()
        tfrs = tfrs.groupby('freq').mean()
        print('tfrs', tfrs)
        tfrs.loc[:, 'cluster'] = cluster
        tfrs.loc[:, 'contrast'] = contrast
        tfrs.loc[:, 'hemi'] = hemi
        cluster_contrasts.append(tfrs)
    logging.info('Done compute contrast')
    return pd.concat(cluster_contrasts)


def augment_data(meta, response_left, stimulus):
    """Augment meta data with fields for specific cases

    Args:
        meta: DataFrame
        response_left: ndarray
            1 if subject made a left_response / yes response
        stimulus: ndarray
            1 if a left_response is correct
    """
    # add columns:
    meta["all"] = 1

    meta["left"] = response_left.astype(int)
    meta["right"] = (~response_left).astype(int)

    meta["hit"] = ((response_left == 1) & (stimulus == 1)).astype(int)
    meta["fa"] = ((response_left == 1) & (stimulus == 0)).astype(int)
    meta["miss"] = ((response_left == 0) & (stimulus == 1)).astype(int)
    meta["cr"] = ((response_left == 0) & (stimulus == 0)).astype(int)
    return meta


def set_jw_style():
    import matplotlib
    import seaborn as sns
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    sns.set(style='ticks', font='Arial', font_scale=1, rc={
        'axes.linewidth': 0.05,
        'axes.labelsize': 7,
        'axes.titlesize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'xtick.major.width': 0.25,
        'xtick.minor.width': 0.25,
        'ytick.major.width': 0.25,
        'text.color': 'Black',
        'axes.labelcolor': 'Black',
        'xtick.color': 'Black',
        'ytick.color': 'Black', })
    sns.plotting_context()

def pmi(*args, **kwargs):
    #from mne.viz.utils import _plot_masked_image as pmi
    import mne
    level = mne.set_log_level('ERROR', return_old_level=True)
    cax = _plot_masked_image(*args, **kwargs)
    mne.set_log_level(level)
    return cax
    
def plot_mosaic(tfr_data, vmin=-5, vmax=5, cmap='RdBu_r',
                ncols=4, epoch='stimulus', stats=False,
                threshold=0.05):
    import matplotlib            
#    from mne.viz.utils import _plot_masked_image as pmi
    if epoch == "stimulus":
        time_cutoff = (-0.3, 0.75)
        xticks = [0, 0.25, 0.5, 0.75, 1]
        xticklabels = ['0\nStim on', '', '.5', '.75\nStim off']
        yticks = [25, 50, 75, 100, 125]
        yticklabels = ['25', '', '75', '', '125']
        xmarker = [0, 1]
        baseline = (-0.3, -0.2)
    elif epoch == "iti":
        time_cutoff = (-0.5, 0.)
        xticks = [-0.5, -0.25, 0]
        xticklabels = ['-0.5', '', '0\nStim on']
        yticks = [25, 50, 75, 100, 125]
        yticklabels = ['25', '', '75', '', '125']
        xmarker = [0, 1]
        baseline = (-0.3, -0.2)        
        
    else:
        time_cutoff = (-1, .5)      
        xticks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5]
        xticklabels = ['-1', '', '-0.5', '', '0\nResponse', '', '0.5']
        yticks = [25, 50, 75, 100, 125]
        yticklabels = ['25', '', '75', '', '125']
        xmarker = [0, 1]
        baseline = None
    from matplotlib import gridspec
    import pylab as plt
    import seaborn as sns
    set_jw_style()
    sns.set_style('ticks')
    nrows = (len(atlas_glasser.areas) // ncols) + 1
    gs = gridspec.GridSpec(nrows, ncols)
    gs.update(wspace=0.01, hspace=0.01)

    for i, (name, area) in enumerate(atlas_glasser.areas.items()):
        try:
            column = np.mod(i, ncols)
            row = i // ncols
            plt.subplot(gs[row, column])
            times, freqs, tfr = get_tfr(tfr_data.query('cluster=="%s"' %
                                                       area), time_cutoff)
                                        
            # cax = plt.gca().pcolormesh(times, freqs, np.nanmean(
            #    tfr, 0), vmin=vmin, vmax=vmax, cmap=cmap, zorder=-2)
            mask = None
            if stats:
                _, _, cluster_p_values, _ = get_tfr_stats(
                    times, freqs, tfr, threshold)
                sig = cluster_p_values.reshape((tfr.shape[1], tfr.shape[2]))
                print(np.shape(cluster_p_values))
                print(tfr.shape[1])
                print(tfr.shape[2])
                mask = sig < threshold
                k, l = np.where((np.nanmean(tfr, 0) > 0) &(sig <= 0.05))
                m, n = np.where((np.nanmean(tfr, 0) < 0) &(sig <= 0.05))
                if name == 'vfcV3ab':
                    print('np.where(tfr > 0)', np.where((np.nanmean(tfr, 0) > 0) &(sig <= 0.05)))
                    print('freqs[k]', freqs[k])
                    print('times[l]', times[l])
                    print('np.where(tfr < 0)', np.where((np.nanmean(tfr, 0) < 0) &(sig <= 0.05)))
                    print('freqs[m]', freqs[m])
                    print('times[n]', times[n])                        
                    print('np.shape(freqs)',np.shape(freqs))
                    print('np.shape(times)', np.shape(times))
                    print('np.shape(np.nanmean(tfr, 0))', np.shape(np.nanmean(tfr, 0)))
            cax = pmi(plt.gca(),  np.nanmean(tfr, 0), times, yvals=freqs,
                      yscale='linear', vmin=vmin, vmax=vmax,
                      mask=mask, mask_alpha=1,
                      mask_cmap=cmap, cmap=cmap)      
            # if area == 'vfcV3ab':
            #     print('name', name)
            #     print('np.where(tfr > 0)', np.where((np.nanmean(tfr, 0) > 0) &(sig <= 0.05)))
            #     print('freqs[k]', freqs[k])
            #     print('times[l]', times[l])
            #     for p in np.unique(freqs[k]):
            #         q = np.where(freqs[k] == p)
            #         print('p', p)
            #         print('times[l][q]', times[l][q])
            #
            #     for r in np.unique(freqs[m]):
            #         s = np.where(freqs[m] == r)
            #         print('r', r)
            #         print('times[n][s]', times[n][s])
            #
            #     # print('np.where(tfr < 0)', np.where((np.nanmean(tfr, 0) < 0) &(sig <= 0.05)))
            #     # print('freqs[m]', freqs[m])
            #     # print('times[n]', times[n])
            #     # print('np.shape(freqs)',np.shape(freqs))
            #     # print('np.shape(times)', np.shape(times))
            #     # print('np.shape(np.nanmean(tfr, 0))', np.shape(np.nanmean(tfr, 0)))
            #     # print('np.nanmean(tfr, 0)', np.nanmean(tfr, 0))
            #     # print('times', times)
            #     # print('np.nanmean(tfr, 0)[:,16:32]', np.nanmean(tfr, 0)[:,16:32])
            #     pdb.set_trace()
            # plt.grid(True, alpha=0.5)
            for xmark in xmarker:
                plt.axvline(xmark, color='k', lw=1, zorder=-1, alpha=0.5)

            plt.yticks(yticks, [''] * len(yticks))
            plt.xticks(xticks, [''] * len(xticks))
            set_title(name, times, freqs, plt.gca())
            plt.tick_params(direction='inout', length=2, zorder=100)
            plt.xlim(time_cutoff)
      #      plt.ylim([1, 147.5])
            plt.axhline(10, color='k', lw=1, alpha=0.5, linestyle='--')

        except ValueError as e:
            print(name, area, e)
    plt.subplot(gs[nrows - 2, 0])

    sns.despine(left=True, bottom=True)
    plt.subplot(gs[nrows - 1, 0])

    pmi(plt.gca(),  np.nanmean(tfr, 0) * 0, times, yvals=freqs,
        yscale='linear', vmin=vmin, vmax=vmax,
        mask=None, mask_alpha=1,
        mask_cmap=cmap, cmap=cmap)

    plt.xticks(xticks, xticklabels)
    plt.yticks(yticks, yticklabels)
    for xmark in xmarker:
        plt.axvline(xmark, color='k', lw=1, zorder=-1, alpha=0.5)
    if baseline is not None:
        plt.fill_between(baseline, y1=[1, 1],
                         y2=[150, 150], color='k', alpha=0.5)
    plt.tick_params(direction='in', length=3)


    plt.xlim(time_cutoff)
    plt.ylim([1, 147.5])
    plt.xlabel('time [s]')
    plt.ylabel('Freq [Hz]')
    sns.despine(ax=plt.gca())

    import matplotlib as mpl
    ax = plt.subplot(gs[nrows - 1, 1])    
    cmap = cmap
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('% signal change')

def plot_2epoch_mosaic(tfr_data, vmin=-5, vmax=5, cmap='RdBu_r',
                       ncols=4, stats=False,
                       threshold=0.05):

    from matplotlib import gridspec
#    from mne.viz.utils import _plot_masked_image as pmi    
    import pylab as plt
    import seaborn as sns
    ncols *= 2
    set_jw_style()
    sns.set_style('ticks')
    nrows = int((len(atlas_glasser.areas) // (ncols / 2)) + 1)
    gs = gridspec.GridSpec(nrows, ncols)

    gs.update(wspace=0.01, hspace=0.05  )
    i = 0
    for (name, area) in atlas_glasser.areas.items():
        for epoch in ['stimulus', 'response']:
            column = int(np.mod(i, ncols))
            row = int(i // ncols)
            if epoch == "stimulus":
                time_cutoff = (-0.3, .75)
                xticks = [0, 0.25, 0.5, 0.75]
                xticklabels = ['0\nStim on', '.25', '.5', '.75\nStim off']
                yticks = [25, 50, 75, 100, 125]
                yticklabels = ['25', '', '75', '', '125']
                xmarker = [0, .75]
                baseline = (-0.3, -0.2)
           
            else:
                time_cutoff = (-0.5, .25)
                xticks = [-0.5, -0.25, 0, 0.25]
                xticklabels = ['-0.5',
                               '', '0\nResponse', '']
                yticks = [1, 25, 50, 75, 100, 125]
                yticklabels = ['1', '25', '', '75', '', '125']
                xmarker = [0]
                baseline = None
                                
            try:
                
                plt.subplot(gs[row, column])
                print(gs, type(row), type(column))
                times, freqs, tfr = get_tfr(
                    tfr_data.query(
                        'cluster=="%s" & epoch=="%s"' % (area, epoch)).dropna(axis=1, how="all"),
                    time_cutoff)
                print(times)
                print(tfr)
                print(column)
#                pdb.set_trace()    
                # cax = plt.gca().pcolormesh(times, freqs, np.nanmean(
                #    tfr, 0), vmin=vmin, vmax=vmax, cmap=cmap, zorder=-2)
                mask = None
                if stats:
                    _, _, cluster_p_values, _ = get_tfr_stats(
                        times, freqs, tfr, threshold)
                    sig = cluster_p_values.reshape(
                        (tfr.shape[1], tfr.shape[2]))
                    mask = sig < threshold                                           
                cax = pmi(plt.gca(),  np.nanmean(tfr, 0), times, yvals=freqs,
                          yscale='linear', vmin=vmin, vmax=vmax, mask_style = "mask",
                          mask=mask, mask_alpha=0.5,
                          mask_cmap=cmap, cmap=cmap)

                # plt.grid(True, alpha=0.5)
                for xmark in xmarker:
                    plt.axvline(xmark, color='k', lw=1, zorder=-1, alpha=0.5)

                plt.yticks(yticks, [''] * len(yticks))
                plt.xticks(xticks, [''] * len(xticks))

                plt.tick_params(direction='inout', length=2, zorder=100)
                plt.xlim(time_cutoff)
                plt.ylim([1, 147.5])
                plt.axhline(10, color='k', lw=1, alpha=0.5, linestyle='--')

                plt.axvline(0, color='k', lw=1, zorder=5, alpha=0.5)
                if epoch == 'stimulus':                    
                    plt.axvline(0.75, color='k', lw=1, zorder=5, alpha=0.5)

            except ValueError as e:
                print(name, area, e)
            i += 1

            if epoch == 'response':
                set_title(name, times[0], freqs, plt.gca())
    sns.despine(left=True, bottom=True)

    epoch = "stimulus"
    time_cutoff = (-0.3, .75)
    xticks = [0, 0.25, 0.5, 0.75]
    xticklabels = ['0\nStim on', '.25', '.5', '.75\nStim off']
    yticks = [25, 50, 75, 100, 125]
    yticklabels = ['25', '', '75', '', '125']
    xmarker = [0, .75]
    baseline = (-0.3, -0.2)
    sns.despine(left=True, bottom=True)
    plt.subplot(gs[nrows - 1, 0])
    # pmi(plt.gca(),  np.nanmean(tfr, 0) * 0, times, yvals=freqs,
    #     yscale='linear', vmin=vmin, vmax=vmax,
    #     mask=None, mask_alpha=1,
    #     mask_cmap=cmap, cmap=cmap)
    plt.xticks(xticks, xticklabels)
    plt.yticks(yticks, yticklabels)
    for xmark in xmarker:
        plt.axvline(xmark, color='k', lw=1, zorder=-1, alpha=0.5)
    if baseline is not None:
        plt.fill_between(baseline, y1=[1, 1],
                         y2=[150, 150], color='k', alpha=0.5)
    plt.tick_params(direction='in', length=3)
    plt.xlim(time_cutoff)
    plt.ylim([1, 147.5])
    plt.xlabel('time [s]')
    plt.ylabel('Freq [Hz]')

    epoch = 'response'
    time_cutoff = (-.5, .25)
    xticks = [-0.5, -0.25, 0, 0.25]
    xticklabels = ['-0.5',
                   '', '0\nResponse', '']
    yticks = [1, 25, 50, 75, 100, 125]
    yticklabels = ['1', '25', '', '75', '', '125']
    xmarker = [0]
    baseline = None

    plt.subplot(gs[nrows - 1, 1])
    # pmi(plt.gca(),  np.nanmean(tfr, 0) * 0, times, yvals=freqs,
    #     yscale='linear', vmin=vmin, vmax=vmax,
    #     mask=None, mask_alpha=1,
    #     mask_cmap=cmap, cmap=cmap)
    plt.xticks(xticks, xticklabels)
    plt.yticks(yticks, [])
    for xmark in xmarker:
        plt.axvline(xmark, color='k', lw=1, zorder=-1, alpha=0.5)
    if baseline is not None:
        plt.fill_between(baseline, y1=[1, 1],
                         y2=[150, 150], color='k', alpha=0.5)
    plt.tick_params(direction='in', length=3)
    plt.xlim(time_cutoff)
    plt.ylim([1, 147.5])
    plt.xlabel('time [s]')
    plt.ylabel('')
    sns.despine(left=True, bottom=True)
    import matplotlib as mpl
    ax = plt.subplot(gs[nrows - 1, 2])    
    cmap = cmap
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('% signal change')
    
def plot_2epoch_mosaic_murphy_order(tfr_data, vmin=-5, vmax=5, cmap='RdBu_r',
                       ncols=9, stats=False,
                       threshold=0.05):

    from matplotlib import gridspec
#    from mne.viz.utils import _plot_masked_image as pmi    
    import pylab as plt
    import seaborn as sns
    ncols *= 2
    set_jw_style()
    sns.set_style('ticks')

    from collections import OrderedDict
    areas = OrderedDict()
    areas['V1'] = 'vfcPrimary'
    areas['V2-4'] = 'vfcEarly'    
    areas['V3A/B'] = 'vfcV3ab'
    areas['IPS0/1'] = 'vfcIPS01'
    areas['IPS2/3'] = 'vfcIPS23'
    areas['aIPS'] = 'JWG_aIPS'
#    areas['aIPS/PCeS'] = 'JWG_aIPS' or 'JWG_IPS_PCeS'
    areas['dlPFC'] = 'HCPMMP1_dlpfc'
    areas['Premotor'] = 'HCPMMP1_premotor'
    areas['M1'] = 'JWG_M1'
    areas['LOC1/2'] = 'vfcLO'   
    areas['MT+'] = 'HCPMMP1_visual_lateral'
    areas['Ventral Occ'] = 'vfcVO'
    areas['PHC'] = 'vfcPHC'
    areas['vlPFC'] = 'HCPMMP1_frontal_inferior'


    nrows = 2    
#    nrows = int((len(areas) // (ncols / 2)) + 1)
    time_cutoff_stim = (-0.3, 0.75)
    time_cutoff_resp = (-0.3, 0.2)
    # time_cutoff_stim = (-0.5, 0.75)
    # time_cutoff_resp = (-0.5, 1.5)
    time_range_stim = time_cutoff_stim[1] - time_cutoff_stim[0]
    time_range_resp = time_cutoff_resp[1] - time_cutoff_resp[0]   
     
    gs = gridspec.GridSpec(nrows, ncols, width_ratios=[time_range_stim, time_range_resp, time_range_stim, time_range_resp, time_range_stim, time_range_resp, time_range_stim, time_range_resp, time_range_stim, time_range_resp, time_range_stim, time_range_resp, time_range_stim, time_range_resp, time_range_stim, time_range_resp, time_range_stim, time_range_resp])

    gs.update(wspace=0.02, hspace=0.1  )
    i = 0
    plt.figure(figsize=(18, 2))
    plt.tight_layout()
    plt.subplots_adjust(left = 0.05, right=0.95)
    
    first_row = ['V1', 'V2-4', 'V3A/B', 'IPS0/1', 'IPS2/3', 'aIPS', 'dlPFC', 'Premotor', 'M1']   
    second_row = ['LOC1/2', 'MT+', 'Ventral Occ', 'PHC', 'vlPFC']
    
    for (name, area) in areas.items():
#        print(name)
#        pdb.set_trace()
        if name == 'LOC1/2':
            i+= 4
        
        for epoch in ['stimulus', 'response']:
#            ax = axes[i]
            column = int(np.mod(i, ncols))
            row = int(i // ncols)  
            if epoch == "stimulus":
                time_cutoff = (-0.3, .75)
 #               time_cutoff = (-0.5, .75)                
                xticks = [0, 0.25, 0.5, 0.75]
                xticklabels = ['0\nStim on', '.25', '.5', '.75\nStim off']
                # yticks = [25, 50, 75, 100, 125]
                # yticklabels = ['25', '', '75', '', '125']
                yticks = [25, 50, 75, 100, 125, 150]
                yticklabels = ['25', '50', '75', '100', '125', '150']                
                xmarker = [0, .75]
                baseline = (-0.35, -0.1)
                
           
            else:
                time_cutoff = (-0.3, .2)
 #               time_cutoff = (-0.5, 1.5)                
                xticks = [-0.2, -0.1, 0, 0.2]
                xticklabels = ['-0.4',
                               '', '0\nResponse', '']
                yticks = [1, 25, 50, 75, 100, 125, 150]
                yticklabels = ['1', '25', '50', '75', '100', '125', '150']
                xmarker = [0]
                baseline = None
                                
            try:
                
                plt.subplot(gs[row, column])
 #               print(gs, type(row), type(column))
 
                tfr_data.query(
                    'cluster=="%s" & epoch=="%s"' % (area, epoch)).dropna(axis=1, how="all")
               # print(tfr_data)
                # pdb.set_trace()
                times, freqs, tfr = get_tfr(
                    tfr_data.query(
                        'cluster=="%s" & epoch=="%s"' % (area, epoch)).dropna(axis=1, how="all"),
                    time_cutoff)
                # if name == 'aIPS/PCeS':
                #     times, freqs, tfr = get_tfr(
                #         (tfr_data.query(
                #             'cluster=="JWG_aIPS" or cluster =="JWG_IPS_PCeS" & epoch=="%s"' % (epoch)).dropna(axis=1, how="all")).groupby(['freq', 'subject']).mean(),
                #         time_cutoff)
                #     print(tfr)
                #     pdb.set_trace()
                # else:
                #     times, freqs, tfr = get_tfr(
                #         tfr_data.query(
                #             'cluster=="%s" & epoch=="%s"' % (area, epoch)).dropna(axis=1, how="all"),
                #         time_cutoff)
 #                   print(tfr)
 #                   pdb.set_trace()    
                # cax = plt.gca().pcolormesh(times, freqs, np.nanmean(
                #    tfr, 0), vmin=vmin, vmax=vmax, cmap=cmap, zorder=-2)
                mask = None
                # if stats:
                #     import joblib
                #     hash = joblib.hash((times, freqs, tfr, threshold))
                #     try:
                #         _, _, cluster_p_values, _ = stats[hash]
                #     except KeyError:
                #         s = get_tfr_stats(
                #             times, freqs, tfr, threshold)
                #         _, _, cluster_p_values, _ = s[hash]
                #
                #     sig = cluster_p_values.reshape(
                #         (tfr.shape[1], tfr.shape[2]))
                #     mask = sig < threshold 
     
                if stats:
                    _, _, cluster_p_values, _ = get_tfr_stats(
                        times, freqs, tfr, threshold)
                    sig = cluster_p_values.reshape(
                        (tfr.shape[1], tfr.shape[2]))
                    print('name', name)
                    print('epoch', epoch)
              # #      print('cluster_p_values', cluster_p_values)
              #       print('np.shape(freqs)', np.shape(freqs))
              #       k, l = np.where(sig <= 0.05)
              #       # print('k', k)
              #       # print('l', l)
                    k, l = np.where((np.nanmean(tfr, 0) > 0) &(sig <= 0.05))
                    m, n = np.where((np.nanmean(tfr, 0) < 0) &(sig <= 0.05))
#                    if name == 'V3A/B':
                        # print('np.where(tfr > 0)', np.where((np.nanmean(tfr, 0) > 0) &(sig <= 0.05)))
                        # print('freqs[k]', freqs[k])
                        # print('times[l]', times[l])
                        # print('np.where(tfr < 0)', np.where((np.nanmean(tfr, 0) < 0) &(sig <= 0.05)))
                        # print('freqs[m]', freqs[m])
                        # print('times[n]', times[n])
                        # print('np.shape(freqs)',np.shape(freqs))
                        # print('np.shape(times)', np.shape(times))
                        # print('np.shape(np.nanmean(tfr, 0))', np.shape(np.nanmean(tfr, 0)))
                        # print('name', name)
                        # print('np.where(tfr > 0)', np.where((np.nanmean(tfr, 0) > 0) &(sig <= 0.05)))
                        # print('freqs[k]', freqs[k])
                        # print('times[l]', times[l])
                    for p in np.unique(freqs[k]):
                        q = np.where(freqs[k] == p)
                        print('p', p)
                        print('times[l][q]', times[l][q])

                    for r in np.unique(freqs[m]):
                        s = np.where(freqs[m] == r)
                        print('r', r)
                        print('times[n][s]', times[n][s])
                        
                    # print('np.shape(tfr)', np.shape(np.nanmean(tfr, 0)))
                    # print('np.nanmean(tfr, 0)[k,l]', np.nanmean(tfr, 0)[k,l])
                    # print('freqs[k]', freqs[k])
                    # print('np.shape(sig)', np.shape(sig))
                    # print('times',times)
                    # if area == 'vfcV3ab':
                    #     print('name', name)

                    
                    
                    mask = sig < threshold  
                    # print('np.shape(mask)', np.shape(mask))
                    # print('mask', mask*np.nanmean(tfr, 0))
                    # pdb.set_trace()                                                 
                cax = pmi(plt.gca(),  np.nanmean(tfr, 0), times, yvals=freqs,
                          yscale='linear', vmin=vmin, vmax=vmax, mask_style = "mask",#mask_style = "both",#mask_style = "contour", #mask_style = "mask", 
                          mask=mask, mask_alpha=0.5,
                          mask_cmap=cmap, cmap=cmap)#
                          

                # plt.grid(True, alpha=0.5)
                for xmark in xmarker:
                    plt.axvline(xmark, color='k', lw=1, zorder=-1, alpha=0.5)

                
                plt.yticks(yticks, [''] * len(yticks))
                plt.xticks(xticks, [''] * len(xticks))

                
                plt.tick_params(direction='inout', length=2, zorder=100)
                plt.xlim(time_cutoff)
                plt.ylim([1, 160])
                plt.axhline(10, color='k', lw=1, alpha=0.5, linestyle='--')

                plt.axvline(0, color='k', lw=1, zorder=5, alpha=0.5)
                if epoch == 'stimulus':                    
                    plt.axvline(0.75, color='k', lw=1, zorder=5, alpha=0.5)
                if column == 0:
                    plt.xlabel('time [s]')
                    plt.ylabel('Freq [Hz]')  
                    plt.gca().set_xticklabels(xticklabels)
                    plt.gca().set_yticklabels(yticklabels)                    
                elif column == 1:    
 #                   plt.ylabel('Freq [Hz]')  
                    plt.gca().set_xticklabels(xticklabels)                    
            except ValueError as e:
                print(name, area, e)
            i += 1

            if epoch == 'response':
                set_title(name, times[0], freqs, plt.gca())
                xticks = [-0.2, 0, 0.2]
                xticklabels = ['', '0\nResponse', '']

            if epoch == 'stimulus':                               
                xticks = [0, 0.25, 0.5, 0.75]
                xticklabels = ['0\nStim on', '.25', '.5', '.75\nStim off']               
                               
    sns.despine(left=True, bottom=True)

    # epoch = "stimulus"
    # time_cutoff = (-0.3, .75)
    # xticks = [0, 0.25, 0.5, 0.75]
    # xticklabels = ['0\nStim on', '.25', '.5', '.75\nStim off']
    # yticks = [25, 50, 75, 100, 125]
    # yticklabels = ['25', '', '75', '', '125']
    # xmarker = [0, .75]
    # baseline = (-0.3, -0.2)
    # sns.despine(left=True, bottom=True)
    # plt.subplot(gs[nrows - 1, 0])
    # # pmi(plt.gca(),  np.nanmean(tfr, 0) * 0, times, yvals=freqs,
    # #     yscale='linear', vmin=vmin, vmax=vmax,
    # #     mask=None, mask_alpha=1,
    # #     mask_cmap=cmap, cmap=cmap)
    # plt.xticks(xticks, xticklabels)
    # plt.yticks(yticks, yticklabels)
    # for xmark in xmarker:
    #     plt.axvline(xmark, color='k', lw=1, zorder=-1, alpha=0.5)
    # if baseline is not None:
    #     plt.fill_between(baseline, y1=[1, 1],
    #                      y2=[150, 150], color='k', alpha=0.5)
    # plt.tick_params(direction='in', length=3)
    # plt.xlim(time_cutoff)
    # plt.ylim([1, 147.5])
    # plt.xlabel('time [s]')
    # plt.ylabel('Freq [Hz]')
    #
    # epoch = 'response'
    # time_cutoff = (-.5, .25)
    # xticks = [-0.5, -0.25, 0, 0.25]
    # xticklabels = ['-0.5',
    #                '', '0\nResponse', '']
    # yticks = [1, 25, 50, 75, 100, 125]
    # yticklabels = ['1', '25', '', '75', '', '125']
    # xmarker = [0]
    # baseline = None
    #
    # plt.subplot(gs[nrows - 1, 1])
    # # pmi(plt.gca(),  np.nanmean(tfr, 0) * 0, times, yvals=freqs,
    # #     yscale='linear', vmin=vmin, vmax=vmax,
    # #     mask=None, mask_alpha=1,
    # #     mask_cmap=cmap, cmap=cmap)
    # plt.xticks(xticks, xticklabels)
    # plt.yticks(yticks, [])
    # for xmark in xmarker:
    #     plt.axvline(xmark, color='k', lw=1, zorder=-1, alpha=0.5)
    # if baseline is not None:
    #     plt.fill_between(baseline, y1=[1, 1],
    #                      y2=[150, 150], color='k', alpha=0.5)
    # plt.tick_params(direction='in', length=3)
    # plt.xlim(time_cutoff)
    # plt.ylim([1, 147.5])
    # plt.xlabel('time [s]')
    # plt.ylabel('')
    # sns.despine(left=True, bottom=True)
    import matplotlib as mpl
    # ax = plt.subplot(gs[nrows - 1, 15])
    # cmap = cmap
    # norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
    #                                 norm=norm, alpha=0.5,
    #                                 orientation='vertical')
    # cb1.set_label('% signal change\n p > 0.05')                                 
    ax = plt.subplot(gs[nrows - 1, 17])
    cmap = cmap
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)                                   
    cb2 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm, 
                                    orientation='vertical')                                
    cb2.set_label('% signal change') 

def plot_tfr_no_stats_single_sj(tfr_data, vmin=-5, vmax=5, cmap='RdBu_r', threshold=0.05, stats = False):
    import pylab as plt
#    from mne.viz.utils import _plot_masked_image as pmi
    from matplotlib import gridspec
    import seaborn as sns

    time_cutoff_stim = (-0.3, 0.75)
    time_cutoff_resp = (-0.3, 0.2)
    # time_cutoff_stim = (-0.5, 0.75)
    # time_cutoff_resp = (-0.5, 1.5)
    time_range_stim = time_cutoff_stim[1] - time_cutoff_stim[0]
    time_range_resp = time_cutoff_resp[1] - time_cutoff_resp[0]   
     
    subjects = np.unique(tfr_data.index.get_level_values('subject'))
    ncols = 6
    ncols *= 2
    nrows = 6   
    gs = gridspec.GridSpec(nrows, ncols, width_ratios=[time_range_stim, time_range_resp]*6)

    gs.update(wspace=0.02, hspace=0.2  )
    
#    gs = gridspec.GridSpec(5, 6)
#    gs.update(wspace=0.05, hspace=0.2  )
    i = 0
    plt.figure(figsize=(18,15))
    plt.tight_layout()
    plt.subplots_adjust(left = 0.05, right=0.95)
    
    areas = ["vfcPrimary", "vfcEarly", "vfcV3ab", "vfcIPS01", "vfcIPS23", "JWG_aIPS", "HCPMMP1_dlpfc", "HCPMMP1_premotor", "JWG_M1", "vfcLO", "HCPMMP1_visual_lateral", "vfcVO", "vfcPHC", "HCPMMP1_frontal_inferior"]
    tfr_data = tfr_data.query('cluster in ["vfcPrimary", "vfcEarly", "vfcV3ab", "vfcIPS01", "vfcIPS23", "JWG_aIPS", "HCPMMP1_dlpfc", "HCPMMP1_premotor", "JWG_M1", "vfcLO", "HCPMMP1_visual_lateral", "vfcVO", "vfcPHC", "HCPMMP1_frontal_inferior"]').groupby(['freq', 'subject', 'epoch']).mean()
        
    i = 0
    for sj in np.unique(tfr_data.index.get_level_values('subject')):
        df = tfr_data.query('subject == "{}"'.format(sj))
        for epoch in ['stimulus', 'response']:
            column = int(np.mod(i, ncols))
            row = int(i // ncols)  

            if epoch == "stimulus":
                time_cutoff = (-0.3, .75)
 #               time_cutoff = (-0.5, .75)
                xticks = [0, 0.25, 0.5, 0.75]
                xticklabels = ['0\nStim on', '.25', '.5', '.75\nStim off']
                # yticks = [25, 50, 75, 100, 125]
                # yticklabels = ['25', '', '75', '', '125']
                yticks = [25, 50, 75, 100, 125, 150]
                yticklabels = ['25', '50', '75', '100', '125', '150']
                xmarker = [0, .75]
                baseline = (-0.3, -0.2)
            else:
                time_cutoff = (-0.3, .2)
 #               time_cutoff = (-0.5, 1.5)
                xticks = [-0.2, 0, 0.2]
                xticklabels = ['-0.4',
                               '', '0\nResponse', '']
                yticks = [1, 25, 50, 75, 100, 125, 150]
                yticklabels = ['1', '25', '50', '75', '100', '125', '150']
                xmarker = [0]
                baseline = None

            plt.subplot(gs[row, column])
            df.query('epoch=="%s"' % (epoch)).dropna(axis=1, how="all")

            times, freqs, tfr = get_tfr(
                df.query(
                    'epoch=="%s"' % (epoch)).dropna(axis=1, how="all"),
                time_cutoff)

            mask = None


            if stats:
                _, _, cluster_p_values, _ = get_tfr_stats(
                    times, freqs, tfr, threshold)
                sig = cluster_p_values.reshape(
                    (tfr.shape[1], tfr.shape[2]))

                k, l = np.where((np.nanmean(tfr, 0) > 0) &(sig <= 0.05))
                m, n = np.where((np.nanmean(tfr, 0) < 0) &(sig <= 0.05))
                if name == 'V3A/B':
                    print('np.where(tfr > 0)', np.where((np.nanmean(tfr, 0) > 0) &(sig <= 0.05)))
                    print('freqs[k]', freqs[k])
                    print('times[l]', times[l])
                    print('np.where(tfr < 0)', np.where((np.nanmean(tfr, 0) < 0) &(sig <= 0.05)))
                    print('freqs[m]', freqs[m])
                    print('times[n]', times[n])                        
                    print('np.shape(freqs)',np.shape(freqs))
                    print('np.shape(times)', np.shape(times))
                    print('np.shape(np.nanmean(tfr, 0))', np.shape(np.nanmean(tfr, 0)))
                # print('np.shape(tfr)', np.shape(np.nanmean(tfr, 0)))
                # print('np.nanmean(tfr, 0)[k,l]', np.nanmean(tfr, 0)[k,l])
                # print('freqs[k]', freqs[k])
                # print('np.shape(sig)', np.shape(sig))
                # print('times',times)
        
                mask = sig < threshold  
                # print('np.shape(mask)', np.shape(mask))
                # print('mask', mask*np.nanmean(tfr, 0))
                # pdb.set_trace()                                                 
            cax = pmi(plt.gca(),  np.nanmean(tfr, 0), times, yvals=freqs,
                      yscale='linear', vmin=vmin, vmax=vmax, mask_style = "mask",
                      mask=mask, mask_alpha=0.5,
                      mask_cmap=cmap, cmap=cmap)

            # plt.grid(True, alpha=0.5)
            for xmark in xmarker:
                plt.axvline(xmark, color='k', lw=1, zorder=-1, alpha=0.5)

    
            plt.yticks(yticks, [''] * len(yticks))
            plt.xticks(xticks, [''] * len(xticks))

    
            plt.tick_params(direction='inout', length=2, zorder=100)
            plt.xlim(time_cutoff)
            plt.ylim([1, 160])
            plt.axhline(10, color='k', lw=1, alpha=0.5, linestyle='--')

            plt.axvline(0, color='k', lw=1, zorder=5, alpha=0.5)
            if epoch == 'stimulus':                    
                plt.axvline(0.75, color='k', lw=1, zorder=5, alpha=0.5)
            if column == 0:
                plt.ylabel('Freq [Hz]')  
                plt.gca().set_yticklabels(yticklabels)                    
#            if column == 1:    
#                   plt.ylabel('Freq [Hz]')  
#                plt.gca().set_xticklabels(xticklabels)       
            if row == nrows -1:    
                plt.gca().set_xticklabels(xticklabels)
                plt.xlabel('time [s]')             

            i += 1

            if epoch == 'response':
                set_title('subject' + sj, times[0], freqs + 15, plt.gca())
                xticks = [-0.4, -0.2, 0, 0.2]
                xticklabels = ['-0.4',
                               '', '0\nResponse', '']

            if epoch == 'stimulus':                               
                xticks = [0, 0.25, 0.5, 0.75]
                xticklabels = ['0\nStim on', '.25', '.5', '.75\nStim off']               
                           
        sns.despine(left=True, bottom=True)
        
        # column = int(np.mod(i, ncols))
        # row = int(i // ncols)
        # plt.subplot(gs[row, column])
        # times, freqs, tfr = get_tfr(df.query('subject == "{}"'.format(sj)), (-0.3, 0.75))
        # cax = pmi(plt.gca(), tfr.mean(0), times, yvals=freqs,
        #           yscale='linear', vmin=vmin, vmax=vmax,
        #           mask_alpha=1, mask_cmap=cmap, cmap=cmap)#, linewidth=1)
        # plt.gca().set_title('subject' + str(sj))
        # plt.gca().set_xlabel('Time [s]')
        # plt.gca().set_ylabel('Freq [Hz]')
        #
        # i += 1         

def plot_tfr_no_stats_leave_one_out(tfr_data, vmin=-5, vmax=5, cmap='RdBu_r', threshold=0.05, stats = False):
    import pylab as plt
#    from mne.viz.utils import _plot_masked_image as pmi
    from matplotlib import gridspec
    import seaborn as sns

    time_cutoff_stim = (-0.3, 0.75)
    time_cutoff_resp = (-0.4, 0.2)
    # time_cutoff_stim = (-0.5, 0.75)
    # time_cutoff_resp = (-0.5, 1.5)
    time_range_stim = time_cutoff_stim[1] - time_cutoff_stim[0]
    time_range_resp = time_cutoff_resp[1] - time_cutoff_resp[0]   
     
    subjects = np.unique(tfr_data.index.get_level_values('subject'))
    ncols = 6
    ncols *= 2
    nrows = 5   
    gs = gridspec.GridSpec(nrows, ncols, width_ratios=[time_range_stim, time_range_resp]*6)

    gs.update(wspace=0.02, hspace=0.2  )
    
#    gs = gridspec.GridSpec(5, 6)
#    gs.update(wspace=0.05, hspace=0.2  )
    i = 0
    plt.figure(figsize=(18,15))
    plt.tight_layout()
    plt.subplots_adjust(left = 0.05, right=0.95)
    
    areas = ["vfcPrimary", "vfcEarly", "vfcV3ab", "vfcIPS01", "vfcIPS23", "JWG_aIPS", "HCPMMP1_dlpfc", "HCPMMP1_premotor", "JWG_M1", "vfcLO", "HCPMMP1_visual_lateral", "vfcVO", "vfcPHC", "HCPMMP1_frontal_inferior"]
    tfr_data = tfr_data.query('cluster in ["vfcPrimary", "vfcEarly", "vfcV3ab", "vfcIPS01", "vfcIPS23", "JWG_aIPS", "HCPMMP1_dlpfc", "HCPMMP1_premotor", "JWG_M1", "vfcLO", "HCPMMP1_visual_lateral", "vfcVO", "vfcPHC", "HCPMMP1_frontal_inferior"]').groupby(['freq', 'subject', 'epoch']).mean()
        
    i = 0
    for sj in np.unique(tfr_data.index.get_level_values('subject')):
        df = tfr_data.query('~(subject == "{}")'.format(sj))
        for epoch in ['stimulus', 'response']:
            column = int(np.mod(i, ncols))
            row = int(i // ncols)  

            if epoch == "stimulus":
                time_cutoff = (-0.3, .75)
 #               time_cutoff = (-0.5, .75)
                xticks = [0, 0.25, 0.5, 0.75]
                xticklabels = ['0\nStim on', '.25', '.5', '.75\nStim off']
                # yticks = [25, 50, 75, 100, 125]
                # yticklabels = ['25', '', '75', '', '125']
                yticks = [25, 50, 75, 100, 125, 150]
                yticklabels = ['25', '50', '75', '100', '125', '150']
                xmarker = [0, .75]
                baseline = (-0.3, -0.2)
            else:
                time_cutoff = (-0.4, .2)
 #               time_cutoff = (-0.5, 1.5)
                xticks = [-0.4, -0.2, 0, 0.2]
                xticklabels = ['-0.4',
                               '', '0\nResponse', '']
                yticks = [1, 25, 50, 75, 100, 125, 150]
                yticklabels = ['1', '25', '50', '75', '100', '125', '150']
                xmarker = [0]
                baseline = None

            plt.subplot(gs[row, column])
            df.query('epoch=="%s"' % (epoch)).dropna(axis=1, how="all")

            times, freqs, tfr = get_tfr(
                df.query(
                    'epoch=="%s"' % (epoch)).dropna(axis=1, how="all"),
                time_cutoff)

            mask = None


            if stats:
                _, _, cluster_p_values, _ = get_tfr_stats(
                    times, freqs, tfr, threshold)
                sig = cluster_p_values.reshape(
                    (tfr.shape[1], tfr.shape[2]))

                k, l = np.where((np.nanmean(tfr, 0) > 0) &(sig <= 0.05))
                m, n = np.where((np.nanmean(tfr, 0) < 0) &(sig <= 0.05))
                if name == 'V3A/B':
                    print('np.where(tfr > 0)', np.where((np.nanmean(tfr, 0) > 0) &(sig <= 0.05)))
                    print('freqs[k]', freqs[k])
                    print('times[l]', times[l])
                    print('np.where(tfr < 0)', np.where((np.nanmean(tfr, 0) < 0) &(sig <= 0.05)))
                    print('freqs[m]', freqs[m])
                    print('times[n]', times[n])                        
                    print('np.shape(freqs)',np.shape(freqs))
                    print('np.shape(times)', np.shape(times))
                    print('np.shape(np.nanmean(tfr, 0))', np.shape(np.nanmean(tfr, 0)))

                # print('np.shape(tfr)', np.shape(np.nanmean(tfr, 0)))
                # print('np.nanmean(tfr, 0)[k,l]', np.nanmean(tfr, 0)[k,l])
                # print('freqs[k]', freqs[k])
                # print('np.shape(sig)', np.shape(sig))
                # print('times',times)
        
                mask = sig < threshold  
                # print('np.shape(mask)', np.shape(mask))
                # print('mask', mask*np.nanmean(tfr, 0))
                # pdb.set_trace()                                                 
            cax = pmi(plt.gca(),  np.nanmean(tfr, 0), times, yvals=freqs,
                      yscale='linear', vmin=vmin, vmax=vmax, mask_style = "mask",
                      mask=mask, mask_alpha=0.5,
                      mask_cmap=cmap, cmap=cmap)

            # plt.grid(True, alpha=0.5)
            for xmark in xmarker:
                plt.axvline(xmark, color='k', lw=1, zorder=-1, alpha=0.5)

    
            plt.yticks(yticks, [''] * len(yticks))
            plt.xticks(xticks, [''] * len(xticks))

    
            plt.tick_params(direction='inout', length=2, zorder=100)
            plt.xlim(time_cutoff)
            plt.ylim([1, 160])
            plt.axhline(10, color='k', lw=1, alpha=0.5, linestyle='--')

            plt.axvline(0, color='k', lw=1, zorder=5, alpha=0.5)
            if epoch == 'stimulus':                    
                plt.axvline(0.75, color='k', lw=1, zorder=5, alpha=0.5)
            if column == 0:
                plt.ylabel('Freq [Hz]')  
                plt.gca().set_yticklabels(yticklabels)                    
#            if column == 1:    
#                   plt.ylabel('Freq [Hz]')  
#                plt.gca().set_xticklabels(xticklabels)       
            if row == nrows -1:    
                plt.gca().set_xticklabels(xticklabels)
                plt.xlabel('time [s]')             

            i += 1

            if epoch == 'response':
                set_title('Without subject' + sj, times[0], freqs + 15, plt.gca())
                xticks = [-0.4, -0.2, 0, 0.2]
                xticklabels = ['-0.4',
                               '', '0\nResponse', '']

            if epoch == 'stimulus':                               
                xticks = [0, 0.25, 0.5, 0.75]
                xticklabels = ['0\nStim on', '.25', '.5', '.75\nStim off']               
                           
        sns.despine(left=True, bottom=True)
        
# def plot_tfr_no_stats_leave_one_out(df, vmin=-5, vmax=5, cmap='RdBu_r', threshold=0.05):
#     import pylab as plt
#     from mne.viz.utils import _plot_masked_image as pmi
#     from matplotlib import gridspec
#     import seaborn as sns
#     ncols = 6
#     gs = gridspec.GridSpec(5, 6)
#     gs.update(wspace=0.05, hspace=0.2  )
#     i = 0
#     plt.figure(figsize=(18,15))
#     plt.tight_layout()
#     plt.subplots_adjust(left = 0.05, right=0.95)
#
#     areas = ["vfcPrimary", "vfcEarly", "vfcV3ab", "vfcIPS01", "vfcIPS23", "JWG_aIPS", "HCPMMP1_dlpfc", "HCPMMP1_premotor", "JWG_M1", "vfcLO", "HCPMMP1_visual_lateral", "vfcVO", "vfcPHC", "HCPMMP1_frontal_inferior"]
#     df = df.query('cluster in ["vfcPrimary", "vfcEarly", "vfcV3ab", "vfcIPS01", "vfcIPS23", "JWG_aIPS", "HCPMMP1_dlpfc", "HCPMMP1_premotor", "JWG_M1", "vfcLO", "HCPMMP1_visual_lateral", "vfcVO", "vfcPHC", "HCPMMP1_frontal_inferior"]').groupby(['freq', 'subject']).mean()
#
# #    times, freqs, tfr = get_tfr(df, (-np.inf, np.inf))
#
#     # T_obs, clusters, cluster_p_values, h0 = get_tfr_stats(
#     #     times, freqs, tfr, 0.05)
#     # sig = cluster_p_values.reshape((tfr.shape[1], tfr.shape[2]))
#     i = 0
#     for sj in np.unique(df.index.get_level_values('subject')):
#
#         column = int(np.mod(i, ncols))
#         row = int(i // ncols)
#         plt.subplot(gs[row, column])
#         times, freqs, tfr = get_tfr(df.query('~(subject == "{}")'.format(sj)), (-0.3, 0.75))
#         cax = pmi(plt.gca(), tfr.mean(0), times, yvals=freqs,
#                   yscale='linear', vmin=vmin, vmax=vmax,
#                   mask_alpha=1, mask_cmap=cmap, cmap=cmap)#, linewidth=1)
#         plt.gca().set_title('without subject' + str(sj))
#         plt.gca().set_xlabel('Time [s]')
#         plt.gca().set_ylabel('Freq [Hz]')
#         i += 1    

def plot_timecourse(tfr_data, vmin=-5, vmax=5, cmap='RdBu_r',
                ncols=4, epoch='stimulus', stats=False,
                threshold=0.05):
#    from mne.viz.utils import _plot_masked_image as pmi
    if epoch == "stimulus":
        time_cutoff = (-0.25, 0.75)
        xticks = [0, 0.25, 0.5, 0.75, 1]
        xticklabels = ['0\nStim on', '', '.5', '.75\nStim off']
  #      yticks = [25, 50, 75, 100, 125]
  #      yticklabels = ['25', '', '75', '', '125']
        xmarker = [0, 1]
        baseline = (-0.25, -0.15)
    elif epoch == "iti":
        time_cutoff = (-0.5, 0.)
        xticks = [-0.5, -0.25, 0]
        xticklabels = ['-0.5', '', '0\nStim on']
   #     yticks = [25, 50, 75, 100, 125]
        yticklabels = ['25', '', '75', '', '125']
        xmarker = [0, 1]
        baseline = (-0.25, -0.15)        
        
    else:
        time_cutoff = (-1, .5)      
        xticks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5]
        xticklabels = ['-1', '', '-0.5', '', '0\nResponse', '', '0.5']
    #    yticks = [1, 25, 50, 75, 100, 125]
    #    yticklabels = ['1', '25', '', '75', '', '125']
        xmarker = [0, 1]
        baseline = None
    from matplotlib import gridspec
    import pylab as plt
    import seaborn as sns
    set_jw_style()
    sns.set_style('ticks')
    nrows = (len(atlas_glasser.areas) // ncols) + 1
    gs = gridspec.GridSpec(nrows, ncols)
    gs.update(wspace=0.01, hspace=0.01)

    for i, (name, area) in enumerate(atlas_glasser.areas.items()):
        try:
            column = np.mod(i, ncols)
            row = i // ncols
            plt.subplot(gs[row, column])
                                                       
            times = np.array(tfr_data.columns, dtype=float)
            time_ind = (times > time_cutoff[0]) & (times < time_cutoff[1])
            time_ind = (times > time_cutoff[0]) & (times < time_cutoff[1])

           # tfrs = [tfr_data.loc[tfr_data.index.isin([subj], level='subject'), time_ind].values
        #            for subj in np.unique(tfr_data.index.get_level_values('subject'))]
                                                           
            # cax = plt.gca().pcolormesh(times, freqs, np.nanmean(
            #    tfr, 0), vmin=vmin, vmax=vmax, cmap=cmap, zorder=-2)
            mask = None
            if stats:
                _, _, cluster_p_values, _ = get_tfr_stats(
                    times, freqs, tfr, threshold)
                sig = cluster_p_values.reshape((tfr.shape[1], tfr.shape[2]))
                mask = sig < threshold
            cax = plt.gca().plot( times, tfr_data.values)

            # plt.grid(True, alpha=0.5)
            for xmark in xmarker:
                plt.axvline(xmark, color='k', lw=1, zorder=-1, alpha=0.5)

            plt.yticks(yticks, [''] * len(yticks))
            plt.xticks(xticks, [''] * len(xticks))
            set_title(name, times, freqs, plt.gca())
            plt.tick_params(direction='inout', length=2, zorder=100)
            plt.xlim(time_cutoff)
      #      plt.ylim([1, 147.5])
            plt.axhline(10, color='k', lw=1, alpha=0.5, linestyle='--')
        except ValueError as e:
            print(name, area, e)
    plt.subplot(gs[nrows - 2, 0])

    sns.despine(left=True, bottom=True)
    plt.subplot(gs[nrows - 1, 0])
    plt.gca().plot( times, tfr_data.values)
    # pmi(plt.gca(),  np.nanmean(tfr, 0) * 0, times, yvals=freqs,
    #     yscale='linear', vmin=vmin, vmax=vmax,
    #     mask=None, mask_alpha=1,
    #     mask_cmap=cmap, cmap=cmap)
    plt.xticks(xticks, xticklabels)
 #   plt.yticks(yticks, yticklabels)
    for xmark in xmarker:
        plt.axvline(xmark, color='k', lw=1, zorder=-1, alpha=0.5)
    if baseline is not None:
        plt.fill_between(baseline, y1=[1, 1],
                         y2=[150, 150], color='k', alpha=0.5)
    plt.tick_params(direction='in', length=3)
    plt.xlim(time_cutoff)
    plt.ylim([1, 147.5])
    plt.xlabel('time [s]')
    plt.ylabel('Freq [Hz]')
    sns.despine(ax=plt.gca())


def plot_tfr(df, vmin=-7.5, vmax=7.5, cmap='RdBu_r', threshold=0.05):
    import pylab as plt
#    from mne.viz.utils import _plot_masked_image as pmi
    times, freqs, tfr = get_tfr(df, (-0.35, 0.75))
    print('freqs', freqs)
    T_obs, clusters, cluster_p_values, h0 = get_tfr_stats(
        times, freqs, tfr, 0.05)
    sig = cluster_p_values.reshape((tfr.shape[1], tfr.shape[2]))
    print('sig', sig)
    k, l = np.where((np.nanmean(tfr, 0) < 0) &(sig <= 0.05))
    print('sig[k,l]', sig[k,l])
    print('freqs[k]', freqs[k])
    print('times[l]',times[l])    
    mask=sig < threshold
    from mne.viz.utils import _plot_masked_image as pmi 
    
    #if yvals is None:  # for e.g. Evoked images
    yvals = freqs #np.arange(df.shape[0])
    ratio = yvals[1:] / yvals[:-1]
    # compute bounds between time samples
    time_diff = np.diff(times) / 2.0 if len(times) > 1 else [0.0005]
    time_lims = np.concatenate(
        [[times[0] - time_diff[0]], times[:-1] + time_diff, [times[-1] + time_diff[-1]]]
    )

    log_yvals = np.concatenate([[yvals[0] / ratio[0]], yvals, [yvals[-1] * ratio[0]]])
    yval_lims = np.sqrt(log_yvals[:-1] * log_yvals[1:])
    # construct a time-yvaluency bounds grid
    time_mesh, yval_mesh = np.meshgrid(time_lims, yval_lims)
    
    
    if mask is not None:
        cax = pmi(plt.gca(), tfr.mean(0), times, yvals=freqs,
                  yscale='log', vmin=vmin, vmax=vmax, mask=sig < threshold,
                  mask_alpha=.5, mask_cmap=cmap, cmap=cmap)#, linewidth=1)

        
        # im = ax.pcolormesh(
        #     time_mesh, yval_mesh, data, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True
        # )
        if mask.sum() > 0:
            big_mask = np.kron(mask, np.ones((10, 10)))
            big_times = np.kron(times, np.ones((10,)))
            big_yvals = np.kron(yvals, np.ones((10,)))
            plt.gca().contour(
                big_times,
                big_yvals,
                big_mask,
                colors=["k"],
                linewidths=[0.75],
                corner_mask=False,
                antialiased=False,
                levels=[0.5],
            )
#    cmap = "RdBu_r"
    # import matplotlib
    # norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    # cb1 = matplotlib.colorbar.ColorbarBase(ax = plt.gca(), cmap=cmap, norm=norm, alpha = 0.3)#, orientation='vertical')#
    # cb1.set_label('% signal change n.s.')
            
#    plt.suptitle("Prev. stim. consistent with current choice")
    return cax, times, freqs, tfr


def plot_tfr_no_stats(df, vmin=-5, vmax=5, cmap='RdBu_r', threshold=0.05):
    import pylab as plt
#    from mne.viz.utils import _plot_masked_image as pmi
#    times, freqs, tfr = get_tfr(df, (-np.inf, np.inf))

    # T_obs, clusters, cluster_p_values, h0 = get_tfr_stats(
    #     times, freqs, tfr, 0.05)
    # sig = cluster_p_values.reshape((tfr.shape[1], tfr.shape[2]))
    times, freqs, tfr = get_tfr(df, (-0.3, 0.75))        
#    times, freqs, tfr = get_tfr(df.query('~(subject == "30")'), (-0.3, 0.75))      
    cax = pmi(plt.gca(), tfr.mean(0), times, yvals=freqs,
              yscale='linear', vmin=vmin, vmax=vmax,
              mask_alpha=1, mask_cmap=cmap, cmap=cmap)#, linewidth=1)

    return cax, times, freqs, tfr

@memory.cache()
def get_tfr_stats(times, freqs, tfr, threshold=0.05):
    from mne.stats import permutation_cluster_1samp_test as cluster_test
    return cluster_test(
        tfr, threshold={'start': 0, 'step': 0.2},
        tail=0, n_permutations=1000, n_jobs=2)        
#        connectivity=None, tail=0, n_permutations=1000, n_jobs=2)

# def get_tfr_stats(times, freqs, tfr, threshold=0.05, n_jobs=2):
#     from mne.stats import permutation_cluster_1samp_test as cluster_test
#     import joblib
#     return {joblib.hash([times, freqs, tfr, threshold]): cluster_test(
#         tfr, threshold={'start': 0, 'step': 0.2},
#         connectivity=None, tail=0, n_permutations=1000, n_jobs=n_jobs)}

def plot_tfr_stats(times, freqs, tfr, threshold=0.05):
    import pylab as plt
    from matplotlib.colors import LinearSegmentedColormap
    T_obs, clusters, cluster_p_values, h0 = get_tfr_stats(
        times, freqs, tfr, threshold)
    vmax = 1
    cmap = LinearSegmentedColormap.from_list('Pvals', [(0 / vmax, (1, 1, 1, 0)),
                                                       (0.04999 / vmax,
                                                        (1, 1, 1, 0)),
                                                       (0.05 / vmax,
                                                        (1, 1, 1, 0.5)),
                                                       (1 / vmax, (1, 1, 1, 0.5))]
                                             )
    sig = cluster_p_values.reshape((tfr.shape[1], tfr.shape[2]))

    # df = np.array(list(np.diff(freqs) / 2) + [freqs[-1] - freqs[-2]])
    # dt = np.array(list(np.diff(times) / 2) + [times[-1] - times[-2]])
    # from scipy.interpolate import interp2d
    # print(np.unique((sig < threshold).astype(float)))
    # i = interp2d(times, freqs, sig.astype(float))
    # X, Y = (np.linspace(times[0], times[-1], len(times) * 25),
    #        np.linspace(freqs[0], freqs[-1], len(freqs) * 25))
    # Z = i(X.ravel(), Y.ravel())

    # plt.gca().pcolormesh(times, freqs, sig, vmin=0, vmax=1, cmap=cmap)

    plt.gca().contour(
        times, freqs,
        sig, (threshold),
        linewidths=0.5, colors=('black'))
    return X, Y, Z


def set_title(text, times, freqs, axes):
    import pylab as plt
    x = np.mean(times)
    y = np.max(freqs)
    plt.text(x, y, text, fontsize=8,
             verticalalignment='top', horizontalalignment='center')


def get_tfr(tfr, time_cutoff):
    # variables:
    times = np.array(tfr.columns, dtype=float)
    print(times)
    freqs = np.array(
        np.unique(tfr.index.get_level_values('freq')), dtype=float)
    print(freqs)    
    time_ind = (times > time_cutoff[0]) & (times < time_cutoff[1])
#    time_ind = (times > time_cutoff[0]) & (times < time_cutoff[1])

    tfrs = [tfr.loc[tfr.index.isin([subj], level='subject'), time_ind].values
            for subj in np.unique(tfr.index.get_level_values('subject'))]
    print(tfrs)
    
    # data:
    X = np.stack(tfrs)
    print(np.shape(X))      
    return times[time_ind], freqs, X


def plot_cluster(names, view):
    from pymeg import atlas_glasser
    all_clusters, _, _, _ = atlas_glasser.get_clusters()
    label_names = []
    for name in names:
        cluster_name = atlas_glasser.areas[name]
        label_names.extend(all_clusters[cluster_name])

    plot_roi('lh', label_names, 'r')


#@memory.cache
def plot_roi(hemi, labels, color, annotation='HCPMMP1',
             view='parietal',
             fs_dir=os.environ['SUBJECTS_DIR'],
             subject_id='S04', surf='inflated'):
    import matplotlib
    import os
    import glob
    from surfer import Brain
    from mne import Label
    color = np.array(matplotlib.colors.to_rgba(color))

    brain = Brain(subject_id, hemi, surf, offscreen=False)
    labels = [label.replace('-rh', '').replace('-lh', '') for label in labels]
    # First select all label files

    label_names = glob.glob(os.path.join(
        fs_dir, subject_id, 'label', 'lh*.label'))
    label_names = [label for label in label_names if any(
        [l in label for l in labels])]

    for label in label_names:
        brain.add_label(label, color=color)

    # Now go for annotations
    from nibabel.freesurfer import io
    ids, colors, annot_names = io.read_annot(os.path.join(
        fs_dir, subject_id, 'label', 'lh.%s.annot' % annotation),
        orig_ids=True)

    for i, alabel in enumerate(annot_names):
        if any([label in alabel.decode('utf-8') for label in labels]):
            label_id = colors[i, -1]
            vertices = np.where(ids == label_id)[0]
            l = Label(np.sort(vertices), hemi='lh')
            brain.add_label(l, color=color)
    brain.show_view(view)
    return brain.screenshot()


def ensure_iter(input):
    if isinstance(input, str):
        yield input
    else:
        try:
            for item in input:
                yield item
        except TypeError:
            yield input
            
def _plot_masked_image(ax, data, times, mask=None, yvals=None,
                       cmap="RdBu_r", vmin=None, vmax=None, ylim=None,
                       mask_style="both", mask_alpha=.25, mask_cmap="Greys",
                       **kwargs):

    from matplotlib import ticker

    if yvals is None:  # for e.g. Evoked images
        yvals = np.arange(data.shape[0])
    ratio = yvals[1:] / yvals[:-1]
    # compute bounds between time samples
    time_diff = np.diff(times) / 2. if len(times) > 1 else [0.0005]
    time_lims = np.concatenate([[times[0] - time_diff[0]], times[:-1] +
                                time_diff, [times[-1] + time_diff[-1]]])

    log_yvals = np.concatenate([[yvals[0] / ratio[0]], yvals,
                                [yvals[-1] * ratio[0]]])
    yval_lims = np.sqrt(log_yvals[:-1] * log_yvals[1:])

    # construct a time-yvaluency bounds grid
    time_mesh, yval_mesh = np.meshgrid(time_lims, yval_lims)
    #pdb.set_trace()
    if mask is not None:
        im = ax.pcolormesh(time_mesh, yval_mesh, data, cmap=cmap,
                           vmin=vmin, vmax=vmax)
        big_mask = np.kron(mask, np.ones((10, 10)))
        big_times = np.kron(times, np.ones((10, )))
        big_yvals = np.kron(yvals, np.ones((10, )))
        print(big_mask.shape)
        ax.contour(big_times, big_yvals, big_mask, colors=["k"],
                   linewidths=[.75], corner_mask=False,
                   antialiased=False, levels=[.5])
    else:
        im = ax.pcolormesh(time_mesh, yval_mesh, data, cmap=cmap,
                           vmin=vmin, vmax=vmax)
    if ylim is None:
        ylim = yval_lims[[0, -1]]

    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    # get rid of minor ticks
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    tick_vals = yvals[np.unique(np.linspace(
        0, len(yvals) - 1, 12).round().astype('int'))]
    ax.set_yticks(tick_vals)

    ax.set_xlim(time_lims[0], time_lims[-1])
    ax.set_ylim(ylim)

    return im            
    
def plot_streams_fig(
    df,
    configuration,
    stats=False,        
    flip_cbar=False,
    suffix="",
):
    """
    Produce a plot that aranges TFR according to a gradient from 
    sensory to motor cortex with association cortex in between. 
    To make this figure somewhat adaptive to different frequencies
    and time windows it expects a PlotConfig object as 2nd 
    argument. This object describes how an axis that describes an
    epoch and contrast should be formatted. See doc string and 
    example config above.
    Args:
        df: pd.DataFrame
    Data frame that contains TFR data, output of contrast_tfr.compute_contrast.
        configuration: PlotConfig object
        stats: True, False or dict
    If False show no cluster permutation test, if True compute permuatation
    test and show result as outline, if dict load results of permuation
    test from this. Dict can be populated by contrast_tfr.get_tfr_stats.
    """
    from collections import namedtuple

    Plot = namedtuple(
        "Plot", ["name", "cluster", "location", "annot_y", "annot_x"]
    )

    top, middle, bottom = slice(0, 2), slice(1, 3), slice(2, 4)
    # fmt: off
    layout = [
        Plot("V1", "vfcPrimary", [0, middle], True, True),
        Plot("V2-V4", "vfcEarly", [1, middle], False, True),
        # Dorsal
        Plot("V3ab", "vfcV3ab", [2, top], False, False),
        Plot("IPS0/1", "vfcIPS01", [3, top], False, False),
        Plot("IPS2/3", "vfcIPS23", [4, top], False, False),
        Plot("IPS Post-central", "JWG_IPS_PCeS", [5, top], False, False),
        Plot("FEF", "vfcFEF", [6, top], False, False),
        Plot("dlPFC", "HCPMMP1_dlpfc", [7, top], False, False),
        # Ventral
        Plot("Lateral Occ", "vfcLO", [2, bottom], False, True),
        Plot("MT+", "vfcTO", [3, bottom], False, True),
        Plot("Ventral Occ", "vfcVO", [4, bottom], False, True),
        Plot("PHC", "vfcPHC", [5, bottom], False, True),
        Plot("Insula", "HCPMMP1_insular_front_opercular", [6, bottom], False, True),
        Plot("vlPFC", "HCPMMP1_frontal_inferior", [7, bottom], False, True),
        Plot("PMd/v", "HCPMMP1_premotor", [8, middle], False, True),
        Plot("M1", "JWG_M1", [9, middle], False, True),
    ]
    # fmt: on
    if flip_cbar:
        cmap = "RdBu"
    else:
        cmap = "RdBu_r"

    for contrast_name in configuration.contrasts:        
        fig = plot_tfr_selected_rois(
            contrast_name, df, layout, configuration, cluster_correct=stats, cmap=cmap
        )            
