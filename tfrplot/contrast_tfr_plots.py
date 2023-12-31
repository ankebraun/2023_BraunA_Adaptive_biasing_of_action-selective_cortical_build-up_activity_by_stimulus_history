""""
This code was adapted from Niklas Wilming
"""
from pymeg import contrast_tfr
import pylab as plt
import numpy as np
import joblib
import seaborn as sns
import matplotlib

# from pymeg import atlas_glasser
import os

from joblib import Memory

memory = Memory(location='/Users/anke/Documents/MEG_study/git/__pycache__', verbose=0)

memory = Memory(cachedir=os.environ['PYMEG_CACHE_DIR'], verbose=0)


class PlotConfig(object):
    """
    Make configureation of plots easier
    """

    def __init__(self, time_windows, contrasts, stat_windows=None):
        self.time_windows = time_windows
        self.stat_windows = stat_windows
        self.contrasts = contrasts
        self.config = {}

    def configure_contrast(self, contrast, **kwargs):
        if not contrast in self.config:
            self.config[contrast] = {}
        self.config[contrast].update(kwargs)

    def configure_epoch(self, epoch, **kwargs):
        if not epoch in self.config:
            self.config[epoch] = {}
        self.config[epoch].update(kwargs)

    def markup(self, epoch, ax, left=True, bottom=True):
        for key, value in self.config[epoch].items():
            try:
                if (key.startswith('x')) and bottom:
                    getattr(ax, "set_" + key)(value)
                elif (key.startswith('y')) and left:
                    getattr(ax, "set_" + key)(value)
                elif key.startswith('x') or key.startswith('y'):
                    getattr(ax, "set_" + key)(value)
            except AttributeError:
                # print ('Can not set:', key)
                pass
        if not left:
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ylabel('')
        if not bottom:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_xlabel('')


example_config = PlotConfig(
    {"stimulus": (-0.35, 0.8), "response": (-0.35, 0.8)},  # Time windows for epochs
    ["all", "choice", "confidence", "confidence_asym", "hand", "stimulus", "stimulus_hi_con"],  # Contrast names
    stat_windows={"stimulus": (-0.35, 0.8), "response": (-1, 0.5)},
)

example_config.configure_epoch(
    "stimulus",
    **{
        "xticks": [0, 0.75],
        "xticklabels": ["0", ".75"],
        "yticks": [25, 50, 75, 100, 125, 150],
        "yticklabels": [25, 50, 75, 100, 125, 150],
        "xlabel": "Time",
        "ylabel": "Freq",
    },
)
example_config.configure_epoch(
    "response",
    **{
        "xticks": [0],
        "xticklabels": ["0"],
        "yticks": [25, 50, 75, 100, 125, 150],
        "yticklabels": [25, 50, 75, 100, 125, 150],
        "xlabel": "Time",
        "ylabel": "Freq",
    },
)
for key, values in {
    "all": {'vmin': -10, 'vmax': 10},
    "hand": {'vmin': -10, 'vmax': 10},
    "prev_hand": {'vmin': -10, 'vmax': 10},
    "stimulus": {'vmin': -25, 'vmax': 25},
    "0.81_coh": {'vmin': -10, 'vmax': 10},
}.items():
    example_config.configure_contrast(key, **values)


def plot_streams_fig(df, contrast_name, configuration, stats=False, cmap="RdBu_r", suffix=""):
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

    Plot = namedtuple("Plot", ["name", "cluster", "location", "annot_y", "annot_x"])

    top, middle, bottom = slice(0, 2), slice(1, 3), slice(2, 4)
    # fmt: off

    # layout = [
    #     Plot("V1", "vfcPrimary", [0, middle], True, True),
    #     Plot("V2-V4", "vfcEarly", [1, middle], False, True),
    #     # Dorsal
    #     Plot("V3ab", "vfcV3ab", [2, top], False, False),
    #     Plot("IPS0/1", "vfcIPS01", [3, top], False, False),
    #     Plot("IPS2/3", "vfcIPS23", [4, top], False, False),
    #     Plot("aIPS", "JWG_aIPS", [5, top], False, False),
    #     # Plot("FEF", "vfcFEF", [6, top], False, False),
    #     # Plot("dlPFC", "HCPMMP1_dlpfc", [7, top], False, False),
    #     Plot("FEF", "vfcFEF", [6, top], False, False),
    #     Plot("dlPFC", "HCPMMP1_dlpfc", [7, top], False, False),
    #     # Ventral
    #     Plot("Lateral Occ", "vfcLO", [2, bottom], False, True),
    #     Plot("MT+", "vfcTO", [3, bottom], False, True),
    #     Plot("Ventral Occ", "vfcVO", [4, bottom], False, True),
    #     Plot("PHC", "vfcPHC", [5, bottom], False, True),
    #     Plot("Insula", "HCPMMP1_insular_front_opercular", [6, bottom], False, True),
    #     Plot("vlPFC", "HCPMMP1_frontal_inferior", [7, bottom], False, True),
    #     Plot("PMd/v", "HCPMMP1_premotor", [8, middle], False, True),
    #     Plot("IPS Post-central", "JWG_IPS_PCeS", [9, middle], False, True),
    #     Plot("M1", "JWG_M1", [10, middle], False, True),
    #     ]

# #        Plot("Insula", "HCPMMP1_insular_front_opercular", [6, bottom], False, True),
# #        Plot("vlPFC", "HCPMMP1_frontal_inferior", [7, bottom], False, True),
# #        Plot("PMd/v", "HCPMMP1_premotor", [8, middle], False, True),
#         Plot("IPS Post-central", "JWG_IPS_PCeS", [6, middle], False, True),
#         Plot("M1", "JWG_M1", [7, middle], False, True),

    layout = [
        Plot("V1", "vfcPrimary", [0, middle], True, True),
        Plot("V2-V4", "vfcEarly", [1, middle], False, True),
        # Dorsal
        Plot("V3ab", "vfcV3ab", [2, top], False, False),
        Plot("IPS0/1", "vfcIPS01", [3, top], False, False),
        Plot("IPS2/3", "vfcIPS23", [4, top], False, False),
        Plot("aIPS", "JWG_aIPS", [5, top], False, False),
        # Plot("FEF", "vfcFEF", [6, top], False, False),
        # Plot("dlPFC", "HCPMMP1_dlpfc", [7, top], False, False),
        # Ventral
        Plot("Lateral Occ", "vfcLO", [2, bottom], False, True),
        Plot("MT+", "vfcTO", [3, bottom], False, True),
        Plot("Ventral Occ", "vfcVO", [4, bottom], False, True),
        Plot("PHC", "vfcPHC", [5, bottom], False, True),
#        Plot("Insula", "HCPMMP1_insular_front_opercular", [6, bottom], False, True),
#        Plot("vlPFC", "HCPMMP1_frontal_inferior", [7, bottom], False, True),
#        Plot("PMd/v", "HCPMMP1_premotor", [8, middle], False, True),
        Plot("IPS Post-central", "JWG_IPS_PCeS", [6, middle], False, True),
        Plot("M1", "JWG_M1", [7, middle], False, True),
    ]






#     layout = [
#         Plot("Oz", "Oz", [0, middle], True, True),
#         Plot("POz", "POz", [1, middle], False, True),
#         Plot("Pz", "Pz", [2, middle], False, True),
#         Plot("CPz", "CPz", [3, middle], False, True),
#         Plot("Cz", "Cz", [4, middle], False, True),
#         # Dorsal
#         Plot("O1/2", "O1/2", [0, top], False, False),
#         Plot("PO3/4", "PO3/4", [1, top], False, False),
#         Plot("P1/2", "P1/2", [2, top], False, False),
#         Plot("CP1/2", "CP1/2", [3, top], False, False),
#         Plot("C1/2", "C1/2", [4, top], False, False),
#  #       Plot("P3/4", "P3/4", [6, top], False, False),
#  #       Plot("P1/2", "P1/2", [7, top], False, False),
#         # Ventral
#         Plot("PO7/8", "PO7/8", [1, bottom], False, True),
#         Plot("P3/4", "P3/4", [2, bottom], False, True),
#         Plot("CP3/4", "CP3/4", [3, bottom], False, True),
#         Plot("C3/4", "C3/4", [4, bottom], False, True),
#  #       Plot("CP5/6", "CP5/6", [6, bottom], False, True),
#  #       Plot("Fz", "Fz", [7, bottom], False, True),
# #        Plot("CP3/4", "CP3/4", [6, middle], False, True),
# #        Plot("C3/4", "C3/4", [7, middle], False, True),
#     ]


    fig = plot_tfr_selected_rois(
        contrast_name, df, layout, configuration, cluster_correct=stats, cmap=cmap
    )
    return fig


def plot_tfr_selected_rois(
    contrast_name, tfrs, layout, conf=None, cluster_correct=False, cmap=None, vmin=-25, vmax=25, gs=None
):
    # nr_cols = np.max([p.location[0] for p in layout]) + 1
    # if gs is None:
    #     fig = plt.figure(figsize=(nr_cols * 1.5, 3.5))
    # else:
    #     fig = plt.gcf()
    # ratio = np.diff(conf.time_windows['response'])[0] / np.diff(conf.time_windows['stimulus'])[0]

    # if gs is None:
    #     gs = matplotlib.gridspec.GridSpec(4, (nr_cols * 3), width_ratios=list(np.tile([1, ratio, 0.1], nr_cols)))
    # #   gs.update(wspace=0.05, hspace=0.5)
    # else:
    #     gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
    #         4,
    #         (nr_cols * 3),
    #         width_ratios=list(np.tile([1, ratio, 0.1], nr_cols)),
    #         subplot_spec=gs,
    #         wspace=0.05,
    #         hspace=0.5,
    #     )
    nr_cols = np.max([p.location[0] for p in layout]) + 1
    fig = plt.figure(figsize=(nr_cols * 1.5, 3.5))
    ratio = np.diff(conf.time_windows['response'])[0] / np.diff(conf.time_windows['stimulus'])[0]

    gs = matplotlib.gridspec.GridSpec(4, (nr_cols * 3), width_ratios=list(np.tile([3, ratio, 0.1], nr_cols)))

    for P in layout:
        for j, timelock in enumerate(["stimulus"]):
            cluster = P.cluster
            print(cluster)
            cluster_name = P.name
            # tfr to plot:
            tfr = tfrs.query('contrast=="%s" & cluster=="%s" & epoch=="%s"' % (contrast_name, cluster, timelock))
            # pdb.set_trace()
            if len(tfr) == 0:
                print('No TFR for ', contrast_name, cluster, timelock)
                continue
            col, row = P.location
            col = col * 3 + j
            ax = plt.subplot(gs[row, col])

            time_cutoff = conf.time_windows[timelock]  # (-0.2, 1.1)
            if conf.stat_windows is None:
                stat_cutoff = time_cutoff
            else:
                stat_cutoff = conf.stat_windows[timelock]  # (-0.5, 1.35)

            try:
                limits = conf.config[contrast_name]
                vmin, vmax = limits['vmin'], limits['vmax']
            except KeyError:
                pass

            print(cluster_correct)
            if P.name == "C3/4":
                plot_tfr(
                    tfr,
                    time_cutoff,
                    vmin,
                    vmax,
                    timelock,
                    cluster_correct=cluster_correct,
                    threshold=0.05,
                    plot_colorbar=True,
                    ax=ax,
                    cmap=cmap,
                    stat_cutoff=stat_cutoff,
                )
            else:
                plot_tfr(
                    tfr,
                    time_cutoff,
                    vmin,
                    vmax,
                    timelock,
                    cluster_correct=cluster_correct,
                    threshold=0.05,
                    plot_colorbar=False,
                    ax=ax,
                    cmap=cmap,
                    stat_cutoff=stat_cutoff,
                )

            conf.markup(timelock, ax, left=P.annot_y, bottom=P.annot_x)
            if j == 1:
                # ax.set_xticks([])
                # ax.set_xticklabels([])
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_ylabel('')
                ax.set_xlabel('')
            if j == 0:
                plt.title(cluster_name, {"fontsize": 8})

    sns.despine()
    return fig


def plot_tfr(
    tfr,
    time_cutoff,
    vmin,
    vmax,
    tl,
    cluster_correct=False,
    threshold=0.05,
    plot_colorbar=False,
    ax=None,
    cmap=None,
    stat_cutoff=None,
):
    #   from pymeg.contrast_tfr import get_tfr_stats

    @memory.cache()
    def get_tfr_stats(times, freqs, tfr, threshold=0.05, n_jobs=2):
        from mne.stats import permutation_cluster_1samp_test as cluster_test
        import joblib

        return {
            joblib.hash([times, freqs, tfr, threshold]): cluster_test(
                tfr, threshold={"start": 0, "step": 0.2}, tail=0, n_permutations=1000, n_jobs=n_jobs
            )
        }

    # colorbar:
    from matplotlib.colors import LinearSegmentedColormap

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("custom", ["blue", "lightblue", "lightgrey", "yellow", "red"], N=100)

    if stat_cutoff is None:
        stat_cutoff = time_cutoff

    # data:
    times, freqs, X = contrast_tfr.get_tfr(tfr, stat_cutoff)
    mask = None
    if cluster_correct:
        hash = joblib.hash([times, freqs, X, threshold])
        # try:
        #     _, _, cluster_p_values, _ = cluster_correct[hash]
        #     sig = cluster_p_values.reshape((X.shape[1], X.shape[2]))
        #     mask = sig < threshold
        # except KeyError:
        s = get_tfr_stats(times, freqs, X, threshold)
        _, _, cluster_p_values, _ = s[hash]
        sig = cluster_p_values.reshape((X.shape[1], X.shape[2]))
        mask = sig < threshold

    cax = pmi(
        plt.gca(),
        np.nanmean(X, 0),
        times,
        yvals=freqs,
        yscale="linear",
        vmin=vmin,
        vmax=vmax,
        mask=mask,
        mask_alpha=1,
        mask_cmap=cmap,
        cmap=cmap,
    )
    plt.xlim(time_cutoff)
    plt.ylim([freqs.min() - 0.5, freqs.max() + 0.5])
    ax.axvline(0, ls="--", lw=0.75, color="black")
    ax.axvline(0.75, ls="--", lw=0.75, color="black")
    if plot_colorbar:
        plt.colorbar(cax, ticks=[vmin, 0, vmax])
    return ax


def pmi(*args, **kwargs):
    cax = _plot_masked_image(*args, **kwargs)
    return cax


def plot_mosaic(tfr_data, vmin=-25, vmax=25, cmap="RdBu_r", ncols=4, epoch="stimulus", stats=False, threshold=0.05):
    if epoch == "stimulus":
        time_cutoff = (-0.5, 1.35)
        xticks = [0, 0.25, 0.5, 0.75, 1]
        xticklabels = ["0\nStim on", "", ".5", "", "1\nStim off"]
        yticks = [25, 50, 75, 100, 125]
        yticklabels = ["25", "", "75", "", "125"]
        xmarker = [0, 1]
        baseline = (-0.25, 0)
    else:
        time_cutoff = (-1, 0.5)
        xticks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5]
        xticklabels = ["-1", "", "-0.5", "", "0\nResponse", "", "0.5"]
        yticks = [1, 25, 50, 75, 100, 125]
        yticklabels = ["1", "25", "", "75", "", "125"]
        xmarker = [0, 1]
        baseline = None
    from matplotlib import gridspec
    import pylab as plt
    import seaborn as sns

    contrast_tfr.set_jw_style()
    sns.set_style("ticks")
    nrows = (len(atlas_glasser.areas) // ncols) + 1
    gs = gridspec.GridSpec(nrows, ncols)
    gs.update(wspace=0.01, hspace=0.01)

    for i, (name, area) in enumerate(atlas_glasser.areas.items()):
        try:
            column = np.mod(i, ncols)
            row = i // ncols
            plt.subplot(gs[row, column])
            times, freqs, tfr = get_tfr(tfr_data.query('cluster=="%s"' % area), time_cutoff)
            # cax = plt.gca().pcolormesh(times, freqs, np.nanmean(
            #    tfr, 0), vmin=vmin, vmax=vmax, cmap=cmap, zorder=-2)
            mask = None

            if stats:
                import joblib

                hash = joblib.hash([times, freqs, tfr, threshold])
                try:
                    _, _, cluster_p_values, _ = stats[hash]
                except KeyError:
                    s = get_tfr_stats(times, freqs, tfr, threshold)
                    _, _, cluster_p_values, _ = s[hash]
                sig = cluster_p_values.reshape((tfr.shape[1], tfr.shape[2]))
                mask = sig < threshold
            cax = pmi(
                plt.gca(),
                np.nanmean(tfr, 0),
                times,
                yvals=freqs,
                yscale="linear",
                vmin=vmin,
                vmax=vmax,
                mask=mask,
                mask_alpha=1,
                mask_cmap=cmap,
                cmap=cmap,
            )

            # plt.grid(True, alpha=0.5)
            for xmark in xmarker:
                plt.axvline(xmark, color="k", lw=1, zorder=-1, alpha=0.5)

            plt.yticks(yticks, [""] * len(yticks))
            plt.xticks(xticks, [""] * len(xticks))
            set_title(name, times, freqs, plt.gca())
            plt.tick_params(direction="inout", length=2, zorder=100)
            plt.xlim(time_cutoff)
            plt.ylim([1, 147.5])
            plt.axhline(10, color="k", lw=1, alpha=0.5, linestyle="--")
        except ValueError as e:
            print(name, area, e)
    plt.subplot(gs[nrows - 2, 0])

    sns.despine(left=True, bottom=True)
    plt.subplot(gs[nrows - 1, 0])

    pmi(
        plt.gca(),
        np.nanmean(tfr, 0) * 0,
        times,
        yvals=freqs,
        yscale="linear",
        vmin=vmin,
        vmax=vmax,
        mask=None,
        mask_alpha=1,
        mask_cmap=cmap,
        cmap=cmap,
    )
    plt.xticks(xticks, xticklabels)
    plt.yticks(yticks, yticklabels)
    for xmark in xmarker:
        plt.axvline(xmark, color="k", lw=1, zorder=-1, alpha=0.5)
    if baseline is not None:
        plt.fill_between(baseline, y1=[1, 1], y2=[150, 150], color="k", alpha=0.5)
    plt.tick_params(direction="in", length=3)
    plt.xlim(time_cutoff)
    plt.ylim([1, 147.5])
    plt.xlabel("time [s]")
    plt.ylabel("Freq [Hz]")
    sns.despine(ax=plt.gca())


def plot_epoch_pair(tfr_data, vmin=-25, vmax=25, cmap="RdBu_r", gs=None, stats=False, threshold=0.05, ylabel=None):
    from matplotlib import gridspec
    import pylab as plt
    import joblib

    if gs is None:
        g = gridspec.GridSpec(1, 2)
    else:
        g = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs, wspace=0.01, width_ratios=[1, 0.4])
    times, freq, tfr = None, None, None
    for epoch in ["stimulus", "response"]:
        row = 0
        if epoch == "stimulus":
            col = 0
            time_cutoff = (-0.35, 1.1)
            xticks = [0, 0.25, 0.5, 0.75, 1]
            yticks = [25, 50, 75, 100, 125]
            xmarker = [0, 1]
        else:
            col = 1
            time_cutoff = (-0.35, 0.1)
            xticks = [0]
            yticks = [1, 25, 50, 75, 100, 125]
            xmarker = [0, 1]

        plt.subplot(g[row, col])
        tdata = tfr_data.query('epoch=="%s"' % (epoch))
        if len(tdata) == 0:
            plt.yticks([], [""])
            plt.xticks([], [""])
            continue
        times, freqs, tfr = get_tfr(tdata, time_cutoff)

        mask = None
        if stats:
            hash = joblib.hash([times, freqs, tfr, threshold])
            try:
                _, _, cluster_p_values, _ = stats[hash]
            except KeyError:
                s = get_tfr_stats(times, freqs, tfr, threshold)
                _, _, cluster_p_values, _ = s[hash]

            sig = cluster_p_values.reshape((tfr.shape[1], tfr.shape[2]))
            mask = sig < threshold

        _ = pmi(
            plt.gca(),
            np.nanmean(tfr, 0),
            times,
            yvals=freqs,
            yscale="linear",
            vmin=vmin,
            vmax=vmax,
            mask=mask,
            mask_alpha=1,
            mask_cmap=cmap,
            cmap=cmap,
        )
        if (ylabel is not None) and (epoch == "stimulus"):
            plt.ylabel(ylabel, labelpad=-2, fontdict={"fontsize": 4})
        # for xmark in xmarker:
        #    plt.axvline(xmark, color='k', lw=1, zorder=-1, alpha=0.5)

        plt.yticks(yticks, [""] * len(yticks))
        plt.xticks(xticks, [""] * len(xticks))

        plt.tick_params(direction="inout", length=2, zorder=100)
        plt.xlim(time_cutoff)
        plt.ylim([1, 147.5])
        # plt.axhline(10, color='k', lw=1, alpha=0.5, linestyle='--')
        # plt.axhline(25, color='k', lw=1, alpha=0.5, linestyle=':')
        # plt.axhline(50, color='k', lw=1, alpha=0.5, linestyle=':')
        plt.axvline(0, color="k", lw=1, zorder=5, alpha=0.5)
        if epoch == "stimulus":
            plt.axvline(1, color="k", lw=1, zorder=5, alpha=0.5)
    return times, freqs, tfr


def plot_2epoch_mosaic(tfr_data, vmin=-25, vmax=25, cmap="RdBu_r", ncols=8, stats=False, threshold=0.05):
    from matplotlib import gridspec

    # ncols *= 2
    set_jw_style()
    sns.set_style("ticks")
    areas = tfr_data.index.get_level_values("cluster").unique()
    nrows = int((len(areas) // (ncols)) + 1)
    gs = gridspec.GridSpec(nrows, ncols)
    # gs.update(wspace=0.001)#, hspace=0.05)
    i = 0
    for area in areas:
        col = int(np.mod(i, ncols))
        row = int(i // ncols)
        cluster_data = tfr_data.query('cluster=="%s"' % area)

        times, freqs, tfr = plot_epoch_pair(
            cluster_data,
            vmin=-25,
            vmax=25,
            cmap=cmap,
            gs=gs[row, col],
            stats=stats,
            ylabel=area.replace("NSWFRONT_", "").replace("HCPMMP1_", ""),
        )
        i += 1


# def plot_cluster(names, view):
#     from pymeg import atlas_glasser
#
#     all_clusters, _, _, _ = atlas_glasser.get_clusters()
#     label_names = []
#     for name in names:
#         cluster_name = atlas_glasser.areas[name]
#         label_names.extend(all_clusters[cluster_name])
#
#     plot_roi("lh", label_names, "r")


# def plot_rois(
#     labels,
#     clusters,
#     view="lat",
#     fs_dir=os.environ["SUBJECTS_DIR"],
#     subject_id="S04",
#     surf="inflated",
# ):
#     hemi = "lh"
#     from surfer import Brain
#     import seaborn as sns
#
#     colors = sns.color_palette("husl", len(clusters))
#     brain = Brain(subject_id, hemi, surf, offscreen=False)
#     for color, cluster in zip(colors, clusters):
#         for label in labels[cluster]:
#
#             if "rh" in label.name:
#                 continue
#             # print(cluster, label)
#             brain.add_label(label, color=color)
#     brain.show_view(view)
#     return brain, brain.screenshot()
#
#
# # def plot_roi(
# #     hemi,
# #     labels,
# #     color,
# #     annotation="HCPMMP1",
# #     view="parietal",
# #     fs_dir=os.environ["SUBJECTS_DIR"],
# #     subject_id="S04",
# #     surf="inflated",
# # ):
# #     import matplotlib
# #     import os
# #     import glob
# #     from surfer import Brain
# #     from mne import Label
# #
# #     color = np.array(matplotlib.colors.to_rgba(color))
# #
# #     brain = Brain(subject_id, hemi, surf, offscreen=False)
# #     labels = [label.replace("-rh", "").replace("-lh", "") for label in labels]
# #     # First select all label files
# #
# #     label_names = glob.glob(os.path.join(fs_dir, subject_id, "label", "lh*.label"))
# #     label_names = [label for label in label_names if any([l in label for l in labels])]
# #
# #     for label in label_names:
# #         brain.add_label(label, color=color)
# #
# #     # Now go for annotations
# #     from nibabel.freesurfer import io
# #
# #     ids, colors, annot_names = io.read_annot(
# #         os.path.join(fs_dir, subject_id, "label", "lh.%s.annot" % annotation),
# #         orig_ids=True,
# #     )
# #
# #     for i, alabel in enumerate(annot_names):
# #         if any([label in alabel.decode("utf-8") for label in labels]):
# #             label_id = colors[i, -1]
# #             vertices = np.where(ids == label_id)[0]
# #             l = Label(np.sort(vertices), hemi="lh")
# #             brain.add_label(l, color=color)
# #     brain.show_view(view)
# #     return brain.screenshot()
#


def _plot_masked_image(
    ax,
    data,
    times,
    mask=None,
    yvals=None,
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    ylim=None,
    mask_style="both",
    mask_alpha=0.25,
    mask_cmap="Greys",
    **kwargs,
):
    from matplotlib import ticker

    if yvals is None:  # for e.g. Evoked images
        yvals = np.arange(data.shape[0])
    ratio = yvals[1:] / yvals[:-1]
    # compute bounds between time samples
    time_diff = np.diff(times) / 2.0 if len(times) > 1 else [0.0005]
    time_lims = np.concatenate([[times[0] - time_diff[0]], times[:-1] + time_diff, [times[-1] + time_diff[-1]]])

    log_yvals = np.concatenate([[yvals[0] / ratio[0]], yvals, [yvals[-1] * ratio[0]]])
    yval_lims = np.sqrt(log_yvals[:-1] * log_yvals[1:])

    # construct a time-yvaluency bounds grid
    time_mesh, yval_mesh = np.meshgrid(time_lims, yval_lims)

    if mask is not None:
        im = ax.pcolormesh(time_mesh, yval_mesh, data, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
        if mask.sum() > 0:
            big_mask = np.kron(mask, np.ones((10, 10)))
            big_times = np.kron(times, np.ones((10,)))
            big_yvals = np.kron(yvals, np.ones((10,)))
            ax.contour(
                big_times,
                big_yvals,
                big_mask,
                colors=["k"],
                linewidths=[0.75],
                corner_mask=False,
                antialiased=False,
                levels=[0.5],
            )
    else:
        im = ax.pcolormesh(time_mesh, yval_mesh, data, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    if ylim is None:
        ylim = yval_lims[[0, -1]]

    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    # get rid of minor ticks
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    tick_vals = yvals[np.unique(np.linspace(0, len(yvals) - 1, 12).round().astype("int"))]
    ax.set_yticks(tick_vals)

    ax.set_xlim(time_lims[0], time_lims[-1])
    ax.set_ylim(ylim)

    return im


def set_title(text, times, freqs, axes):
    import pylab as plt

    x = np.min(times)
    y = np.max(freqs) + 20
    plt.text(x, y, text, fontsize=8, verticalalignment="top", horizontalalignment="center")


def set_jw_style():
    import matplotlib
    import seaborn as sns

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    sns.set(
        style="ticks",
        font="Arial",
        font_scale=1,
        rc={
            "axes.linewidth": 0.05,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "xtick.major.width": 0.25,
            "xtick.minor.width": 0.25,
            "ytick.major.width": 0.25,
            "text.color": "Black",
            "axes.labelcolor": "Black",
            "xtick.color": "Black",
            "ytick.color": "Black",
        },
    )
    sns.plotting_context()
