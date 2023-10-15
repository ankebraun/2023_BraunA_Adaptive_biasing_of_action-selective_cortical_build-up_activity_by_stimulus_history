"""
Plot beta lateralization separately for current correct and error response locked to 
stimulus onset and locked to response
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
from mne.stats import permutation_cluster_1samp_test
from scipy import stats

matplotlib.rcParams["pdf.fonttype"] = 42
sns.set(
    style="ticks",
    font="Helvetica",
    font_scale=1,
    rc={
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.25,
        "xtick.major.width": 0.25,
        "ytick.major.width": 0.25,
        "text.color": "Black",
        "axes.labelcolor": "Black",
        "xtick.color": "Black",
        "ytick.color": "Black",
        "font.sans-serif": ["Helvetica"],
        "xtick.major.pad": 1,
        "ytick.major.pad": 1,
    },
)

sns.plotting_context()

fullwidth = 6.3
halfwidth = 0.45 * fullwidth
laby = 1.01
thlev = 0.85
sns.plotting_context()

fullwidth = 6.3
halfwidth = 0.45 * fullwidth
laby = 1.01
thlev = 0.85

pl.rcParams["legend.fontsize"] = "small"
pl.rcParams["legend.fontsize"] = "small"

list_lh_accuracy = pd.read_hdf("Figure 2-Figure Supplement 2-Source Data 1.hdf")


def get_data_for_plot(tfr, cluster):

    tfr_beta = (
        tfr.query(f'cluster == "{cluster}" & 12 <= freq <= 36')
        .groupby("subject")
        .mean()
    )
    tfr_beta_reshaped = np.array(
        [
            (tfr_beta.loc[tfr_beta.index.isin([subj], level="subject")].values)
            for subj in np.unique(tfr_beta.index.get_level_values("subject"))
        ]
    )
    k, l, m = np.shape(tfr_beta_reshaped)
    tfr_beta_reshaped = tfr_beta_reshaped.reshape(k, m)
    return tfr_beta, tfr_beta_reshaped  # , clusters, cluster_p_values


def figure_beta_timecourses_current_accuracy(
    cluster, contrast_correct, contrast_error, fname
):
    plt.figure(figsize=(7, 5))
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    axes = [ax1, ax2]  # , ax3]

    plt.subplots_adjust(
        left=0.1, bottom=0.15, top=0.85, right=0.9, wspace=0.5, hspace=0.7
    )

    start_plot = np.min(
        np.where(list_lh_accuracy.columns >= -0.2275)
    )  # start to plot from mean of baseline
    stop_plot = np.min(np.where(list_lh_accuracy.columns > 0.75))  

    start_plot_resp_locked = np.min(
        np.where(list_lh_accuracy.columns >= -0.35)
    )  
    stop_plot_resp_locked = np.min(np.where(list_lh_accuracy.columns > 0.1))

    def comp_data(contrast, cluster, start_plot, stop_plot, epo):
        data_contrast = list_lh_accuracy.query(
            f'epoch == "{epo}" & ~(hemi == "avg") & (contrast == "{contrast}")'
        )
        tfr_beta, tfr_beta_reshaped = get_data_for_plot(data_contrast, cluster)
        residual = []
        for i in range(0, len(tfr_beta_reshaped)):
            residual.append(tfr_beta_reshaped[i][start_plot:stop_plot])
        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
            np.array(residual),
            threshold={"start": 0, "step": 0.2},
            tail=0,
            n_permutations=1000,
            out_type="mask",
        )
        return residual, clusters, cluster_p_values

    axes[0].plot([0, 0], [-15, 0], color="grey", linestyle="--", linewidth=0.5)
    axes[1].plot([0, 0], [-15, 0], color="grey", linestyle="--", linewidth=0.5)

    axes[0].plot(
        [list_lh_accuracy.columns[start_plot], list_lh_accuracy.columns[stop_plot]],
        [0, 0],
        color="grey",
        linestyle="--",
        linewidth=0.5,
    )
    axes[1].plot(
        [
            list_lh_accuracy.columns[start_plot_resp_locked],
            list_lh_accuracy.columns[stop_plot_resp_locked],
        ],
        [0, 0],
        color="grey",
        linestyle="--",
        linewidth=0.5,
    )

    residual_correct, clusters_correct, cluster_p_values_correct = comp_data(
        contrast_correct, cluster, start_plot, stop_plot, "stimulus"
    )
    residual_error, clusters_error, cluster_p_values_error = comp_data(
        contrast_error, cluster, start_plot, stop_plot, "stimulus"
    )

    (
        residual_correct_resp_locked,
        clusters_correct_resp_locked,
        cluster_p_values_correct_resp_locked,
    ) = comp_data(
        contrast_correct,
        cluster,
        start_plot_resp_locked,
        stop_plot_resp_locked,
        "response",
    )
    (
        residual_error_resp_locked,
        clusters_error_resp_locked,
        cluster_p_values_error_resp_locked,
    ) = comp_data(
        contrast_error,
        cluster,
        start_plot_resp_locked,
        stop_plot_resp_locked,
        "response",
    )

    (
        T_obs_rep,
        clusters_rep,
        cluster_p_values_rep,
        H0_rep,
    ) = permutation_cluster_1samp_test(
        np.array(residual_correct) - np.array(residual_error),
        threshold={"start": 0, "step": 0.2},
        tail=0,
        n_permutations=1000,
        out_type="mask",
    )
    (
        T_obs_rep_resp_locked,
        clusters_rep_resp_locked,
        cluster_p_values_rep_resp_locked,
        H0_rep_resp_locked,
    ) = permutation_cluster_1samp_test(
        np.array(residual_correct_resp_locked) - np.array(residual_error_resp_locked),
        threshold={"start": 0, "step": 0.2},
        tail=0,
        n_permutations=1000,
        out_type="mask",
    )

    times = list_lh_accuracy.columns[start_plot : stop_plot + 1]
    times_resp_locked = list_lh_accuracy.columns[
        start_plot_resp_locked : stop_plot_resp_locked + 1
    ]

    for i_c, c in enumerate(clusters_correct):
        if cluster_p_values_correct[i_c] <= 0.05:
            ax1.plot(
                [times[c.start], times[c.stop]],
                [-15.5, -15.5],
                marker="_",
                color="darkgrey",
                ms=3,
                zorder=1,
                markeredgecolor="w",
                markeredgewidth=0.1,
            )
    for i_c, c in enumerate(clusters_error):
        if cluster_p_values_error[i_c] <= 0.05:
            ax1.plot(
                [times[c.start], times[c.stop]],
                [-16.5, -16.5],
                marker="_",
                color="lightgrey",
                ms=3,
                zorder=1,
                markeredgecolor="w",
                markeredgewidth=0.1,
            )

    for i_c, c in enumerate(clusters_rep):
        if cluster_p_values_rep[i_c] <= 0.05:
            ax1.plot(
                [times[c.start], times[c.stop]],
                [-17.5, -17.5],
                marker="_",
                color="k",
                ms=3,
                zorder=1,
                markeredgecolor="w",
                markeredgewidth=0.1,
            )

    for i_c, c in enumerate(clusters_correct_resp_locked):
        if cluster_p_values_correct_resp_locked[i_c] <= 0.05:
            ax2.plot(
                [times_resp_locked[c.start], times_resp_locked[c.stop]],
                [-15.5, -15.5],
                marker="_",
                color="darkgrey",
                ms=3,
                zorder=1,
                markeredgecolor="w",
                markeredgewidth=0.1,
            )
    for i_c, c in enumerate(clusters_error_resp_locked):
        if cluster_p_values_error_resp_locked[i_c] <= 0.05:
            ax2.plot(
                [times_resp_locked[c.start], times_resp_locked[c.stop]],
                [-16.5, -16.5],
                marker="_",
                color="lightgrey",
                ms=3,
                zorder=1,
                markeredgecolor="w",
                markeredgewidth=0.1,
            )

    for i_c, c in enumerate(clusters_rep_resp_locked):
        if cluster_p_values_rep_resp_locked[i_c] <= 0.05:
            ax2.plot(
                [times_resp_locked[c.start], times_resp_locked[c.stop]],
                [-17.5, -17.5],
                marker="_",
                color="k",
                ms=3,
                zorder=1,
                markeredgecolor="w",
                markeredgewidth=0.1,
            )

    def plot_timecourses(j, residual, col, lab, start_plot, stop_plot):
        axes[j].plot(
            list_lh_accuracy.columns[start_plot:stop_plot],
            np.mean(residual, axis=0),
            color=col,
            label=lab,
        )
        axes[j].fill_between(
            list_lh_accuracy.columns[start_plot:stop_plot],
            np.mean(residual, axis=0) + stats.sem(residual, axis=0),
            np.mean(residual, axis=0) - stats.sem(residual, axis=0),
            facecolor=col,
            alpha=0.5,
        )

    plot_timecourses(
        j=0,
        residual=residual_correct,
        col="darkgrey",
        lab="Correct",
        start_plot=start_plot,
        stop_plot=stop_plot,
    )
    plot_timecourses(
        j=0,
        residual=residual_error,
        col="lightgrey",
        lab="Error",
        start_plot=start_plot,
        stop_plot=stop_plot,
    )

    plot_timecourses(
        j=1,
        residual=residual_correct_resp_locked,
        col="darkgrey",
        lab="Correct",
        start_plot=start_plot_resp_locked,
        stop_plot=stop_plot_resp_locked,
    )
    plot_timecourses(
        j=1,
        residual=residual_error_resp_locked,
        col="lightgrey",
        lab="Error",
        start_plot=start_plot_resp_locked,
        stop_plot=stop_plot_resp_locked,
    )

    plt.suptitle(cluster)
    for j in range(0, 2):
        sns.despine(ax=axes[j], offset=10, right=True, left=False)
        axes[j].legend(
            bbox_to_anchor=(0.4, 1.0),
            loc=2,
            borderaxespad=0.0,
            fontsize=7,
            frameon=True,
        )

    ax1.set_xlabel("Time around stimulus onset in s")
    ax1.set_ylabel("% Power change Beta (12 - 36 Hz)")
    ax2.set_xlabel("Time around response onset in s")
    ax1.set_xticks(np.arange(0.0, 1.0, 0.25))
    ax2.set_xticks(np.arange(-0.3, 0.2, 0.1))
    ax1.set_yticks(np.arange(-20, 5, 5))
    ax2.set_yticks(np.arange(-20, 5, 5))
    ax1.set_ylim(-18, 1)
    ax2.set_ylim(-18, 1)
    pl.savefig(fname + cluster + ".pdf")
    plt.show()


figure_beta_timecourses_current_accuracy(
    cluster="JWG_M1",
    contrast_correct="hand_current_correct",
    contrast_error="hand_current_error",
    fname="beta_lat_current_corr_vs_err",
)
