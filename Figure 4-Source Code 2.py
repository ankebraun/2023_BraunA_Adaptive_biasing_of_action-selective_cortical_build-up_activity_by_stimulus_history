"""
Plot impact of single-trial history bias as well as of signed stimulus strength
on amplitude and slope of M1 beta lateralization
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from mne.stats import permutation_cluster_1samp_test
import seaborn as sns
from mpl_axes_aligner import align


regr_beta_slope_bias_pooled_resp = pd.read_csv(
    "Figure 4-Source Data 3.csv", sep="\t", index_col=0
)
regr_beta_slope_bias_pooled_resp = regr_beta_slope_bias_pooled_resp.set_index(
    ["subject"]
)
regr_beta_slope_signed_stim_pooled_resp = pd.read_csv(
    "Figure 4-Source Data 4.csv", sep="\t", index_col=0
)
regr_beta_slope_signed_stim_pooled_resp = (
    regr_beta_slope_signed_stim_pooled_resp.set_index(["subject"])
)

df_pooled_bias = pd.read_csv("Figure 4-Source Data 5.csv", sep="\t", index_col=0)
df_pooled_bias = df_pooled_bias.set_index(["subject"])
df_pooled_signed_stim = pd.read_csv("Figure 4-Source Data 6.csv", sep="\t", index_col=0)
df_pooled_signed_stim = df_pooled_signed_stim.set_index(["subject"])


def figure_beta_lat_weights_pooled(df_slope, df_lat, regressor, pos):
    plt.figure(figsize=(2.5, 2.5))
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax2 = ax1.twinx()
    plt.subplots_adjust(left=0.35, bottom=0.2, top=0.9, right=0.8, wspace=1, hspace=1.0)

    times = df_lat.columns.values
    times = np.array(times, dtype=float)
    max_time = min(np.where(times >= 0.75)[0]) - 1
    times = times[:max_time]
    mean_pooled_per_sj = df_lat.values
    mean_pooled_per_sj = mean_pooled_per_sj[:, :max_time]
    # average across subjects
    mean_pooled = np.mean(mean_pooled_per_sj, axis=0)
    sem_pooled = stats.sem(mean_pooled_per_sj, axis=0)

    ax1.plot(
        [min(times), max(times)], [0, 0], linestyle="--", color="grey", linewidth=0.5
    )
    ax1.plot(times, mean_pooled, color="k", label="mean(up, -down)")
    ax1.fill_between(
        times, mean_pooled - sem_pooled, mean_pooled + sem_pooled, color="k", alpha=0.5
    )
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        mean_pooled_per_sj, threshold=None, tail=0, n_permutations=1000
    )

    y_min = min(mean_pooled - sem_pooled)
    if regressor == "single trial bias":
        y_min = -0.064
    elif regressor == "single trail signed stim":
        y_min = -0.22
    for i_c, c in enumerate(clusters):
        if cluster_p_values[i_c] <= 0.05:
            print([times[c[0][0]], times[c[0][-1]]])
            ax1.plot([times[c[0][0]], times[c[0][-1]]], [y_min, y_min], color="k")

    times = df_lat.columns.values
    times = np.array(times, dtype=float)
    times_ = times + 0.1  # shift by 100ms to get the center of the sliding window
    max_time = min(np.where(times_ >= 0.75)[0]) - 1
    times_ = times_[:max_time]

    mean_pooled_per_sj_slope = df_slope.values
    mean_pooled_per_sj_slope = mean_pooled_per_sj_slope[:, :max_time]
    # average across subjects
    mean_pooled_slope = np.mean(mean_pooled_per_sj_slope, axis=0)
    sem_pooled_slope = stats.sem(mean_pooled_per_sj_slope, axis=0)

    ax2.plot(times_, mean_pooled_slope, color="darkgrey", label="mean(up, -down)")
    ax2.fill_between(
        times_,
        mean_pooled_slope - sem_pooled_slope,
        mean_pooled_slope + sem_pooled_slope,
        color="darkgrey",
        alpha=0.5,
    )
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        mean_pooled_per_sj_slope, threshold=None, tail=0, n_permutations=1000
    )

    if regressor == "single trial bias":
        y_min = -0.022
    elif regressor == "single trial signed stim":
        y_min = -0.08

    for i_c, c in enumerate(clusters):
        if cluster_p_values[i_c] <= 0.05:
            print([times_[c[0][0]], times_[c[0][-1]]])
            ax2.plot(
                [times_[c[0][0]], times_[c[0][-1]]], [y_min, y_min], color="darkgrey"
            )

    sns.despine(ax=ax1, offset=10, right=True, left=False)

    ax1.set_xlabel("Time from evidence onset in s")
    ax1.set_ylabel("Beta weights M1 beta lat. vs." + regressor)
    ax2.set_ylabel("Beta weights M1 beta slope vs." + regressor)

    if regressor == "single trial signed stim":
        ax1.set_ylim(-0.2, 0.03)
        ax2.set_ylim(-0.09, 0.01)
    elif regressor == "single trial bias":
        ax2.set_ylim(-0.023, 0.013)
        ax1.set_ylim(-0.065, 0.02)
    # Adjust the plotting range of two y axes
    org1 = 0.0  # Origin of first axis
    org2 = 0.0  # Origin of second axis
    align.yaxes(ax1, org1, ax2, org2, pos)

    plt.savefig("Regression_beta_lat_slope_pooled_double_y_" + regressor + ".pdf")
    plt.show()


figure_beta_lat_weights_pooled(
    regr_beta_slope_bias_pooled_resp, df_pooled_bias, "single trial bias", 0.65
)

figure_beta_lat_weights_pooled(
    regr_beta_slope_signed_stim_pooled_resp,
    df_pooled_signed_stim,
    "single trial signed stim",
    0.85,
)
