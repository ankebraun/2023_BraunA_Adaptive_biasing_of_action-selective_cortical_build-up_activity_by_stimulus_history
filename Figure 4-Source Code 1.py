"""
Plot component of action selective lateralization goverened by single-trial bias
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
from mne.stats import permutation_cluster_1samp_test
from scipy import stats

from stats import permutation_test
from stats.stars import stars

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

list_all_sessions = pd.read_hdf("Figure 4-Source Data 1.hdf")
list_lh_baseline_per_session_bin_on_bias_bins_subsample = pd.read_hdf(
    "Figure 4-Source Data 2.hdf"
)


def contrast_lateralized(epoch, list_of_data_lh, condition):
    c_lh = list_of_data_lh.query(
        f'epoch == "{epoch}" & ~(hemi == "avg") & contrast == "{condition}"'
    ).dropna(axis=1, how="all")
    return c_lh


def get_data_for_plot_stats(tfr, cluster):
    if cluster == "aIPS + M1":
        tfr_cluster = (
            tfr.query('cluster == "JWG_M1" or cluster == "JWG_aIPS"')
            .groupby(["freq", "subject"])
            .mean()
        )
        tfr_beta = tfr_cluster.query("12 <= freq <= 36").groupby("subject").mean()
    else:
        tfr_beta = (
            tfr.query(f'cluster == "{cluster}" & 12 <= freq <= 36')
            .groupby("subject")
            .mean()
        )
    tfr_beta_reshaped = np.array(
        [
            (tfr_beta.loc[tfr_beta.index.isin([subj], level="subject")].to_numpy())
            for subj in np.unique(tfr_beta.index.get_level_values("subject"))
        ]
    )
    k, l, m = np.shape(tfr_beta_reshaped)
    tfr_beta_reshaped = tfr_beta_reshaped.reshape(k, m)
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        tfr_beta_reshaped,
        threshold={"start": 0, "step": 0.2},
        tail=0,
        n_permutations=1000,
        out_type="mask",
    )
    return tfr_beta, clusters, cluster_p_values


def beta_slope_bilinear_fit_methods():
    plt.figure(figsize=(4, 4))
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    plt.subplots_adjust(left=0.2, bottom=0.2, top=0.9, right=0.9, wspace=1, hspace=1.0)
    hand = contrast_lateralized(
        epoch="stimulus", list_of_data_lh=list_all_sessions, condition="hand"
    )

    tfr_beta, clusters, cluster_p_values = get_data_for_plot_stats(hand, "JWG_M1")

    def get_pwlf(x, y):
        import pwlf

        myPWLF = pwlf.PiecewiseLinFit(x, y)
        res = myPWLF.fit(2)
        slopes = myPWLF.calc_slopes()
        xHat = np.linspace(np.min(np.array(x)), np.max(np.array(x)), num=10000)
        yHat = myPWLF.predict(xHat)
        return xHat, yHat, res, slopes

    times = tfr_beta.columns
    start_fit = np.min(np.where(times >= 0))
    xHat, yHat, min_hand, slopes = get_pwlf(
        x=times[start_fit : np.argmin(tfr_beta.mean().values)],
        y=tfr_beta.mean().values[start_fit : np.argmin(tfr_beta.mean().values)],
    )

    start_plot = np.min(
        np.where(times >= -0.2275)
    )  # start to plot from mean of baseline
    stop_plot = np.min(
        np.where(times >= 0.75 + 0.433333)
    )  # end to plot at median rt plus 100 ms
    ax1.plot(
        [times[start_plot], times[stop_plot]],
        [0, 0],
        color="grey",
        linestyle="--",
        linewidth=0.5,
    )
    ax1.fill_between(
        [
            min_hand[1] + 0.25,
            np.round(times[np.argmin(tfr_beta.mean().values)] - 0.05, 4),
        ],
        -15,
        5,
        facecolor="lightgray",
        alpha=0.5,
    )
    ax1.plot(
        [min_hand[1], min_hand[1]],
        [-12, 2],
        color="grey",
        linestyle="--",
        linewidth=0.5,
    )
    ax1.plot(
        [
            times[np.argmin(tfr_beta.mean().values)],
            times[np.argmin(tfr_beta.mean().values)],
        ],
        [-12, 2],
        color="grey",
        linestyle="--",
        linewidth=0.5,
    )
    ax1.plot(
        [min_hand[1] + 0.25, min_hand[1] + 0.25],
        [-12, 2],
        color="grey",
        linestyle="--",
        linewidth=0.5,
    )
    ax1.plot(
        [
            np.round(times[np.argmin(tfr_beta.mean().values)] - 0.05, 4),
            np.round(times[np.argmin(tfr_beta.mean().values)] - 0.05, 4),
        ],
        [-12, 2],
        color="grey",
        linestyle="--",
        linewidth=0.5,
    )
    ax1.plot(
        times[start_plot:stop_plot],
        tfr_beta.mean().values[start_plot:stop_plot],
        color="k",
    )
    ax1.fill_between(
        tfr_beta.columns[start_plot:stop_plot],
        tfr_beta.mean().values[start_plot:stop_plot]
        + tfr_beta.sem().values[start_plot:stop_plot],
        tfr_beta.mean().values[start_plot:stop_plot]
        - tfr_beta.sem().values[start_plot:stop_plot],
        facecolor="gray",
        alpha=0.5,
    )
    ax1.plot(xHat, yHat, color="r")
    for i_c, c in enumerate(clusters):
        if cluster_p_values[i_c] <= 0.05:
            ax1.plot(
                times[c.start],
                -15,
                marker="s",
                color="k",
                ms=3,
                zorder=1,
                markeredgecolor="w",
                markeredgewidth=0.1,
            )

    ax1.set_xlabel("Time around stimulus onset in s")
    ax1.set_ylabel("% Power change Beta (12 - 36 Hz)")
    sns.despine(ax=ax1, offset=10, right=True, left=False)
    # pl.savefig('bilin_fit.pdf')
    plt.show()
    return min_hand[1], times[np.argmin(tfr_beta.mean().values)], tfr_beta


def get_data_for_plot(tfr, cluster):
    if cluster == "aIPS + M1":
        tfr_cluster = (
            tfr.query('cluster == "JWG_M1" or cluster == "JWG_aIPS"')
            .groupby(["freq", "subject"])
            .mean()
        )
        tfr_beta = tfr_cluster.query("12 <= freq <= 36").groupby("subject").mean()
    elif cluster == "aIPS + IPS_PCeS":
        tfr_cluster = (
            tfr.query('cluster == "JWG_aIPS" or cluster == "JWG_IPS_PCeS"')
            .groupby(["freq", "subject"])
            .mean()
        )
        tfr_beta = tfr_cluster.query("12 <= freq <= 36").groupby("subject").mean()
    else:
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


def get_slope(residual, times, lower_time, upper_time):
    lower_idx = np.min(np.where(times >= lower_time))
    upper_idx = np.min(np.where(times > upper_time))
    x = np.array(times[lower_idx:upper_idx])
    A = np.vstack([x, np.ones(len(x))]).T
    y = np.mean(residual, axis=0)[lower_idx:upper_idx]
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    m_single_subjects = []
    c_single_subjects = []
    for i in range(0, len(residual)):
        y_single_sj = residual[i][lower_idx:upper_idx]
        m_single_sj, c_single_sj = np.linalg.lstsq(A, y_single_sj, rcond=None)[0]
        m_single_subjects.append(m_single_sj)
        c_single_subjects.append(c_single_sj)
    return x, A, y, m, c, m_single_subjects, c_single_subjects


def figure_residual_beta_timecourses_bias_3_bins_contra_vs_ipsi_preferred_hemi_2(
    cluster, segment, contrast_bias_eq, contrast_bias_opp, lower_time, upper_time
):
    plt.figure(figsize=(8, 3))
    ax1 = plt.subplot2grid((1, 4), (0, 0))
    axes = [ax1]

    plt.subplots_adjust(
        left=0.1, bottom=0.15, top=0.8, right=0.95, wspace=0.5, hspace=0.7
    )
    bias_eq = list_lh_baseline_per_session_bin_on_bias_bins_subsample.query(
        f'epoch == "stimulus" & ~(hemi == "avg") & (contrast == "{contrast_bias_eq}")'
    )
    bias_opp = list_lh_baseline_per_session_bin_on_bias_bins_subsample.query(
        f'epoch == "stimulus" & ~(hemi == "avg") & (contrast == "{contrast_bias_opp}")'
    )
    tfr_beta_bias_eq, tfr_beta_bias_eq_reshaped = get_data_for_plot(bias_eq, cluster)
    tfr_beta_bias_opp, tfr_beta_bias_opp_reshaped = get_data_for_plot(bias_opp, cluster)
    tfr_beta_bias_reshaped = np.array(
        [
            np.mean(
                [
                    tfr_beta_bias_eq.query(f'subject == "{sj}"').values,
                    -tfr_beta_bias_opp.query(f'subject == "{sj}"').values,
                ],
                axis=0,
            )
            for sj in np.unique(tfr_beta_bias_eq.index.get_level_values("subject"))
        ]
    )
    k, l, m = np.shape(tfr_beta_bias_reshaped)
    tfr_beta_bias_reshaped = tfr_beta_bias_reshaped.reshape(k, m)
    max_bias = max_time
    min_bias = min_time
    start_plot = np.min(
        np.where(tfr_beta_bias_eq.columns >= -0.2275)
    )  # start to plot from mean of baseline
    stop_plot = np.min(np.where(tfr_beta_bias_eq.columns > max_bias))
    ax1.plot(
        [
            list_lh_baseline_per_session_bin_on_bias_bins_subsample.columns[start_plot],
            list_lh_baseline_per_session_bin_on_bias_bins_subsample.columns[stop_plot],
        ],
        [0, 0],
        color="grey",
        linestyle="--",
        linewidth=0.5,
    )
    residual_bias = []
    for i in range(0, len(tfr_beta_bias_reshaped)):
        residual_bias.append(tfr_beta_bias_reshaped[i][start_plot:stop_plot])
    times = list_lh_baseline_per_session_bin_on_bias_bins_subsample.columns[
        start_plot : stop_plot + 1
    ]
    (
        T_obs_bias,
        clusters_bias,
        cluster_p_values_bias,
        H0_bias,
    ) = permutation_cluster_1samp_test(
        np.array(residual_bias),
        threshold={"start": 0, "step": 0.2},
        tail=0,
        n_permutations=1000,
        out_type="mask",
    )
    min_ylim = min(
        np.array(np.nanmean(residual_bias, axis=0)[start_plot:stop_plot])
        - np.array(stats.sem(residual_bias, axis=0)[start_plot:stop_plot])
    )
    max_ylim = max(
        np.array(
            np.nanmean(residual_bias, axis=0)
            + stats.sem(residual_bias, axis=0, nan_policy="omit")
        )
    )
    for i_c, c in enumerate(clusters_bias[:-1]):
        if cluster_p_values_bias[i_c] <= 0.05:
            print("times[c.start], times[c.stop]", times[c.start], times[c.stop])
            ax1.plot(
                [times[c.start], times[c.stop]],
                [min_ylim - 0.5, min_ylim - 0.5],
                marker="s",
                color="gray",
                ms=3,
                zorder=1,
                markeredgecolor="w",
                markeredgewidth=0.1,
            )
    ax1.plot(
        list_lh_baseline_per_session_bin_on_bias_bins_subsample.columns[
            start_plot:stop_plot
        ],
        np.nanmean(residual_bias, axis=0),
        color="gray",
        label="Pooled across conditions",
    )
    ax1.fill_between(
        list_lh_baseline_per_session_bin_on_bias_bins_subsample.columns[
            start_plot:stop_plot
        ],
        np.nanmean(residual_bias, axis=0)
        + stats.sem(residual_bias, axis=0, nan_policy="omit"),
        np.nanmean(residual_bias, axis=0)
        - stats.sem(residual_bias, axis=0, nan_policy="omit"),
        facecolor="gray",
        alpha=0.5,
    )
    #################
    (
        x_bias,
        A_bias,
        y_bias,
        m_bias,
        c_bias,
        m_single_subjects_bias,
        c_single_subjects_bias,
    ) = get_slope(residual_bias, times, lower_time, upper_time)
    ax1.plot(x_bias, c_bias + x_bias * m_bias, color="gray", linestyle="--")
    for ax in axes:
        ax.plot(
            [0.75 + 0.333333, 0.75 + 0.333333],
            [min_ylim, 3.5],
            color="lightgrey",
            linestyle="--",
            linewidth=0.5,
        )  # vertical line at median RT
    ax1.plot(
        [min_bias, min_bias],
        [min_ylim, 3.5],
        color="grey",
        linestyle="--",
        linewidth=0.5,
    )
    ax1.plot(
        [max_bias, max_bias],
        [min_ylim, 3.5],
        color="grey",
        linestyle="--",
        linewidth=0.5,
    )
    sns.despine(ax=ax1, offset=10, right=True, left=False)
    # ax1.set_title(cluster)
    for ax in axes:
        ax.set_ylim(min_ylim - 1, max_ylim)
        sns.despine(ax=ax1, offset=10, right=True, left=False)
    ax1.set_xlabel("Time around stimulus onset in s")
    ax1.set_ylabel("% Power change Beta (12 - 36 Hz)")
    ax1.set_ylim(min_ylim - 1, max_ylim)
    plt.show()
    return m_single_subjects_bias, residual_bias, times


def figure_strip_slopes_pref_hemi():
    plt.figure(figsize=(1.3, 1.7))
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    plt.subplots_adjust(
        left=0.3, bottom=0.2, top=0.9, right=0.9, wspace=0.5, hspace=0.5
    )

    df = pd.DataFrame({"Late": df_late.Slope_bias, "Early": df_early.Slope_bias})

    p_late = permutation_test.perm_test(np.zeros(len(df.Late)), df.Late)
    p_early = permutation_test.perm_test(np.zeros(len(df.Early)), df.Early)
    print("p_late", p_late)
    print("p_early", p_early)
    ax1.plot([-0.4, 1.4], [0.0, 0.0], color="grey", linestyle="--")
    ax1 = sns.stripplot(
        data=df,
        marker="o",
        color="w",
        edgecolor="grey",
        alpha=1,
        zorder=2,
        size=5,
        linewidth=0.7,
        jitter=True,
    )
    ax1.plot(
        [-0.4, 0.4], [np.mean(df.Late), np.mean(df.Late)], color="k", linestyle="-"
    )
    ax1.plot(
        [0.6, 1.4], [np.mean(df.Early), np.mean(df.Early)], color="k", linestyle="-"
    )

    ax1.text(0, 40, stars(p_late), va="center", ha="center", fontsize=7)
    ax1.text(1, 40, stars(p_early), va="center", ha="center", fontsize=7)
    ax1.set_ylabel("Slope")
    ax1.set_yticks(range(-40, 40, 20))
    sns.despine(ax=ax1, offset=10, right=True, left=False)
    # pl.savefig('Slopes_early_late_start_pref_hemi' + '.pdf')
    plt.show()


min_time, max_time, tfr_beta = beta_slope_bilinear_fit_methods()
min_time = min_time + 0.25
max_time = np.round(max_time - 0.05, 4)


min_time_late = 0.58
max_time_late = 0.8475

min_time_early = 0.1475
max_time_early = 0.4725

(
    m_single_subjects_bias_late,
    residual_bias_late,
    times,
) = figure_residual_beta_timecourses_bias_3_bins_contra_vs_ipsi_preferred_hemi_2(
    cluster="JWG_M1",
    segment="late_segment",
    contrast_bias_eq="bias_high_tertile_hand",
    contrast_bias_opp="bias_low_tertile_hand",
    lower_time=min_time_late,
    upper_time=max_time_late,
)

(
    m_single_subjects_bias_early,
    residual_bias_early,
    times,
) = figure_residual_beta_timecourses_bias_3_bins_contra_vs_ipsi_preferred_hemi_2(
    cluster="JWG_M1",
    segment="early_segment",
    contrast_bias_eq="bias_high_tertile_hand",
    contrast_bias_opp="bias_low_tertile_hand",
    lower_time=min_time_early,
    upper_time=max_time_early,
)

df_late = pd.DataFrame(
    {
        "Slope_bias": m_single_subjects_bias_late,
        "subject": np.unique(
            list_lh_baseline_per_session_bin_on_bias_bins_subsample.index.get_level_values(
                "subject"
            )
        ).astype("int"),
    }
)
df_late = df_late.sort_values("subject")

df_early = pd.DataFrame(
    {
        "Slope_bias": m_single_subjects_bias_early,
        "subject": np.unique(
            list_lh_baseline_per_session_bin_on_bias_bins_subsample.index.get_level_values(
                "subject"
            )
        ).astype("int"),
    }
)
df_early = df_early.sort_values("subject")

figure_strip_slopes_pref_hemi()
