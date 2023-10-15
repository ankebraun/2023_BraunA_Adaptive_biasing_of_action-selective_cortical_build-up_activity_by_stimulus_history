"""
Plot psychometric functions conditioned on previous 
stimulus category for the three environments
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Compute Bayes Factors
import rpy2.robjects as robjects
import scipy
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr
from scipy import stats
from sklearn import linear_model

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
        "xtick.major.pad": 1,
        "ytick.major.pad": 1,
        "axes.labelpad": 1.0,
    },
)

sns.plotting_context()

fullwidth = 6.3
halfwidth = 0.45 * fullwidth
laby = 1.01
thlev = 0.85
sns.plotting_context()

# Load behavioral data
df1 = pd.read_csv("Figure 1-Source Data 1.csv", sep="\t")


def comp_dat_prev_stim(condition):
    df2 = df1.loc[df1["condition"] == condition]
    response_prev_up_stim_all_sj = []
    response_prev_down_stim_all_sj = []
    stim_prev_up_stim_all_sj = []
    stim_prev_down_stim_all_sj = []

    def comp_prev_stim(measure, prev_target):
        measure_prev_stim = [
            measure[i] for i in range(1, len(measure)) if target[i - 1] == prev_target
        ]
        measure_prev_stim = np.array(measure_prev_stim)
        return measure_prev_stim

    for sj in np.unique(df2["subject"]):
        df3 = df2.loc[df2["subject"] == sj]
        df3["coherence"].tolist()
        response = df3["response"].tolist()
        response = [x * 2 - 1 for x in response]
        target = df3["target"].tolist()
        target = [x * 2 - 1 for x in target]
        stim = df3["coherence"] * target
        stim = stim.tolist()
        stim_prev_up_stim = comp_prev_stim(stim, 1)
        stim_prev_down_stim = comp_prev_stim(stim, -1)
        response_prev_up_stim = comp_prev_stim(response, 1)
        response_prev_down_stim = comp_prev_stim(response, -1)
        stim_prev_up_stim_all_sj.append(stim_prev_up_stim)
        stim_prev_down_stim_all_sj.append(stim_prev_down_stim)
        response_prev_up_stim_all_sj.append(response_prev_up_stim)
        response_prev_down_stim_all_sj.append(response_prev_down_stim)
    return (
        stim_prev_up_stim_all_sj,
        stim_prev_down_stim_all_sj,
        response_prev_up_stim_all_sj,
        response_prev_down_stim_all_sj,
    )


def figure1c(
    stim_prev_up, response_prev_up, stim_prev_down, response_prev_down, condition
):
    fig, ax = plt.subplots(figsize=(2.5, 1.5))
    axins = zoomed_inset_axes(ax, 2, loc=4)  # zoom-factor: 2, location: lower-right
    plt.subplots_adjust(
        left=0.3, bottom=0.1, top=0.9, right=0.7, wspace=0.5, hspace=0.5
    )

    ax.plot([0, 0], [0, 1], color="k", linestyle="--", linewidth=0.5)
    ax.plot([-0.81, 0.81], [0.5, 0.5], color="k", linestyle="--", linewidth=0.5)
    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_ticks(np.arange(-0.81, 0.9, 0.81))
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.5))

    axins.set_xlim(-0.1, 0.1)
    axins.set_ylim(0.4, 0.6)
    axins.xaxis.set_ticks(np.arange(-0.1, 0.11, 0.1))
    axins.yaxis.set_ticks(np.arange(0.4, 0.65, 0.1))
    axins.set_xticklabels(np.arange(-0.1, 0.11, 0.1), fontsize=6)
    axins.set_yticklabels(np.arange(0.4, 0.65, 0.1), fontsize=6)

    clf_prev_up_coef_all = []
    clf_prev_down_coef_all = []

    clf_prev_up_intercept_all = []
    clf_prev_down_intercept_all = []

    s_prev_up_all = []
    s_prev_down_all = []

    def model(x):
        return 1 / (1 + np.exp(-x))

    def comp_clf(stim, response):
        if len(np.unique(response)) == 2:
            clf = linear_model.LogisticRegression(C=1e5)
            clf.fit(stim.reshape(-1, 1), response)
            clf_coef = clf.coef_
            clf_intercept = clf.intercept_[0]
        else:
            clf_intercept = np.nan
            clf_coef = np.nan

        return clf_intercept, clf_coef

    def comp_datapoints(stim, response):
        d_ = np.zeros(len(np.unique(stim[i])))
        t_ = np.zeros(len(np.unique(stim[i])))
        for l in range(0, len(response[i])):
            for k, o in enumerate(np.unique(stim[i])):
                if stim[i][l] == o:
                    d_[k] += (response[i][l] + 1) / 2
                    t_[k] += 1

        return d_ / t_

    x_test = np.linspace(-0.81, 0.81, 300)
    for i in range(0, len(stim_prev_up)):
        clf_intercept_prev_up, clf_coef_prev_up = comp_clf(
            stim_prev_up[i].reshape(-1, 1), response_prev_up[i]
        )
        clf_intercept_prev_down, clf_coef_prev_down = comp_clf(
            stim_prev_down[i].reshape(-1, 1), response_prev_down[i]
        )

        model(x_test * clf_coef_prev_up + clf_intercept_prev_up).ravel()
        model(x_test * clf_coef_prev_down + clf_intercept_prev_down).ravel()

        clf_prev_down_coef_all.append(clf_coef_prev_down)
        clf_prev_down_intercept_all.append(clf_intercept_prev_down)

        clf_prev_up_coef_all.append(clf_coef_prev_up)
        clf_prev_up_intercept_all.append(clf_intercept_prev_up)

        s_prev_up = comp_datapoints(stim_prev_up, response_prev_up)
        s_prev_down = comp_datapoints(stim_prev_down, response_prev_down)
        s_prev_up_all.append(s_prev_up)
        s_prev_down_all.append(s_prev_down)

    loss_up_mean = model(
        x_test * np.mean(clf_prev_up_coef_all) + np.mean(clf_prev_up_intercept_all)
    ).ravel()
    loss_down_mean = model(
        x_test * np.mean(clf_prev_down_coef_all) + np.mean(clf_prev_down_intercept_all)
    ).ravel()

    ax.plot(
        x_test, loss_up_mean, "k", label="Previous stim:\n up", linewidth=1, zorder=1
    )
    axins.plot(x_test, loss_up_mean, "k", label="Previous stim:\n up", linewidth=1)

    ax.plot(
        x_test,
        loss_down_mean,
        "grey",
        label="Previous stim: down",
        linewidth=1,
        zorder=1,
    )
    axins.plot(x_test, loss_down_mean, "grey", label="Previous stim: down", linewidth=1)

    ax.errorbar(
        np.unique(stim_prev_up[0]),
        np.mean(s_prev_up_all, axis=0),
        stats.sem(s_prev_up_all),
        ecolor="k",
        elinewidth=0.5,
        linewidth=0.25,
        zorder=2,
    )

    ax.errorbar(
        np.unique(stim_prev_down[0]),
        np.mean(s_prev_down_all, axis=0),
        stats.sem(s_prev_down_all),
        ecolor="grey",
        elinewidth=0.5,
        linewidth=0.25,
        zorder=2,
    )

    ax.scatter(
        np.unique(stim_prev_up[0]),
        np.mean(s_prev_up_all, axis=0),
        color="k",
        marker="o",
        s=5,
        linewidth=0.25,
        edgecolor="w",
        zorder=3,
    )
    ax.scatter(
        np.unique(stim_prev_down[0]),
        np.mean(s_prev_down_all, axis=0),
        color="grey",
        marker="o",
        s=5,
        linewidth=0.25,
        edgecolor="w",
        zorder=3,
    )

    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5", zorder=3)

    ax.set_ylabel("Probability up choices")
    ax.set_xlabel("Upward motion coherence")
    ax.set_title(condition)

    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    axins.yaxis.set_ticks_position("left")
    axins.xaxis.set_ticks_position("bottom")

    sns.despine(ax=ax, offset=10, trim=True)
    ax.legend(bbox_to_anchor=(1.05, 1), fontsize=6, loc=2, frameon=True)
    plt.show()
    return (
        np.array(clf_prev_up_intercept_all),
        np.array(clf_prev_down_intercept_all),
        np.array(clf_prev_up_coef_all),
        np.array(clf_prev_down_coef_all),
        np.array(clf_prev_up_intercept_all) - np.array(clf_prev_down_intercept_all),
        np.array(clf_prev_up_coef_all) - np.array(clf_prev_down_coef_all),
    )


(
    stim_prev_up_stim_repetitive,
    stim_prev_down_stim_repetitive,
    response_prev_up_stim_repetitive,
    response_prev_down_stim_repetitive,
) = comp_dat_prev_stim("Repetitive")
(
    stim_prev_up_stim_alternating,
    stim_prev_down_stim_alternating,
    response_prev_up_stim_alternating,
    response_prev_down_stim_alternating,
) = comp_dat_prev_stim("Alternating")
(
    stim_prev_up_stim_neutral,
    stim_prev_down_stim_neutral,
    response_prev_up_stim_neutral,
    response_prev_down_stim_neutral,
) = comp_dat_prev_stim("Neutral")

(
    bias_prev_up_rep,
    bias_prev_down_rep,
    slope_prev_up_rep,
    slope_prev_down_rep,
    shift_bias_rep,
    shift_slope_rep,
) = figure1c(
    stim_prev_up_stim_repetitive,
    response_prev_up_stim_repetitive,
    stim_prev_down_stim_repetitive,
    response_prev_down_stim_repetitive,
    "Repetitive",
)
(
    bias_prev_up_alt,
    bias_prev_down_alt,
    slope_prev_up_alt,
    slope_prev_down_alt,
    shift_bias_alt,
    shift_slope_alt,
) = figure1c(
    stim_prev_up_stim_alternating,
    response_prev_up_stim_alternating,
    stim_prev_down_stim_alternating,
    response_prev_down_stim_alternating,
    "Alternating",
)
(
    bias_prev_up_neutr,
    bias_prev_down_neutr,
    slope_prev_up_neutr,
    slope_prev_down_neutr,
    shift_bias_neutr,
    shift_slope_neutr,
) = figure1c(
    stim_prev_up_stim_neutral,
    response_prev_up_stim_neutral,
    stim_prev_down_stim_neutral,
    response_prev_down_stim_neutral,
    "Neutral",
)


shift_slope_rep = shift_slope_rep.reshape(len(shift_slope_rep))
shift_slope_alt = shift_slope_alt.reshape(len(shift_slope_alt))
shift_slope_neutr = shift_slope_neutr.reshape(len(shift_slope_neutr))

slope_prev_up_rep = slope_prev_up_rep.reshape(len(slope_prev_up_rep))
slope_prev_down_rep = slope_prev_down_rep.reshape(len(slope_prev_down_rep))
slope_prev_up_alt = slope_prev_up_alt.reshape(len(slope_prev_up_alt))
slope_prev_down_alt = slope_prev_down_alt.reshape(len(slope_prev_down_alt))
slope_prev_up_neutr = slope_prev_up_neutr.reshape(len(slope_prev_up_neutr))
slope_prev_down_neutr = slope_prev_down_neutr.reshape(len(slope_prev_down_neutr))

df = pd.DataFrame(
    {
        "subject": np.unique(df1.subject),
        "bias_prev_up_rep": bias_prev_up_rep,
        "bias_prev_down_rep": bias_prev_down_rep,
        "bias_prev_up_alt": bias_prev_up_alt,
        "bias_prev_down_alt": bias_prev_down_alt,
        "bias_prev_up_neutr": bias_prev_up_neutr,
        "bias_prev_down_neutr": bias_prev_down_neutr,
        "shift_bias_rep": shift_bias_rep,
        "shift_bias_alt": shift_bias_alt,
        "shift_bias_neutr": shift_bias_neutr,
        "shift_slope_rep": shift_slope_rep,
        "shift_slope_alt": shift_slope_alt,
        "shift_slope_neutr": shift_slope_neutr,
        "slope_prev_up_rep": slope_prev_up_rep,
        "slope_prev_down_rep": slope_prev_down_rep,
        "slope_prev_up_alt": slope_prev_up_alt,
        "slope_prev_down_alt": slope_prev_down_alt,
        "slope_prev_up_neutr": slope_prev_up_neutr,
        "slope_prev_down_neutr": slope_prev_down_neutr,
    }
)

# df.to_csv('pmf_shifts.csv', sep = '\t')

# df = pd.read_csv('pmf_shifts.csv', sep = '\t')

# Compute p-values
t_shift_bias_rep = scipy.stats.ttest_rel(df.bias_prev_up_rep, df.bias_prev_down_rep)
t_shift_slope_rep = scipy.stats.ttest_rel(df.slope_prev_up_rep, df.slope_prev_down_rep)

t_shift_bias_alt = scipy.stats.ttest_rel(df.bias_prev_up_alt, df.bias_prev_down_alt)
t_shift_slope_alt = scipy.stats.ttest_rel(df.slope_prev_up_alt, df.slope_prev_down_alt)

t_shift_bias_neutr = scipy.stats.ttest_rel(
    df.bias_prev_up_neutr, df.bias_prev_down_neutr
)
t_shift_slope_neutr = scipy.stats.ttest_rel(
    df.slope_prev_up_neutr, df.slope_prev_down_neutr
)


pandas2ri.activate()

BayesFactor = importr("BayesFactor")

robjects.globalenv["slope_prev_up_rep"] = slope_prev_up_rep
robjects.globalenv["slope_prev_down_rep"] = slope_prev_down_rep
ttest_output = r(
    'print(t.test(slope_prev_up_rep, slope_prev_down_rep,\
                 alternative="two.sided", paired = TRUE))'
)
r(
    "bf = ttestBF(y=slope_prev_up_rep, x=slope_prev_down_rep, nullInterval = NULL,\
  paired = TRUE)"
)
r("print(bf)")


robjects.globalenv["slope_prev_up_alt"] = slope_prev_up_alt
robjects.globalenv["slope_prev_down_alt"] = slope_prev_down_alt
ttest_output = r(
    'print(t.test(slope_prev_up_alt, slope_prev_down_alt,\
                 alternative="two.sided", paired = TRUE))'
)
r(
    "bf = ttestBF(y=slope_prev_up_alt, x=slope_prev_down_alt, nullInterval = NULL,\
  paired = TRUE)"
)
r("print(bf)")


robjects.globalenv["slope_prev_up_neutr"] = slope_prev_up_neutr
robjects.globalenv["slope_prev_down_neutr"] = slope_prev_down_neutr
ttest_output = r(
    'print(t.test(slope_prev_up_neutr, slope_prev_down_neutr,\
                 neutrernative="two.sided", paired = TRUE))'
)
r(
    "bf = ttestBF(y=slope_prev_up_neutr, x=slope_prev_down_neutr, nullInterval = NULL,\
  paired = TRUE)"
)
r("print(bf)")


robjects.globalenv["bias_prev_up_rep"] = bias_prev_up_rep
robjects.globalenv["bias_prev_down_rep"] = bias_prev_down_rep
ttest_output = r(
    'print(t.test(bias_prev_up_rep, bias_prev_down_rep,\
                 alternative="two.sided", paired = TRUE))'
)
r(
    "bf = ttestBF(y=bias_prev_up_rep, x=bias_prev_down_rep, nullInterval = NULL,\
  paired = TRUE)"
)
r("print(bf)")


robjects.globalenv["bias_prev_up_alt"] = bias_prev_up_alt
robjects.globalenv["bias_prev_down_alt"] = bias_prev_down_alt
ttest_output = r(
    'print(t.test(bias_prev_up_alt, bias_prev_down_alt,\
                 alternative="two.sided", paired = TRUE))'
)
r(
    "bf = ttestBF(y=bias_prev_up_alt, x=bias_prev_down_alt, nullInterval = NULL,\
  paired = TRUE)"
)
r("print(bf)")

robjects.globalenv["bias_prev_up_neutr"] = bias_prev_up_neutr
robjects.globalenv["bias_prev_down_neutr"] = bias_prev_down_neutr
ttest_output = r(
    'print(t.test(bias_prev_up_neutr, bias_prev_down_neutr,\
                 neutrernative="two.sided", paired = TRUE))'
)
r(
    "bf = ttestBF(y=bias_prev_up_neutr, x=bias_prev_down_neutr, nullInterval = NULL,\
  paired = TRUE)"
)
r("print(bf)")
