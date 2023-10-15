"""
Plot partial correlation between previous stimulus weights and individual 
performance in each environment after factoring out the correlation of 
both variables with perceptual sensitivity
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import statsmodels.api as sm


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

df_rep = pd.read_csv("Figure 1-Figure Supplement 3-Source Data 1.csv", sep="\t")
out_rep = pg.partial_corr(
    data=df_rep,
    x="stim_weights_rep",
    y="performance_rep",
    covar="sensitivity_rep",
    method="pearson",
)

df_neutr = pd.read_csv("Figure 1-Figure Supplement 3-Source Data 2.csv", sep="\t")
out_neutr = pg.partial_corr(
    data=df_neutr,
    x="stim_weights_neutr",
    y="performance_neutr",
    covar="sensitivity_neutr",
    method="pearson",
)

df_alt = pd.read_csv("Figure 1-Figure Supplement 3-Source Data 3.csv", sep="\t")
out_alt = pg.partial_corr(
    data=df_alt,
    x="stim_weights_alt",
    y="performance_alt",
    covar="sensitivity_alt",
    method="pearson",
)


def figure_partial_diff_stim_weights_sensitivity_performance():
    plt.figure(figsize=(7.5, 2.5))
    ax1 = plt.subplot2grid((1, 3), (0, 0))
    ax2 = plt.subplot2grid((1, 3), (0, 1))
    ax3 = plt.subplot2grid((1, 3), (0, 2))
    axes = [ax1, ax2, ax3]

    plt.tight_layout()

    sm.graphics.plot_partregress(
        endog="performance_rep",
        exog_i="stim_weights_rep",
        exog_others=["sensitivity_rep"],
        data=df_rep,
        obs_labels=False,
        ax=ax1,
        color="g",
    )
    sm.graphics.plot_partregress(
        endog="performance_neutr",
        exog_i="stim_weights_neutr",
        exog_others=["sensitivity_neutr"],
        data=df_neutr,
        obs_labels=False,
        ax=ax2,
        color="r",
    )
    sm.graphics.plot_partregress(
        endog="performance_alt",
        exog_i="stim_weights_alt",
        exog_others=["sensitivity_alt"],
        data=df_alt,
        obs_labels=False,
        ax=ax3,
        color="b",
    )

    ax1.text(0.2, -0.03, "r = " + str(round(out_rep["r"]["pearson"], 4)), fontsize=6)
    ax1.text(
        0.2, -0.04, "p = " + str(round(out_rep["p-val"]["pearson"], 4)), fontsize=6
    )

    ax2.text(0.2, -0.03, "r = " + str(round(out_neutr["r"]["pearson"], 4)), fontsize=6)
    ax2.text(
        0.2, -0.04, "p = " + str(round(out_neutr["p-val"]["pearson"], 4)), fontsize=6
    )

    ax3.text(0.2, -0.03, "r = " + str(round(out_alt["r"]["pearson"], 4)), fontsize=6)
    ax3.text(
        0.2, -0.04, "p = " + str(round(out_alt["p-val"]["pearson"], 4)), fontsize=6
    )

    for j in range(0, 3):
        axes[j].set_xlim(-0.6, 1.2)
        axes[j].get_xaxis().set_tick_params(direction="out")
        axes[j].get_yaxis().set_tick_params(direction="out")
        axes[j].yaxis.set_ticks_position("left")
        axes[j].xaxis.set_ticks_position("bottom")
        axes[j].spines["right"].set_color("none")
        axes[j].spines["top"].set_color("none")

    ax1.xaxis.set_ticks(np.arange(-0.6, 1.4, 0.2))

    for ax in axes:
        sns.despine(ax=ax, offset=10, right=True, left=False)

    plt.show()


figure_partial_diff_stim_weights_sensitivity_performance()
