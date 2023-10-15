"""
Plot previous stimulus weights for the three environments
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from stats import permutation_test_two_tailed
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

# Load data
df_weights_rep = pd.read_csv("Figure 1-Source Data 2.csv", sep="\t")
df_weights_alt = pd.read_csv("Figure 1-Source Data 3.csv", sep="\t")
df_weights_neutr = pd.read_csv("Figure 1-Source Data 4.csv", sep="\t")

kz_rep = df_weights_rep.kz_repetitive
kz_alt = df_weights_alt.kz_alternating
kz_neutr = df_weights_neutr.kz_neutral

df = pd.DataFrame({"kz_rep": kz_rep, "kz_neutr": kz_neutr, "kz_alt": kz_alt})

# Compute stats
p_rep = permutation_test_two_tailed.perm_test_two_tailed(
    df_weights_rep.kz_repetitive, np.zeros(len(df_weights_rep.kz_repetitive))
)
p_alt = permutation_test_two_tailed.perm_test_two_tailed(
    df_weights_alt.kz_alternating, np.zeros(len(df_weights_alt.kz_alternating))
)
p_neutr = permutation_test_two_tailed.perm_test_two_tailed(
    df_weights_neutr.kz_neutral, np.zeros(len(df_weights_neutr.kz_neutral))
)
p_rep_vs_neutr = permutation_test_two_tailed.perm_test_two_tailed(
    df_weights_neutr.kz_neutral, df_weights_rep.kz_repetitive
)
p_neutr_vs_alt = permutation_test_two_tailed.perm_test_two_tailed(
    df_weights_neutr.kz_neutral, df_weights_alt.kz_alternating
)
p_rep_vs_alt = permutation_test_two_tailed.perm_test_two_tailed(
    df_weights_rep.kz_repetitive, df_weights_alt.kz_alternating
)


def figure_stim_weights(df):
    """Plots previous stimulus weights from logistic regression model

    Parameters
    ----------
    df : DataFrame
        previous stimulus weights for Repetitive, Neutral and Alternating conditions

    Returns
    -------
    Figure
        Plot of previous stimulus weights
    """
    plt.figure(figsize=(2.5, 2))
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    plt.subplots_adjust(
        left=0.25, bottom=0.25, right=0.95, top=0.85, wspace=1, hspace=0.5
    )

    ax1.text(
        2.0,
        1.2,
        stars(p_alt),
        zorder=3,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=7,
    )
    ax1.text(
        1,
        1.2,
        stars(p_neutr),
        zorder=3,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=7,
    )
    ax1.text(
        0,
        1.2,
        stars(p_rep),
        zorder=3,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=7,
    )
    ax1.text(
        0.5,
        1.2,
        stars(p_rep_vs_neutr),
        zorder=3,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=7,
    )
    ax1.text(
        1.5,
        1.2,
        stars(p_neutr_vs_alt),
        zorder=3,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=7,
    )
    ax1.text(
        1,
        1.5,
        stars(p_rep_vs_alt),
        zorder=3,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=7,
    )
    ax1.plot([-0.25, 2.25], [0, 0], color="gray", linestyle="--", linewidth=0.25)
    ax1.plot(
        [1.75, 2.25],
        [np.mean(df.kz_alt, axis=0), np.mean(df.kz_alt, axis=0)],
        color="b",
    )
    ax1.plot(
        [0.75, 1.25],
        [np.mean(df.kz_neutr, axis=0), np.mean(df.kz_neutr, axis=0)],
        color="r",
    )
    ax1.plot(
        [-0.25, 0.25],
        [np.mean(df.kz_rep, axis=0), np.mean(df.kz_rep, axis=0)],
        color="g",
    )
    df = pd.DataFrame(
        {
            "stim weight rep.": df.kz_rep,
            "stim weight neur.": df.kz_neutr,
            "stim weight alt.": df.kz_alt,
        }
    )
    df = df[["stim weight rep.", "stim weight neur.", "stim weight alt."]]
    sns.stripplot(
        data=df,
        marker="o",
        color="w",
        edgecolor="grey",
        alpha=1,
        zorder=2,
        size=5,
        linewidth=0.7,
        jitter=True,
        ax=ax1,
    )
    ax1.set_ylabel("Previous stimulus weight")
    ax1.set_ylim(-1.5, 1.5)
    ax1.yaxis.set_ticks(np.arange(-0.75, 1.25, 0.25))
    sns.despine(ax=ax1, offset=10, right=True, left=False)
    #    pl.savefig('stim_weights_stripplot.pdf')
    plt.show()


figure_stim_weights(df)
