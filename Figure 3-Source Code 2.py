"""
Plots baseline beta lateralization contra- vs. ipsilateral to
previous button-press separately for each environment.
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

plt.rcParams["legend.fontsize"] = "small"
plt.rcParams["legend.fontsize"] = "small"


df = pd.read_csv("Figure 3-Source Data 2.csv", sep="\t")

p_rep = permutation_test_two_tailed.perm_test_two_tailed(df.rep, np.zeros(len(df.rep)))
p_alt = permutation_test_two_tailed.perm_test_two_tailed(df.alt, np.zeros(len(df.alt)))
p_neutr = permutation_test_two_tailed.perm_test_two_tailed(
    df.neutr, np.zeros(len(df.neutr))
)
p_rep_vs_alt = permutation_test_two_tailed.perm_test_two_tailed(df.alt, df.rep)
p_neutr_vs_alt = permutation_test_two_tailed.perm_test_two_tailed(df.alt, df.neutr)
p_neutr_vs_rep = permutation_test_two_tailed.perm_test_two_tailed(df.rep, df.neutr)


def figure(df):
    plt.figure(figsize=(2.5, 2))
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    plt.subplots_adjust(
        left=0.25, bottom=0.25, right=0.95, top=0.85, wspace=1, hspace=0.5
    )

    ax1.text(
        2.0,
        35,
        stars(p_alt),
        zorder=3,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=7,
    )
    ax1.text(
        1,
        35,
        stars(p_neutr),
        zorder=3,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=7,
    )
    ax1.text(
        0,
        35,
        stars(p_rep),
        zorder=3,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=7,
    )
    ax1.text(
        0.5,
        40,
        stars(p_neutr_vs_rep),
        zorder=3,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=7,
    )
    ax1.text(
        1.5,
        40,
        stars(p_neutr_vs_alt),
        zorder=3,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=7,
    )
    ax1.text(
        1,
        45,
        stars(p_rep_vs_alt),
        zorder=3,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=7,
    )
    ax1.plot([-0.25, 2.25], [0, 0], color="gray", linestyle="--", linewidth=0.25)
    df2 = df[["rep", "neutr", "alt"]]
    sns.stripplot(
        data=df2,
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
    ax1.plot(
        [1.75, 2.25], [np.mean(df.alt, axis=0), np.mean(df.alt, axis=0)], color="b"
    )
    ax1.plot(
        [0.75, 1.25], [np.mean(df.neutr, axis=0), np.mean(df.neutr, axis=0)], color="r"
    )
    ax1.plot(
        [-0.25, 0.25], [np.mean(df.rep, axis=0), np.mean(df.rep, axis=0)], color="g"
    )
    ax1.set_ylabel("M1 baseline beta lateralization contra vs. ipsi prev. hand")
    sns.despine(ax=ax1, offset=10, right=True, left=False)
   # plt.savefig("Figure3B.pdf")
    plt.show()


figure(df)
