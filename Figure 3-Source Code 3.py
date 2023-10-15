"""
Plot beta weights of M1 beta lateralization vs. single-trial bias during baseline
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from stats import permutation_test_two_tailed
from stats.stars import stars

############ Load data from baseline regression model ###############
regr_baseline_ampl_bias_pooled = pd.read_csv("Figure 3-Source Data 3.csv", sep="\t")
regr_baseline_ampl_bias_pooled_rep = pd.read_csv("Figure 3-Source Data 4.csv", sep="\t")
regr_baseline_ampl_bias_pooled_alt = pd.read_csv("Figure 3-Source Data 5.csv", sep="\t")
regr_baseline_ampl_bias_pooled_neutr = pd.read_csv(
    "Figure 3-Source Data 6.csv", sep="\t"
)

bl_bias_pooled = regr_baseline_ampl_bias_pooled["0"].to_numpy()
bl_bias_pooled_rep = regr_baseline_ampl_bias_pooled_rep["0"].to_numpy()
bl_bias_pooled_alt = regr_baseline_ampl_bias_pooled_alt["0"].to_numpy()
bl_bias_pooled_neutr = regr_baseline_ampl_bias_pooled_neutr["0"].to_numpy()

df = pd.DataFrame(
    {
        "All": bl_bias_pooled,
        "Repetitive": bl_bias_pooled_rep,
        "Neutral": bl_bias_pooled_neutr,
        "Alternating": bl_bias_pooled_alt,
    }
)

p_all = permutation_test_two_tailed.perm_test_two_tailed(
    np.zeros(len(df)), df.All.to_numpy()
)
p_rep = permutation_test_two_tailed.perm_test_two_tailed(
    np.zeros(len(df)), df.Repetitive.to_numpy()
)
p_alt = permutation_test_two_tailed.perm_test_two_tailed(
    np.zeros(len(df)), df.Alternating.to_numpy()
)
p_neutr = permutation_test_two_tailed.perm_test_two_tailed(
    np.zeros(len(df)), df.Neutral.to_numpy()
)


def figure_bl_beta():
    plt.figure(figsize=(3, 3))
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    plt.subplots_adjust(
        left=0.3, bottom=0.2, top=0.9, right=0.9, wspace=0.5, hspace=0.5
    )
    ax1.plot([-0.4, 3.4], [0.0, 0.0], color="grey", linestyle="--")
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
    ax1.plot([-0.4, 0.4], [np.mean(df.All), np.mean(df.All)], color="k", linestyle="-")
    ax1.plot(
        [0.6, 1.4],
        [np.mean(df.Repetitive), np.mean(df.Repetitive)],
        color="g",
        linestyle="-",
    )
    ax1.plot(
        [1.6, 2.4], [np.mean(df.Neutral), np.mean(df.Neutral)], color="r", linestyle="-"
    )
    ax1.plot(
        [2.6, 3.4],
        [np.mean(df.Alternating), np.mean(df.Alternating)],
        color="b",
        linestyle="-",
    )

    ax1.text(0, 0.3, stars(p_all), va="center", ha="center", fontsize=7)
    ax1.text(1, 0.3, stars(p_rep), va="center", ha="center", fontsize=7)
    ax1.text(2, 0.3, stars(p_neutr), va="center", ha="center", fontsize=7)
    ax1.text(3, 0.3, stars(p_alt), va="center", ha="center", fontsize=7)

    ax1.set_ylim([-0.45, 0.45])
    ax1.set_ylabel("Beta weights M1 beta lat.\n vs. single-trial bias during baseline")
    sns.despine(ax=ax1, offset=10, right=True, left=False)
    plt.savefig("Figure3C.pdf")
    plt.show()


figure_bl_beta()
