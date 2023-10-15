"""
Plot previous choice weights against previous stimulus weights
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns

import pycircstat as pc

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
        "font.family": ["sans-serif"],
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


prcformatter = matplotlib.ticker.FuncFormatter(lambda x, pos: "%g" % (100 * x,))
myformatter = matplotlib.ticker.FuncFormatter(lambda x, pos: "%g" % (x,))

df = pd.read_csv("Figure 1-Figure Supplement 2-Source Data 1.csv", sep="\t")

kr_neutral = df.kr_neutral.to_numpy()
kr_repetitive = df.kr_repetitive.to_numpy()
kr_alternating = df.kr_alternating.to_numpy()

kz_neutral = df.kz_neutral.to_numpy()
kz_repetitive = df.kz_repetitive.to_numpy()
kz_alternating = df.kz_alternating.to_numpy()

kr_mean_neutral = np.mean(kr_neutral)
kr_mean_repetitive = np.mean(kr_repetitive)
kr_mean_alternating = np.mean(kr_alternating)

kz_mean_neutral = np.mean(kz_neutral)
kz_mean_repetitive = np.mean(kz_repetitive)
kz_mean_alternating = np.mean(kz_alternating)


def figure5_strategy_plots(
    kr_neutral,
    kz_neutral,
    kr_mean_neutral,
    kz_mean_neutral,
    kr_repetitive,
    kz_repetitive,
    kr_mean_repetitive,
    kz_mean_repetitive,
    kr_alternating,
    kz_alternating,
    kr_mean_alternating,
    kz_mean_alternating,
):
    plt.figure(figsize=(3.0, 1.8))
    ax1 = plt.subplot2grid((1, 1), (0, 0), aspect="equal")
    plt.subplots_adjust(left=0.2, right=0.65, bottom=0.2)
    ax1.plot([0, 0], [-1.5, 1.5], color="gray", linestyle="--", linewidth=0.5)
    ax1.plot([-1.5, 1.5], [0, 0], color="gray", linestyle="--", linewidth=0.5)
    ax1.plot([-1.5, 1.5], [1.5, -1.5], color="gray", linestyle="--", linewidth=0.5)

    r_repetitive = np.zeros(len(kr_repetitive))
    r_alternating = np.zeros(len(kr_alternating))
    r_repetitive_vs_zero = np.zeros(len(kr_repetitive))
    r_alternating_vs_zero = np.zeros(len(kr_alternating))

    phi_repetitive = np.zeros(len(kr_repetitive))
    phi_alternating = np.zeros(len(kr_alternating))
    phi_repetitive_vs_zero = np.zeros(len(kr_repetitive))
    phi_alternating_vs_zero = np.zeros(len(kr_alternating))

    phi_cond_2_3 = np.zeros(len(kr_alternating))
    phi_neutral = np.zeros(len(kr_neutral))
    for j in range(0, len(kr_repetitive)):
        ax1.plot(
            kr_repetitive[j],
            kz_repetitive[j],
            marker="o",
            color="g",
            alpha=0.4,
            markeredgecolor="w",
            zorder=1,
            markeredgewidth=0.5,
            ms=6,
        )
        r_repetitive[j] = np.sqrt(
            (kr_repetitive[j] - kr_neutral[j]) ** 2
            + (kz_repetitive[j] - kz_neutral[j]) ** 2
        )
        r_repetitive_vs_zero[j] = np.sqrt(
            (kr_repetitive[j]) ** 2 + (kz_repetitive[j]) ** 2
        )
        phi_repetitive[j] = np.arctan2(
            (kz_repetitive[j] - kz_neutral[j]), (kr_repetitive[j] - kr_neutral[j])
        )
        phi_neutral[j] = np.arctan2((kz_neutral[j]), (kr_neutral[j]))
        phi_repetitive_vs_zero[j] = np.arctan2((kz_repetitive[j]), (kr_repetitive[j]))
    for j in range(0, len(kr_alternating)):
        ax1.plot(
            kr_alternating[j],
            kz_alternating[j],
            marker="^",
            color="b",
            alpha=0.4,
            markeredgecolor="w",
            zorder=1,
            markeredgewidth=0.5,
            ms=6,
        )
        ax1.plot(
            [kr_repetitive[j], kr_alternating[j]],
            [kz_repetitive[j], kz_alternating[j]],
            color="gray",
            linewidth=0.25,
        )
        r_alternating[j] = np.sqrt(
            (kr_alternating[j] - kr_neutral[j]) ** 2
            + (kz_alternating[j] - kz_neutral[j]) ** 2
        )
        r_alternating_vs_zero[j] = np.sqrt(
            (kr_alternating[j]) ** 2 + (kz_alternating[j]) ** 2
        )
        phi_alternating[j] = np.arctan2(
            (kz_alternating[j] - kz_neutral[j]), (kr_alternating[j] - kr_neutral[j])
        )
        phi_alternating_vs_zero[j] = np.arctan2(
            (kz_alternating[j]), (kr_alternating[j])
        )
        phi_cond_2_3[j] = np.arccos(
            float(
                kr_repetitive[j] * kr_alternating[j]
                + kz_repetitive[j] * kz_alternating[j]
            )
            / float(
                np.sqrt(kr_repetitive[j] ** 2 + kz_repetitive[j] ** 2)
                * np.sqrt(kr_alternating[j] ** 2 + kz_alternating[j] ** 2)
            )
        )

    pval_neutral, z_neutral = pc.rayleigh(phi_neutral, w=None, d=None, axis=None)
    pval_repetitive, z_repetitive = pc.rayleigh(
        phi_repetitive, w=None, d=None, axis=None
    )
    pval_alternating, z_alternating = pc.rayleigh(
        phi_alternating, w=None, d=None, axis=None
    )

    pval_repetitive_vs_zero, z_repetitive_vs_zero = pc.rayleigh(
        phi_repetitive_vs_zero, w=None, d=None, axis=None
    )
    pval_alternating_vs_zero, z_alternating_vs_zero = pc.rayleigh(
        phi_alternating_vs_zero, w=None, d=None, axis=None
    )

    pval_repetitive_3, z_repetitive_3 = pc.rayleigh(
        phi_cond_2_3, w=None, d=None, axis=None
    )

    pval_watson, T = pc.watson_williams(phi_repetitive, phi_alternating)

    print("pval_neutral", pval_neutral)
    print("z_neutral", z_neutral)
    print("pval_repetitive", pval_repetitive)
    print("z_repetitive", z_repetitive)
    print("pval_alternating", pval_alternating)
    print("z_alternating", z_alternating)

    ax1.arrow(
        kr_mean_neutral,
        kz_mean_neutral,
        kr_mean_repetitive - kr_mean_neutral,
        kz_mean_repetitive - kz_mean_neutral,
        lw=3.5,
        color="g",
        length_includes_head=True,
        zorder=2,
    )
    ax1.arrow(
        kr_mean_neutral,
        kz_mean_neutral,
        kr_mean_alternating - kr_mean_neutral,
        kz_mean_alternating - kz_mean_neutral,
        lw=3.5,
        color="b",
        length_includes_head=True,
        zorder=2,
    )
    ax1.text(
        kr_mean_neutral,
        kz_mean_neutral,
        "x",
        zorder=3,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=9,
        color="r",
        fontweight="bold",
    )
    ax1.legend(bbox_to_anchor=(1.1, 1), fontsize=7, loc=2, frameon=True)
    ax1.set_xlabel("Previous choice weights")
    ax1.set_ylabel("Previous stimulus weights")
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.xaxis.set_major_formatter(myformatter)
    ax1.yaxis.set_major_formatter(myformatter)
    ax1.get_yaxis().set_tick_params(direction="out")
    ax1.get_xaxis().set_tick_params(direction="out")


figure5_strategy_plots(
    kr_neutral,
    kz_neutral,
    kr_mean_neutral,
    kz_mean_neutral,
    kr_repetitive,
    kz_repetitive,
    kr_mean_repetitive,
    kz_mean_repetitive,
    kr_alternating,
    kz_alternating,
    kr_mean_alternating,
    kz_mean_alternating,
)
plt.show()
