"""
Plot stimulus weights as a function of lags
"""
import matplotlib
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
#    'axes.labelpad': 1.0},)

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


stim_kernels_rep_nan = pd.read_csv(
    "Figure 1-Figure Supplement 2-Source Data 2.csv", sep="\t", index_col=False
)
stim_kernels_rep_nan = stim_kernels_rep_nan.drop("Unnamed: 0", axis=1)

stim_kernels_neutr_nan = pd.read_csv(
    "Figure 1-Figure Supplement 2-Source Data 3.csv", sep="\t", index_col=False
)
stim_kernels_neutr_nan = stim_kernels_neutr_nan.drop("Unnamed: 0", axis=1)

stim_kernels_alt_nan = pd.read_csv(
    "Figure 1-Figure Supplement 2-Source Data 4.csv", sep="\t", index_col=False
)
stim_kernels_alt_nan = stim_kernels_alt_nan.drop("Unnamed: 0", axis=1)


fig = plt.figure(figsize=(5, 2))
ax1 = plt.subplot2grid((1, 3), (0, 0))
ax2 = plt.subplot2grid((1, 3), (0, 1))
ax3 = plt.subplot2grid((1, 3), (0, 2))
axes = [ax1, ax2, ax3]
plt.subplots_adjust(left=0.25, bottom=0.25, right=0.95, top=0.85, wspace=1, hspace=0.5)
ax1.plot([1, 7], [0, 0], color="gray", linestyle="--", linewidth=0.25)
ax2.plot([1, 7], [0, 0], color="gray", linestyle="--", linewidth=0.25)
ax3.plot([1, 7], [0, 0], color="gray", linestyle="--", linewidth=0.25)
for i in range(len(stim_kernels_rep_nan)):
    stim_kernels_rep = stim_kernels_rep_nan.loc[i].to_numpy()[
        ~np.isnan(stim_kernels_rep_nan.loc[i].to_numpy())
    ]
    if len(stim_kernels_rep) == 1:
        ax1.scatter(
            range(1, len(stim_kernels_rep) + 1),
            stim_kernels_rep,
            color="lightgreen",
            s=3,
        )
    ax1.plot(
        range(1, len(stim_kernels_rep_nan.loc[i]) + 1),
        stim_kernels_rep_nan.loc[i].values,
        linewidth=0.5,
        color="lightgreen",
    )
    ax1.plot(
        range(1, 8), np.nanmean(stim_kernels_rep_nan, axis=0), linewidth=1, color="g"
    )
for i in range(len(stim_kernels_neutr_nan)):
    stim_kernels_neutr = stim_kernels_neutr_nan.loc[i].to_numpy()[
        ~np.isnan(stim_kernels_neutr_nan.loc[i].to_numpy())
    ]
    if len(stim_kernels_neutr) == 1:
        ax2.scatter(
            range(1, len(stim_kernels_neutr) + 1),
            stim_kernels_neutr,
            color="salmon",
            s=3,
        )
    ax2.plot(
        range(1, len(stim_kernels_neutr_nan.loc[i]) + 1),
        stim_kernels_neutr_nan.loc[i].values,
        linewidth=0.5,
        color="salmon",
    )
    ax2.plot(
        range(1, 8), np.nanmean(stim_kernels_neutr_nan, axis=0), linewidth=1, color="r"
    )

for i in range(len(stim_kernels_alt_nan)):
    stim_kernels_alt = stim_kernels_alt_nan.loc[i].to_numpy()[
        ~np.isnan(stim_kernels_alt_nan.loc[i].to_numpy())
    ]
    if len(stim_kernels_alt) == 1:
        ax3.scatter(
            range(1, len(stim_kernels_alt) + 1),
            stim_kernels_alt,
            color="lightblue",
            s=3,
        )
    ax3.plot(
        range(1, len(stim_kernels_alt_nan.loc[i]) + 1),
        stim_kernels_alt_nan.loc[i].values,
        linewidth=0.5,
        color="lightblue",
    )
    ax3.plot(
        range(1, 8), np.nanmean(stim_kernels_alt_nan, axis=0), linewidth=1, color="b"
    )


ax1.set_ylabel("Stimulus weight")
ax2.set_xlabel("Lag")
ax1.set_ylim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
ax1.set_title("Repetitive")
ax2.set_title("Neutral")
ax3.set_title("Alternating")
ax1.xaxis.set_ticks(np.arange(1, 8, 1))
ax2.xaxis.set_ticks(np.arange(1, 8, 1))
ax3.xaxis.set_ticks(np.arange(1, 8, 1))
sns.despine(ax=ax1, offset=10, right=True, left=False)
sns.despine(ax=ax2, offset=10, right=True, left=False)
sns.despine(ax=ax3, offset=10, right=True, left=False)
# plt.savefig("stim_kernels.pdf")
plt.show()
