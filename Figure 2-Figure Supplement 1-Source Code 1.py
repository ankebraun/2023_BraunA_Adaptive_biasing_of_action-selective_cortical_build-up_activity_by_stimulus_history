"""
Plot correlation of LCMV beamformer weights
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

df = pd.read_csv(
    "Figure 2-Figure Supplement 1-Source Data 1.csv", sep="\t", index_col=0
)
plt.plot(df.columns, np.nanmean(df, axis=0), color="k")
plt.fill_between(
    df.columns,
    np.nanmean(df, axis=0) - stats.sem(df, axis=0),
    np.nanmean(df, axis=0) + stats.sem(df, axis=0),
    facecolor="grey",
)
plt.ylabel("Pearson correlation")
plt.xlabel("Distance (cm)")
sns.despine(ax=plt.gca(), offset=10, right=True, left=False)
plt.show()

df = pd.read_csv(
    "Figure 2-Figure Supplement 1-Source Data 2.csv", sep="\t", index_col=0
)
corr = df.round(2)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f")
plt.show()
