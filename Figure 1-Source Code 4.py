"""
Plot partial regression between bias shift and performance
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

df = pd.read_csv("Figure 1-Source Data 6.csv", sep="\t")
out = pg.partial_corr(
    data=df, x="vector_len", y="performance", covar="sensitivity", method="pearson"
)

fig = plt.figure(figsize=(2.5, 2.5))
ax1 = plt.subplot2grid((1, 1), (0, 0))
axes = [ax1]
plt.tight_layout()
sm.graphics.plot_partregress(
    endog="performance",
    exog_i="vector_len",
    exog_others=["sensitivity"],
    data=df,
    obs_labels=False,
    ax=ax1,
    color="gray",
)

ax1.text(0.2, -0.03, "r = " + str(round(out["r"]["pearson"], 4)), fontsize=6)
ax1.text(0.2, -0.04, "p = " + str(round(out["p-val"]["pearson"], 4)), fontsize=6)
ax1.set_xlim(-0.6, 1.2)

ax1.get_xaxis().set_tick_params(direction="out")
ax1.get_yaxis().set_tick_params(direction="out")
ax1.yaxis.set_ticks_position("left")
ax1.xaxis.set_ticks_position("bottom")
ax1.spines["right"].set_color("none")
ax1.spines["top"].set_color("none")

ax1.xaxis.set_ticks(np.arange(-0.6, 1.4, 0.2))

sns.despine(ax=ax1, offset=10, right=True, left=False)

plt.show()
