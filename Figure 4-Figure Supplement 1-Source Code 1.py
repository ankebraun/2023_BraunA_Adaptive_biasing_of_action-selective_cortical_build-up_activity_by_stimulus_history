"""
Plot distribution of coherences for up (top row) and down choices (bottom row)
averaged across the low and high single-trial bias bins before (left column)
and after subsampling (right column) to yield an equal number of up and down
choices within each single-trial bias bin
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

df_down_high = pd.read_csv(
    "Figure 4-Figure Supplement 1-Source Data 1.csv",
    sep="\t",
    usecols=["zero", "three", "nine", "two_seven", "eight_one"],
    index_col=None,
)
df_up_high = pd.read_csv(
    "Figure 4-Figure Supplement 1-Source Data 2.csv",
    sep="\t",
    usecols=["zero", "three", "nine", "two_seven", "eight_one"],
    index_col=None,
)
df_down_low = pd.read_csv(
    "Figure 4-Figure Supplement 1-Source Data 3.csv",
    sep="\t",
    usecols=["zero", "three", "nine", "two_seven", "eight_one"],
    index_col=None,
)
df_up_low = pd.read_csv(
    "Figure 4-Figure Supplement 1-Source Data 4.csv",
    sep="\t",
    usecols=["zero", "three", "nine", "two_seven", "eight_one"],
    index_col=None,
)

df_orig_down_high = pd.read_csv(
    "Figure 4-Figure Supplement 1-Source Data 5.csv",
    sep="\t",
    usecols=["zero", "three", "nine", "two_seven", "eight_one"],
    index_col=None,
)
df_orig_up_high = pd.read_csv(
    "Figure 4-Figure Supplement 1-Source Data 6.csv",
    sep="\t",
    usecols=["zero", "three", "nine", "two_seven", "eight_one"],
    index_col=None,
)
df_orig_down_low = pd.read_csv(
    "Figure 4-Figure Supplement 1-Source Data 7.csv",
    sep="\t",
    usecols=["zero", "three", "nine", "two_seven", "eight_one"],
    index_col=None,
)
df_orig_up_low = pd.read_csv(
    "Figure 4-Figure Supplement 1-Source Data 8.csv",
    sep="\t",
    usecols=["zero", "three", "nine", "two_seven", "eight_one"],
    index_col=None,
)


fig = plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0))
ax4 = plt.subplot2grid((2, 2), (1, 1))

axes = [ax1, ax2, ax3, ax4]

plt.subplots_adjust(left=0.1, bottom=0.15, top=0.85, right=0.9, wspace=0.5, hspace=0.7)

df_concat_up = pd.concat((df_up_high, df_up_low))
by_row_index_up = df_concat_up.groupby(df_concat_up.index)
df_means_up = by_row_index_up.mean()

df_concat_down = pd.concat((df_down_high, df_down_low))
by_row_index_down = df_concat_down.groupby(df_concat_down.index)
df_means_down = by_row_index_down.mean()

df_orig_concat_up = pd.concat((df_orig_up_high, df_orig_up_low))
by_row_index_orig_up = df_orig_concat_up.groupby(df_orig_concat_up.index)
df_orig_means_up = by_row_index_orig_up.mean()

df_orig_concat_down = pd.concat((df_orig_down_high, df_orig_down_low))
by_row_index_orig_down = df_orig_concat_down.groupby(df_orig_concat_down.index)
df_orig_means_down = by_row_index_orig_down.mean()

contrasts = ["0.0", "0.03", "0.09", "0.27", "0.81"]
ax1.bar(
    contrasts,
    df_means_up.mean().values,
    yerr=stats.sem(df_means_up),
    align="center",
    alpha=0.5,
    ecolor="black",
    capsize=10,
)
ax2.bar(
    contrasts,
    df_orig_means_up.mean().values,
    yerr=stats.sem(df_orig_means_up),
    align="center",
    alpha=0.5,
    ecolor="black",
    capsize=10,
)
ax3.bar(
    contrasts,
    df_means_down.mean().values,
    yerr=stats.sem(df_means_down),
    align="center",
    alpha=0.5,
    ecolor="black",
    capsize=10,
)
ax4.bar(
    contrasts,
    df_orig_means_down.mean().values,
    yerr=stats.sem(df_orig_means_down),
    align="center",
    alpha=0.5,
    ecolor="black",
    capsize=10,
)
ax1.set_title("Up resp. avg. across low and high bin subsampled")
ax2.set_title("Up resp. avg. across low and high bin orig. data")
ax3.set_title("Down resp. avg. across low and high bin subsampled")
ax4.set_title("Down resp. avg. across low and high bin orig. data")
for ax in axes:
    ax.set_ylim(0, 0.3)
    sns.despine(ax=ax, offset=10, right=True, left=False)
    ax.set_ylabel("Proportion of trials")
    ax.set_xlabel("Motion coherence")
#plt.savefig("Proportion_coherences_subsamples_mean_low_high_bin.pdf")
plt.show()
