import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns


matplotlib.rcParams['pdf.fonttype'] = 42
sns.set(
    style='ticks',
    font='Helvetica',
    font_scale=1,
    rc={
        'axes.labelsize': 7,
        'axes.titlesize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.linewidth': 0.25,
        'xtick.major.width': 0.25,
        'ytick.major.width': 0.25,
        # 'lines.linewidth': 0.5,
        'text.color': 'Black',
        'axes.labelcolor': 'Black',
        'xtick.color': 'Black',
        'ytick.color': 'Black',
        'font.family': [u'sans-serif'],
        'font.sans-serif': [u'Helvetica'],
        'xtick.major.pad': 1,
        'ytick.major.pad': 1,
        'axes.labelpad': 1.0,
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

pl.rcParams['legend.fontsize'] = 'small'
pl.rcParams['legend.fontsize'] = 'small'


df1_neutral = pd.read_csv('likelihood_model_comparison_neutral.csv', sep='\t')
df1_repetitive = pd.read_csv('likelihood_model_comparison_repetitive.csv', sep='\t')
df1_alternating = pd.read_csv('likelihood_model_comparison_alternating.csv', sep='\t')


fig = plt.figure(figsize=(9, 3))
ax1 = plt.subplot2grid((1, 3), (0, 0))
ax2 = plt.subplot2grid((1, 3), (0, 1))
ax3 = plt.subplot2grid((1, 3), (0, 2))

axes = [ax1, ax2, ax3]
plt.subplots_adjust(left=0.1, bottom=0.15, top=0.8, right=0.95, wspace=0.5, hspace=0.7)

x1 = [0, 1, 2, 3, 4, 5, 6, 7]
my_xticks = [0, 1, 2, 3, 4, 5, 6, 7]


def plot_ll(df1, col, lab, ax):
    y1 = [
        np.sum(df1['model_order'] == 'likelihood_no_hist'),
        np.sum(df1['model_order'] == 'likelihood_1_lags'),
        np.sum(df1['model_order'] == 'likelihood_2_lags'),
        np.sum(df1['model_order'] == 'likelihood_3_lags'),
        np.sum(df1['model_order'] == 'likelihood_4_lags'),
        np.sum(df1['model_order'] == 'likelihood_5_lags'),
        np.sum(df1['model_order'] == 'likelihood_6_lags'),
        np.sum(df1['model_order'] == 'likelihood_7_lags'),
    ]

    mean_across_sj = [
        df1.likelihood_no_hist.mean(),
        df1.likelihood_1_lags.mean(),
        df1.likelihood_2_lags.mean(),
        df1.likelihood_3_lags.mean(),
        df1.likelihood_4_lags.mean(),
        df1.likelihood_5_lags.mean(),
        df1.likelihood_6_lags.mean(),
        df1.likelihood_7_lags.mean(),
    ]
    mean_model_order = mean_across_sj.index(max(mean_across_sj))

    model_mean = (
        np.sum(df1['model_order'] == 'likelihood_no_hist') * 0
        + np.sum(df1['model_order'] == 'likelihood_1_lags') * 1
        + np.sum(df1['model_order'] == 'likelihood_2_lags') * 2
        + np.sum(df1['model_order'] == 'likelihood_3_lags') * 3
        + np.sum(df1['model_order'] == 'likelihood_4_lags') * 4
        + np.sum(df1['model_order'] == 'likelihood_5_lags') * 5
        + np.sum(df1['model_order'] == 'likelihood_6_lags') * 6
        + np.sum(df1['model_order'] == 'likelihood_7_lags') * 7
    ) / (
        np.sum(df1['model_order'] == 'likelihood_no_hist')
        + np.sum(df1['model_order'] == 'likelihood_1_lags')
        + np.sum(df1['model_order'] == 'likelihood_2_lags')
        + np.sum(df1['model_order'] == 'likelihood_3_lags')
        + np.sum(df1['model_order'] == 'likelihood_4_lags')
        + np.sum(df1['model_order'] == 'likelihood_5_lags')
        + np.sum(df1['model_order'] == 'likelihood_6_lags')
        + np.sum(df1['model_order'] == 'likelihood_7_lags')
    )

    ax.bar(x1, y1, fill=True, color=col, width=0.5, label=lab)  # ,
    ax.plot([mean_model_order, mean_model_order], [0, 12], color='gray', linestyle='-.', linewidth=1)
    ax.plot([model_mean, model_mean], [0, 12], color='k', linestyle='--', linewidth=1)
    ax.set_title(lab)

    sns.despine(ax=ax, offset=10, right=True, left=False)
    ax.set_xticklabels(my_xticks, fontsize=6)
    ax.set_ylim(0, 16)
    ax.set_xticks(x1)


plot_ll(df1=df1_repetitive, col='g', lab='Repetitive', ax=ax1)
plot_ll(df1=df1_neutral, col='r', lab='Neutral', ax=ax2)
plot_ll(df1=df1_alternating, col='b', lab='Alternating', ax=ax3)

ax2.set_xlabel('Model order/Number of lags')
ax1.set_ylabel('Number of subjects')
#plt.savefig('model_comp_crossval.pdf')
plt.show()
