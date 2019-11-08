import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import numpy as np

sns.set()
sns.set_context("notebook")

metrics = ['SDR', 'SIR', 'SAR', 'ISR']
targets = ['vocals', 'acc']
selected_targets = ['vocals', 'accompaniment']
oracles = [
    'IBM1', 'IBM2', 'IRM1', 'IRM2', 'MWF', 'IMSK'
]

# Convert to Pandas Dataframes
df = pd.read_pickle("sisec18_mus.pandas")
df['oracle'] = df.method.isin(oracles)
# df = df[df.target.isin(selected_targets)].dropna()

# aggregate methods by mean using median by track
df = df.groupby(
    ['method', 'track', 'target', 'metric']
).median().reset_index()
df.loc[df['target'] == 'accompaniment', 'target'] = 'acc'

# Get sorting keys (sorted by median of SDR:vocals)
df_sort_by = df[
    (df.metric == "SDR") &
    (df.target == "vocals")
]

# # Open URL
# raw_data = urlopen(url)
# Load into Pandas Dataframes
df_1 = pd.read_pickle("attention.pd")
# destingish oracles from non-oracle methods
df_1['oracle'] = df_1.method.isin(oracles)

# Aggregate all frames of each track using the median
# Note that `nan` values are skipped by default
df_1['method'] = 'Attention'
df_1 = df_1.groupby(
    ['method', 'track', 'target', 'metric']
).median().reset_index()
df_1.loc[df_1['target'] == 'accompaniment', 'target'] = 'acc'
# Get sorting keys (sorted by median of SDR:vocals)
df_1_sort_by = df_1[
    (df_1.metric == "SDR") &
    (df_1.target == "vocals")
]

frames = [df_sort_by, df_1_sort_by]
result = pd.concat(frames)
methods_by_sdr = result.score.groupby(
    result.method
).median().sort_values().index.tolist()

frames_1 = [df, df_1]
result_1 = pd.concat(frames_1)

# plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.unicode'] = 'True'

sns.set()
sns.set_context("paper")

params = {
    'backend': 'ps',
    'axes.labelsize': 18,
    'font.size': 15,
    'legend.fontsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 15,
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'ptmrr8re',
}

sns.set_style("darkgrid", {
    'pgf.texsystem': 'xelatex',  # pdflatex, xelatex, lualatex
    "axes.facecolor": "0.925",
    'text.usetex': True,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'font.size': 14,
    'legend.fontsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 17,
    'font.serif': [],
})
plt.rcParams.update(params)

g = sns.FacetGrid(
    result_1,
    row="target",
    col="metric",
    row_order=targets,
    col_order=metrics,
    size=6,
    sharex=False,
    aspect=0.7
)
g = (g.map(
    sns.boxplot,
    "score",
    "method",
    "oracle",
    orient='h',
    order=methods_by_sdr[::-1],
    hue_order=[True, False],
    showfliers=False,
    notch=True
))

g.fig.tight_layout()
plt.subplots_adjust(hspace=0.2, wspace=0.1)
g.fig.savefig(
    "boxplot.pdf",
    bbox_inches='tight',
    dpi=300
)
