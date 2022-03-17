"""
    This script is used to generate figures
"""

import numpy as np
from matplotlib import pyplot as plt


from cycler import cycler
import matplotlib as mpl

# style
plt.style.use('fivethirtyeight')

mpl.rcParams['axes.prop_cycle'] = cycler(color=
    ['#3887be', '#00ab6c', '#fbb03b', '#8a8acb', '#91afa5', '#e55e5e',])

params={
        'font.family': 'Linux Biolinum O',
        'font.weight': 'regular',
        'legend.fontsize': 13,
        'savefig.dpi': 150,
        'figure.dpi': 150,
        'axes.labelsize': 15,
        'font.size': 15,
        'axes.titlesize': 17,
        'savefig.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'white',
        'mathtext.rm': 'serif',
        'mathtext.fontset': 'custom',
        'text.usetex': False,
        'figure.autolayout': True,
        'savefig.edgecolor': '#000000',
        'savefig.facecolor': '#FFFFFF',
        #'savefig.frameon': False,
        'savefig.transparent': False,
        'savefig.pad_inches': 0.1
        }

plt.rcParams.update(params)

print([(i, plt.rcParams[i])  for i in plt.rcParams if 'savefig' in i])

# data
data = {
    "name": ["F1", "Acc"],
    "Freq": [0.2214, 0.2179],
}

# figure
fig, ax = plt.subplots(1,4, figsize=(9,4), constrained_layout=True)
fig.tight_layout()

width = 0.1
x = np.arange(len(data['name']))

offset = - 3 * width

# plot
ax[0].bar(x+width*0+offset, data['Freq'], width*0.8, label='Freq')
ax[0].set_xticks(x)
ax[0].set_xticklabels(data['name'])
ax[0].set_title('Steam')

ax[1].bar(x+width*0+offset, data['Freq'], width*0.8, label='Freq')
ax[1].set_xticks(x)
ax[1].set_xticklabels(data['name'])
ax[1].set_title('Electronics')

ax[2].bar(x+width*0+offset, data['Freq'], width*0.8, label='Freq')
ax[2].set_xticks(x)
ax[2].set_xticklabels(data['name'])
ax[2].set_title('Clothing')

ax[3].bar(x+width*0+offset, data['Freq'], width*0.8, label='Freq')
ax[3].set_xticks(x)
ax[3].set_xticklabels(data['name'])
ax[3].set_title('Home')

# others
handles, labels = ax[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=6, bbox_to_anchor=(0.5, .07))

plt.savefig('todo.pdf', bbox_inches='tight')
plt.show()
