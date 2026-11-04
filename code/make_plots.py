"""
Makes the three figures from results.json (written by run_experiment.py).

  fig1_switching.png  trials to switch after the swap vs mood, one panel per pair
  fig2_dopamine.png   average |DA| on rewarded (solid) and missed (dashed) trials
  fig3_best_mood.png  the mood that switches fastest, per pair, for each model

Run:  py make_plots.py      (needs matplotlib: py -m pip install matplotlib)
Figures are saved in the figures/ folder next to code/.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'figures')
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(HERE, 'results.json')) as f:
    R = json.load(f)
MOOD = np.array(R['mood'])

# panels ordered from rarest to most common rewards
PAIRS = ['0-50', '10-60', '20-70', '30-80', '40-90', '50-100']
MODELS = {'BeliefTD': ('TD (belief state)', '#2a78d6'),
          'ANCCR': ('ANCCR', '#eb6834')}
INK, MUTED, GRID = '#0b0b0b', '#898781', '#e1e0d9'

plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'legend.frameon': False,
})


def series(pair, model, key):
    rows = R[pair][model]
    m = np.array([r[key][0] for r in rows])
    se = np.array([r[key][1] for r in rows])
    return m, se


def panel_grid(title):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True, sharey=True)
    fig.suptitle(title, x=0.01, ha='left', fontsize=12, color=INK)
    return fig, axes.ravel()


def finish(fig, ax_list, ylabel, name, legend_handles):
    for ax, pair in zip(ax_list, PAIRS):
        lo, hi = pair.split('-')
        ax.set_title(f'{lo}% vs {hi}%', fontsize=10, color=INK, loc='left')
        ax.set_xticks([-5, -3, -1, 0, 1, 3, 5])
        ax.axvline(0, color='#c3c2b7', lw=0.8, zorder=0)
    for ax in ax_list[3:]:
        ax.set_xlabel('mood (−5 sad … +5 happy)')
    for ax in ax_list[::3]:
        ax.set_ylabel(ylabel)
    fig.legend(handles=legend_handles, loc='upper right', ncol=len(legend_handles))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print('saved', os.path.normpath(path))


# ---- Fig 1: trials to switch ----
fig, axs = panel_grid('Fig 1. Trials needed to switch to the new better button after the swap (lower = faster)')
handles = []
for ax, pair in zip(axs, PAIRS):
    for model, (label, col) in MODELS.items():
        m, se = series(pair, model, 'adapt')
        ax.fill_between(MOOD, m - se, m + se, color=col, alpha=0.15, lw=0)
        h, = ax.plot(MOOD, m, color=col, lw=2, marker='o', ms=4, label=label)
        if pair == PAIRS[0]:
            handles.append(h)
finish(fig, axs, 'trials to switch', 'fig1_switching.png', handles)

# ---- Fig 2: dopamine ----
fig, axs = panel_grid('Fig 2. Average size of the dopamine (surprise) signal across all 200 trials')
handles = []
for ax, pair in zip(axs, PAIRS):
    for model, (label, col) in MODELS.items():
        for key, ls, kind in (('da_good', '-', 'reward'), ('da_bad', '--', 'miss')):
            m, se = series(pair, model, key)
            h, = ax.plot(MOOD, m, color=col, lw=2, ls=ls, marker='o', ms=3.5,
                         label=f'{label}, {kind}')
            if pair == PAIRS[0]:
                handles.append(h)
finish(fig, axs, '|DA|', 'fig2_dopamine.png', handles)

# ---- Fig 3: best mood per pair ----
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(PAIRS))
offset = {'BeliefTD': -0.12, 'ANCCR': 0.12}
for model, (label, col) in MODELS.items():
    best = [MOOD[int(np.argmin(series(p, model, 'adapt')[0]))] for p in PAIRS]
    ax.plot(x + offset[model], best, ls='none', marker='o', ms=10, color=col,
            markeredgecolor='white', markeredgewidth=2, label=label)
ax.axhline(0, color='#c3c2b7', lw=0.8, zorder=0)
ax.set_xticks(x)
ax.set_xticklabels([p.replace('-', '% vs ') + '%' for p in PAIRS])
ax.set_yticks(range(-5, 6))
ax.set_ylim(-5.6, 5.6)
ax.set_xlabel('reward pair (rare rewards → common rewards)')
ax.set_ylabel('mood that switches fastest')
ax.set_title('Fig 3. Which mood switches fastest, for each model', loc='left',
             fontsize=12, color=INK)
ax.legend(loc='upper left')
fig.tight_layout()
path = os.path.join(OUT, 'fig3_best_mood.png')
fig.savefig(path, dpi=200)
plt.close(fig)
print('saved', os.path.normpath(path))
