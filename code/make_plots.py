"""
Makes figures 1 to 5 from results.json (made by run_experiment.py).

  fig1_switching.png   trials to switch after the swap, for each mood.
                       One panel per reward pair. Shaded band = 95% confidence
                       interval (CI) of the average.
  fig2_best_mood.png   the mood that switches fastest for each pair and model.
                       Bar = all moods that are not clearly slower: each run is
                       compared with the same run at the fastest mood, and the
                       95% CI of that difference still includes 0.
  fig3_vs_neutral.png  change in trials to switch compared with neutral mood,
                       worked out run by run (each run is compared with the same
                       run at mood 0). Main figure of the paper.
  fig4_dopamine.png    average size of the dopamine signal on rewarded (solid)
                       and missed (dashed) trials, after the swap only.

  fig5_switched_and_accuracy.png  share of runs that switched at all, and
                       accuracy in the last 50 trials.
(fig6_button_values.png is made by make_fig6_values.py.)

How the 95% CI is found: pick 50 runs at random (repeats allowed), take the
average, do this 2,000 times, and keep the middle 95% of those averages.
(Older results.json files without per-run values fall back to
average +/- 1.96 x standard error, and Fig 3 is skipped.)

Run it from this folder:  py make_plots.py   (or python3 on Mac/Linux)
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


HAS_RUNS = 'runs' in R[PAIRS[0]]['BeliefTD'][0]
BOOT = np.random.default_rng(0)


def boot_ci(x, n=2000):
    """95% bootstrap CI of the mean of x (NaNs dropped)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    means = BOOT.choice(x, size=(n, len(x))).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def runs(pair, model, key):
    """array (moods x runs) of per-run values"""
    return np.array([r['runs'][key] for r in R[pair][model]], float)


def mean_ci(pair, model, key):
    m, se = series(pair, model, key)
    if HAS_RUNS:
        lo, hi = np.array([boot_ci(v) for v in runs(pair, model, key)]).T
    else:
        lo, hi = m - 1.96 * se, m + 1.96 * se
    return m, lo, hi


def band(ax, m, lo, hi, col, ls='-', label=None, ms=4):
    ax.fill_between(MOOD, lo, hi, color=col, alpha=0.18, lw=0)
    return ax.plot(MOOD, m, color=col, lw=2, ls=ls, marker='o', ms=ms, label=label)[0]


def panel_grid(title):
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharex=True, sharey=True)
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
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles),
               bbox_to_anchor=(0.5, 0.0), handlelength=3)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print('saved', os.path.normpath(path))


# ---- Fig 1: trials to switch ----
fig, axs = panel_grid('Fig 1. Trials needed to switch to the new better button after the swap (lower = faster)\n'
                      'shaded band = 95% confidence interval')
handles = []
for ax, pair in zip(axs, PAIRS):
    for model, (label, col) in MODELS.items():
        h = band(ax, *mean_ci(pair, model, 'adapt'), col, label=label)
        if pair == PAIRS[0]:
            handles.append(h)
finish(fig, axs, 'trials to switch', 'fig1_switching.png', handles)

# ---- Fig 2: best mood per pair ----
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(PAIRS))
offset = {'BeliefTD': -0.15, 'ANCCR': 0.15}
for model, (label, col) in MODELS.items():
    for i, p in enumerate(PAIRS):
        m, lo, hi = mean_ci(p, model, 'adapt')
        b = int(np.argmin(m))
        if HAS_RUNS:
            # paired: compare each run with the same run at the fastest mood; a mood
            # counts as "about as fast" if the 95% CI of that difference includes 0
            xr = runs(p, model, 'adapt')
            ok = np.array([k == b or boot_ci(xr[k] - xr[b])[0] <= 0
                           for k in range(len(MOOD))])
        else:
            ok = m <= hi[b]                     # fallback: inside the fastest mood's CI
        near = MOOD[ok]
        ax.plot([x[i] + offset[model]] * 2, [near.min(), near.max()], color=col,
                lw=6, alpha=0.3, solid_capstyle='round')
        ax.plot(x[i] + offset[model], MOOD[b], ls='none', marker='o', ms=10,
                color=col, markeredgecolor='white', markeredgewidth=2,
                label=label if i == 0 else None)
ax.axhline(0, color='#c3c2b7', lw=0.8, zorder=0)
ax.set_xticks(x)
ax.set_xticklabels([p.replace('-', '% vs ') + '%' for p in PAIRS])
ax.set_yticks(range(-5, 6))
ax.set_ylim(-5.6, 5.6)
ax.set_xlabel('reward pair (rare rewards → common rewards)')
ax.set_ylabel('mood that switches fastest')
ax.set_title('Fig 2. Which mood switches fastest, for each model\n'
             'dot = fastest mood, bar = moods not clearly slower (paired 95% CI includes 0)',
             loc='left', fontsize=11, color=INK)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
fig.tight_layout()
path = os.path.join(OUT, 'fig2_best_mood.png')
fig.savefig(path, dpi=200)
plt.close(fig)
print('saved', os.path.normpath(path))


# ---- Fig 3: paired change vs neutral mood (needs per-run values) ----
if HAS_RUNS:
    i0 = int(np.where(MOOD == 0)[0][0])
    fig, axs = panel_grid('Fig 3. Change in trials to switch compared with neutral mood (below 0 = faster than neutral)\n'
                          'each run is compared with the same run at mood 0 (paired); band = 95% CI')
    handles = []
    for ax, pair in zip(axs, PAIRS):
        ax.axhline(0, color='#c3c2b7', lw=0.8, zorder=0)
        for model, (label, col) in MODELS.items():
            x = runs(pair, model, 'adapt')
            d = x - x[i0]                       # paired difference, per run
            lo, hi = np.array([boot_ci(v) if k != i0 else (0.0, 0.0)
                               for k, v in enumerate(d)]).T
            h = band(ax, d.mean(axis=1), lo, hi, col, label=label)
            if pair == PAIRS[0]:
                handles.append(h)
    finish(fig, axs, 'change vs mood 0 (trials)', 'fig3_vs_neutral.png', handles)
else:
    print('fig3 skipped: re-run run_experiment.py to save per-run values')


# ---- Fig 4: dopamine ----
fig, axs = panel_grid('Fig 4. Average size of the dopamine (surprise) signal in the 100 trials after the swap\n'
                      'solid = trials that gave a reward, dashed = trials that missed, band = 95% CI')
handles = []
for ax, pair in zip(axs, PAIRS):
    for model, (label, col) in MODELS.items():
        for key, ls, kind in (('da_good', '-', 'reward'), ('da_bad', '--', 'miss')):
            h = band(ax, *mean_ci(pair, model, key), col, ls=ls, ms=3.5,
                     label=f'{label}, {kind}')
            if pair == PAIRS[0]:
                handles.append(h)
finish(fig, axs, '|DA|', 'fig4_dopamine.png', handles)



# ---- Fig 5: did the model really switch? ----
# "Trials to switch" is capped at 100 and can be passed by luck, so this figure
# shows two plainer things: the share of runs that switched at all, and how
# often the model picked the new better button in the last 50 trials.
if HAS_RUNS:
    fig, axs = panel_grid('Fig 5. Share of runs that switched (solid) and accuracy in the last 50 trials (dashed)\n'
                          'band = 95% CI')
    handles = []
    for ax, pair in zip(axs, PAIRS):
        for model, (label, col) in MODELS.items():
            sw = (runs(pair, model, 'adapt') < 100).astype(float)   # 100 = never switched
            acc = runs(pair, model, 'post_acc')
            for x, ls, kind in ((sw, '-', 'runs that switched'), (acc, '--', 'final accuracy')):
                lo, hi = np.array([boot_ci(v) for v in x]).T
                h = band(ax, x.mean(axis=1), lo, hi, col, ls=ls, ms=3.5, label=f'{label}, {kind}')
                if pair == PAIRS[0]:
                    handles.append(h)
        ax.set_ylim(0, 1.05)
    finish(fig, axs, 'share (0 to 1)', 'fig5_switched_and_accuracy.png', handles)
