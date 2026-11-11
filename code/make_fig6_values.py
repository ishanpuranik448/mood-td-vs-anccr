"""
Fig 6: what each model thinks the two buttons are worth, trial by trial.

Runs both models on the 30% vs 80% pair at moods -5, 0 and +5 (50 runs each,
same seeds as the main experiment) and plots the average value of the old
button and the new button. This is the picture behind the Discussion: it shows
why low mood stops ANCCR from telling the buttons apart, and why high mood
keeps TD on the old button.

Run it from this folder:  py make_fig6_values.py   (takes about a minute)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.argv = sys.argv[:1]                      # main settings
import run_experiment as E
from anccr_model import ANCCR
from td_model import BeliefTD

PAIR, SEED = (0.3, 0.8), E.PAIRS.index((0.3, 0.8))
MOODS, N = (-5, 0, 5), 50


def td_values(ratio, rng):
    ap, am = E.split_rates(E.TD_ALPHA, ratio)
    ag = BeliefTD(*E.td_task_model(PAIR, True), E.TD_ALPHA, E.TD_ALPHA, E.TD_GAMMA)
    Q = []
    for t in range(E.N_TRIALS):
        if t == E.N_BASE:
            ag.ap, ag.am = ap, am
        q = [ag.value(ag.belief_after(E.PRESS1 + b)[0]) for b in (0, 1)]
        Q.append(q)
        a = E.softmax_pick(q, E.TD_BETA, rng)
        ag.step(E.PRESS1 + a, 0.0)
        r = float(rng.random() < E.p_reward(a, t, PAIR))
        ag.step(E.REWARD_EV if r else E.NOREWARD_EV, r)
    return np.array(Q)


def anccr_values(ratio, rng):
    p = E.ANCCR_PARAMS
    ap, am = E.split_rates(p['alpha'], ratio)
    ag = ANCCR(4, E.ANCCR_T, p['alpha'], p['k'], E.SAMPLING, p['w'], p['threshold'],
               E.MIN_RATE, beta=[0, 0, 1, 0], alpha_r_plus=p['alpha_r'],
               alpha_r_minus=p['alpha_r'], omidx=[E.A_OMIT, E.A_REWARD])
    ag.omtrue[:] = True
    Q, now = [], 0.0
    for t in range(E.N_TRIALS):
        if t == E.N_BASE:
            ag.ap, ag.am = ap, am
        q = [ag.R[b, E.A_REWARD] * (ag.NC[b, E.A_REWARD] - ag.NC[b, E.A_OMIT]) for b in (0, 1)]
        Q.append(q)
        a = E.softmax_pick(q, E.ANCCR_BETA, rng)
        ag.step(E.A_PRESS1 + a, now, 0.0)
        now += E.CUE_REWARD_DELAY
        r = float(rng.random() < E.p_reward(a, t, PAIR))
        ag.step(E.A_REWARD if r else E.A_OMIT, now, r)
        now += E.iti(rng)
    return np.array(Q)


INK, OLD, NEW = '#0b0b0b', '#898781', '#1a9b5f'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e1e0d9', 'grid.linewidth': 0.6,
                     'legend.frameon': False})
fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True, sharey='row')
x = np.arange(E.N_TRIALS) - E.N_BASE
for row, (name, f) in enumerate((('TD (belief state)', td_values), ('ANCCR', anccr_values))):
    for col, m in enumerate(MOODS):
        Q = np.mean([f(4.0 ** (-m / 5), np.random.default_rng([SEED, k])) for k in range(N)], axis=0)
        ax = axes[row, col]
        h1, = ax.plot(x, Q[:, 0], color=OLD, lw=2, label='old button (better before the swap)')
        h2, = ax.plot(x, Q[:, 1], color=NEW, lw=2, label='new button (better after the swap)')
        ax.axvline(0, color=INK, lw=0.8, ls=':')
        ax.set_title(f'{name}, mood {m:+d}'.replace('+0', '0'), fontsize=10, loc='left')
        if row == 1:
            ax.set_xlabel('trials since the swap')
        if col == 0:
            ax.set_ylabel('button value')
fig.legend(handles=[h1, h2], loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.06, 1, 1))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'fig6_button_values.png')
fig.savefig(out, dpi=200)
print('saved', os.path.normpath(out))
