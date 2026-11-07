"""
Runs the whole experiment and saves the results to results.json.

The task
  - Two buttons. Each one gives a reward with its own chance.
  - 100 trials, then the two chances secretly swap, then 100 more trials.
    Button 1 is the better one at the start.
  - Six pairs of chances (percent): 0/50, 10/60, 20/70, 30/80, 40/90, 50/100.
    The gap is always 50 points, so only how common rewards are changes.
  - Each trial: the model presses a button, the reward (if any) comes 3.4 s
    later, then there is a 4 to 10 s wait (about 6.5 s on average), drawn the
    same way as in the original simulations.

Mood
  - Mood m goes from -5 (very sad) to +5 (very happy).
  - It sets the balance between learning from bad surprises (alpha_minus) and
    good surprises (alpha_plus): alpha_minus / alpha_plus = 4 ** (-m / 5).
    So m = -5 learns 4x more from bad surprises, m = +5 4x more from good ones.
  - The two rates always add up to 0.2, so m = 0 gives 0.1 each.
  - The first 100 trials are always at neutral mood. Mood turns on at the swap.

How each model picks a button (the original models never choose anything)
  - TD:    a button's value = how good the model expects things to be after
           pressing it.
  - ANCCR: a button's value = reward worth x (link to reward - link to miss).
  - Both then pick with a softmax rule (beta = 5 for TD, 10 for ANCCR).

What gets measured (per run)
  - adapt:     trials after the swap before the first stretch of 10 trials with
               at least 7 presses on the new better button (100 if never).
  - pre_acc:   share of presses on the better button in the 50 trials before
               the swap. post_acc: the same for the last 50 trials.
  - da_good / da_bad: average size of the dopamine signal on rewarded / missed
               trials, over all 200 trials.

Run it from this folder:  py run_experiment.py   (or python3 on Mac/Linux)
The full run takes about 6 to 8 minutes. Set N_RUNS = 5 for a quick test.

Fairness controls (each saves its own results_<name>.json, the main run is untouched)
  py run_experiment.py anccr_beta5      ANCCR picks with beta = 5, like TD
  py run_experiment.py td_no_hint       TD is not told swaps can happen (one
                                        context only, no swap chance)
  py run_experiment.py td_outcome_split TD splits learning by reward vs miss,
                                        like ANCCR, not by the sign of the surprise
  py run_experiment.py all_fair         all three at once
"""
import json, os, sys
import numpy as np
from td_model import BeliefTD
from anccr_model import ANCCR

OUT = os.path.dirname(os.path.abspath(__file__))

# ---- task ----
N_BASE, N_POST = 100, 100
N_TRIALS = N_BASE + N_POST
PAIRS = [(0.0, 0.5), (0.5, 1.0), (0.1, 0.6), (0.2, 0.7), (0.3, 0.8), (0.4, 0.9)]
CUE_REWARD_DELAY = 3.4
N_RUNS = 50                                  # runs per model x mood x pair
MOOD = np.arange(-5, 6)
RATIOS = 4.0 ** (-MOOD / 5)                  # alpha_minus / alpha_plus

# ---- TD settings ----
# alpha (learning rate) is from the original. gamma (how much future rewards
# matter) is inside the range the original tried (0.5 to 0.975).
# beta = how strongly the model picks the higher-value button.
TD_ALPHA, TD_GAMMA, TD_BETA = 0.1, 0.9, 5.0
# Chance of a swap per trial that TD assumes (one swap in 200 trials).
# Note: TD is given this hint, ANCCR has nothing like it.
HAZARD = 1 / N_TRIALS

# ---- ANCCR settings (from the original's settings unless noted) ----
# alpha = learning rate for the "link" (how strongly a button leads to reward).
# We raised it from 0.05 to 0.1 so both models learn at the same overall speed.
# Mood splits it so alpha_plus + alpha_minus = 0.2, like TD.
# alpha_r = learning rate for reward worth. Mood does not touch it.
ANCCR_PARAMS = dict(alpha=0.1, k=0.05, w=0.5, threshold=0.3, alpha_r=0.1)
# Memory window T (seconds). The original sets T = 1.2 x the average time
# between rewards. Here that is about 1 to 5 trials, so a reward gets credited
# to presses of both buttons and the model can't tell which one worked.
# We set T to about one trial instead (3.4 s wait + 6.5 s average gap).
ANCCR_T = CUE_REWARD_DELAY + 6.5
# Higher than TD's 5 because the gap between ANCCR's two button values is
# 2 to 6 times smaller than TD's. With beta = 5 it would pick the better
# button much less often. (This beta is only for choosing; it is not the
# beta inside the ANCCR model.)
ANCCR_BETA = 10.0
SAMPLING, MIN_RATE = 0.2, 0.001              # as in run_anccr_model.m

# ---- fairness controls (see the top of this file) ----
CONTROLS = {'anccr_beta5', 'td_no_hint', 'td_outcome_split'}
CONTROL = sys.argv[1] if len(sys.argv) > 1 else 'main'
ON = CONTROLS if CONTROL == 'all_fair' else {CONTROL} & CONTROLS
if CONTROL != 'main' and not ON:
    sys.exit(f"Unknown control '{CONTROL}'. Use one of: {', '.join(sorted(CONTROLS))}, all_fair")
if 'anccr_beta5' in ON:
    ANCCR_BETA = TD_BETA
TD_HINT = 'td_no_hint' not in ON
TD_SPLIT = 'outcome' if 'td_outcome_split' in ON else 'sign'


def split_rates(alpha, ratio):
    """Two learning rates with mean alpha and alpha_minus / alpha_plus = ratio."""
    ap = 2 * alpha / (1 + ratio)
    return ap, 2 * alpha - ap                # alpha_plus, alpha_minus


def iti(rng):
    x = rng.exponential(6)
    while x < 4 or x > 10:
        x = rng.exponential(6)
    return x


def p_reward(button, trial, pair):
    lo, hi = pair
    good = 0 if trial < N_BASE else 1        # button 1 (index 0) is better at first
    return hi if button == good else lo


def softmax_pick(q, beta, rng):
    p1 = 1 / (1 + np.exp(-beta * (q[1] - q[0])))
    return int(rng.random() < p1)


# ---- BeliefTD on the two-button task ----
# Hidden states: context c (0 = button 1 better, 1 = button 2 better) x
# phase (0 = waiting, 1 = just pressed button 1, 2 = just pressed button 2).
WAIT_EV, PRESS1, PRESS2, REWARD_EV, NOREWARD_EV = range(5)


def td_task_model(pair, hint=True):
    """hint=False (control): one context only, so TD has no idea that a swap
    can happen. It is then plain TD over wait / pressed 1 / pressed 2."""
    lo, hi = pair
    C = 2 if hint else 1
    S, E = 3 * C, 5
    sid = lambda c, ph: 3 * c + ph
    T, O = np.zeros((S, S)), np.zeros((S, S, E))
    for c in range(C):
        # with one context the outcome tells TD nothing about the state, so
        # any non-zero chance works; 0.5 avoids a zero-chance belief reset
        probs = ((hi, lo) if c == 0 else (lo, hi)) if hint else (0.5, 0.5)
        for b in (0, 1):
            T[sid(c, 0), sid(c, 1 + b)] = 0.5
            O[sid(c, 0), sid(c, 1 + b), PRESS1 + b] = 1
            for c2 in range(C):
                stay = 1 - HAZARD if hint else 1.0
                T[sid(c, 1 + b), sid(c2, 0)] = stay if c2 == c else HAZARD
                O[sid(c, 1 + b), sid(c2, 0), REWARD_EV] = probs[b]
                O[sid(c, 1 + b), sid(c2, 0), NOREWARD_EV] = 1 - probs[b]
    return T, O


def run_belief_td(ratio, pair, rng):
    ap, am = split_rates(TD_ALPHA, ratio)
    agent = BeliefTD(*td_task_model(pair, TD_HINT), TD_ALPHA, TD_ALPHA, TD_GAMMA,
                     split=TD_SPLIT)                                       # neutral
    rec = dict(choice=[], reward=[], da=[])
    for t in range(N_TRIALS):
        if t == N_BASE:                          # swap: mood switches on
            agent.ap, agent.am = ap, am
        q = [agent.value(agent.belief_after(PRESS1 + b)[0]) for b in (0, 1)]
        a = softmax_pick(q, TD_BETA, rng)
        agent.step(PRESS1 + a, 0.0)
        r = float(rng.random() < p_reward(a, t, pair))
        da = agent.step(REWARD_EV if r else NOREWARD_EV, r, outcome=int(r))
        rec['choice'].append(a); rec['reward'].append(r); rec['da'].append(da)
    return {k: np.array(v) for k, v in rec.items()}


# ---- ANCCR on the two-button task ----
# Events: 0 = pressed button 1, 1 = pressed button 2, 2 = reward, 3 = no reward
A_PRESS1, A_PRESS2, A_REWARD, A_OMIT = range(4)


def run_anccr(ratio, pair, rng):
    p = ANCCR_PARAMS
    ap, am = split_rates(p['alpha'], ratio)
    agent = ANCCR(4, ANCCR_T, p['alpha'], p['k'], SAMPLING, p['w'], p['threshold'],
                  MIN_RATE, beta=[0, 0, 1, 0], alpha_r_plus=p['alpha_r'],
                  alpha_r_minus=p['alpha_r'], omidx=[A_OMIT, A_REWARD])  # neutral
    # In this task a missed reward always means something, so misses count
    # from the first trial (the original waits until reward has a known cause).
    agent.omtrue[:] = True
    rec = dict(choice=[], reward=[], da=[])
    now = 0.0
    for t in range(N_TRIALS):
        if t == N_BASE:                          # swap: mood switches on
            agent.ap, agent.am = ap, am
        q = [agent.R[b, A_REWARD] * (agent.NC[b, A_REWARD] - agent.NC[b, A_OMIT])
             for b in (0, 1)]
        a = softmax_pick(q, ANCCR_BETA, rng)
        agent.step(A_PRESS1 + a, now, 0.0)
        now += CUE_REWARD_DELAY
        r = float(rng.random() < p_reward(a, t, pair))
        da = agent.step(A_REWARD if r else A_OMIT, now, r)
        now += iti(rng)
        rec['choice'].append(a); rec['reward'].append(r); rec['da'].append(da)
    return {k: np.array(v) for k, v in rec.items()}


# ---- measures ----
def adapt_trials(choice):
    """Trials after the swap until a 10-trial window picks the new better button >= 70%."""
    post = choice[N_BASE:]
    for i in range(len(post) - 9):           # windows starting at 0..90
        if (post[i:i + 10] == 1).mean() >= 0.7:
            return i
    return len(post)


def summarise(runner, pair, seed):
    out = []
    for ratio in RATIOS:
        m = dict(pre_acc=[], post_acc=[], adapt=[], da_good=[], da_bad=[])
        for run in range(N_RUNS):
            # one seed per (pair, run): run k sees the same random numbers at every
            # mood, so moods can be compared run-by-run (paired)
            rng = np.random.default_rng([seed, run])
            x = runner(ratio, pair, rng)
            m['pre_acc'].append((x['choice'][N_BASE - 50:N_BASE] == 0).mean())
            m['post_acc'].append((x['choice'][-50:] == 1).mean())
            m['adapt'].append(adapt_trials(x['choice']))
            good, bad = x['da'][x['reward'] == 1], x['da'][x['reward'] == 0]
            m['da_good'].append(np.abs(good).mean() if len(good) else np.nan)
            m['da_bad'].append(np.abs(bad).mean() if len(bad) else np.nan)
        row = {k: (float(np.nanmean(v)), float(np.nanstd(v) / np.sqrt(N_RUNS)))
               for k, v in m.items()}
        row['runs'] = {k: [float(a) for a in v] for k, v in m.items()}  # per-run values, for CIs
        out.append(row)
    return out


if __name__ == '__main__':
    models = {'BeliefTD': run_belief_td, 'ANCCR': run_anccr}
    results = {'mood': MOOD.tolist(), 'ratios': RATIOS.tolist(), 'pairs': PAIRS,
               'control': CONTROL}
    print(f"Control: {CONTROL}  (ANCCR beta {ANCCR_BETA}, TD swap hint {TD_HINT}, "
          f"TD split by {TD_SPLIT})")
    for i, pair in enumerate(PAIRS):
        key = f"{int(pair[0] * 100)}-{int(pair[1] * 100)}"
        results[key] = {name: summarise(run, pair, seed=i) for name, run in models.items()}
        print(f"== pair {key} %")
        for name in models:
            print(name)
            for mo, s in zip(MOOD, results[key][name]):
                print(f"  mood {mo:+d}  pre {s['pre_acc'][0]:.2f}  post {s['post_acc'][0]:.2f}  "
                      f"adapt {s['adapt'][0]:5.1f}  |DA good| {s['da_good'][0]:.3f}  "
                      f"|DA bad| {s['da_bad'][0]:.3f}")
    name = 'results.json' if CONTROL == 'main' else f'results_{CONTROL}.json'
    with open(os.path.join(OUT, name), 'w') as f:
        json.dump(results, f)
