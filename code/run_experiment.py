"""
Two-button reversal task across an 11-point mood dial, for the two models
from Qian, Burrell et al. (2025) in td_model.py and anccr_model.py.

Task (from the draft's Experimental Design):
  - 100 baseline trials, then the two buttons' reward probabilities swap,
    then 100 more trials.
  - Six probability pairs: (0, 50), (50, 100), (10, 60), (20, 70), (30, 80),
    (40, 90) percent. Button 1 starts as the better one.
  - Each trial: press a button (time 0), outcome 3.4 s later (the cue-reward
    delay in the original simulations), then an inter-trial interval drawn
    the same way as the original (exponential, mean 6 s, kept within 4-10 s).

Mood dial: m = -5 ... +5. alpha_minus / alpha_plus = 4 ** (-m / 5), so m = -5
learns 4x more from bad outcomes and m = +5 learns 4x more from good ones.
The two rates average to the model's original single rate, so m = 0 is the
original model.

Choice rule (not part of the originals, which have no choices):
  BeliefTD: value of pressing a button = learned value of the belief the
            agent would have after pressing it.
  ANCCR:    value of pressing a button = net contingency between that button
            and reward, times the reward value it predicts (NC * R).
  Both use a softmax.

Run:  py run_experiment.py
Writes results.json and prints a summary table. Takes a while; lower N_RUNS
for a quick test.
"""
import json, os
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

# ---- BeliefTD settings (alpha from the original; gamma inside its 0.5-0.975 sweep) ----
TD_ALPHA, TD_GAMMA, TD_BETA = 0.1, 0.9, 5.0
HAZARD = 1 / N_TRIALS                        # assumed swap rate, as if estimated from the task

# ---- ANCCR settings (each value is from the original's parameter grid) ----
ANCCR_PARAMS = dict(Tratio=1.2, alpha=0.05, k=0.05, w=0.5, threshold=0.3, alpha_r=0.2)
ANCCR_BETA = 10.0
SAMPLING, MIN_RATE = 0.2, 0.001              # as in run_anccr_model.m


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


def td_task_model(pair):
    lo, hi = pair
    S, E = 6, 5
    sid = lambda c, ph: 3 * c + ph
    T, O = np.zeros((S, S)), np.zeros((S, S, E))
    for c in (0, 1):
        probs = (hi, lo) if c == 0 else (lo, hi)
        for b in (0, 1):
            T[sid(c, 0), sid(c, 1 + b)] = 0.5
            O[sid(c, 0), sid(c, 1 + b), PRESS1 + b] = 1
            for c2 in (0, 1):
                T[sid(c, 1 + b), sid(c2, 0)] = 1 - HAZARD if c2 == c else HAZARD
                O[sid(c, 1 + b), sid(c2, 0), REWARD_EV] = probs[b]
                O[sid(c, 1 + b), sid(c2, 0), NOREWARD_EV] = 1 - probs[b]
    return T, O


def run_belief_td(ratio, pair, rng):
    ap, am = split_rates(TD_ALPHA, ratio)
    agent = BeliefTD(*td_task_model(pair), ap, am, TD_GAMMA)
    rec = dict(choice=[], reward=[], da=[])
    for t in range(N_TRIALS):
        q = [agent.value(agent.belief_after(PRESS1 + b)[0]) for b in (0, 1)]
        a = softmax_pick(q, TD_BETA, rng)
        agent.step(PRESS1 + a, 0.0)
        r = float(rng.random() < p_reward(a, t, pair))
        da = agent.step(REWARD_EV if r else NOREWARD_EV, r)
        rec['choice'].append(a); rec['reward'].append(r); rec['da'].append(da)
    return {k: np.array(v) for k, v in rec.items()}


# ---- ANCCR on the two-button task ----
# Events: 0 = pressed button 1, 1 = pressed button 2, 2 = reward, 3 = no reward
A_PRESS1, A_PRESS2, A_REWARD, A_OMIT = range(4)


def run_anccr(ratio, pair, rng):
    p = ANCCR_PARAMS
    # T = mean inter-reward interval x Tratio (as in run_anccr_model.m), using
    # the interval expected when choosing at random
    trial_len = CUE_REWARD_DELAY + 6.5
    T = p['Tratio'] * trial_len / max(np.mean(pair), 1e-3)
    arp, arm = split_rates(p['alpha_r'], ratio)
    agent = ANCCR(4, T, p['alpha'], p['k'], SAMPLING, p['w'], p['threshold'],
                  MIN_RATE, beta=[0, 0, 1, 0], alpha_r_plus=arp, alpha_r_minus=arm,
                  omidx=[A_OMIT, A_REWARD])
    rec = dict(choice=[], reward=[], da=[])
    now = 0.0
    for t in range(N_TRIALS):
        q = [agent.NC[b, A_REWARD] * agent.R[b, A_REWARD] for b in (0, 1)]
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
    for i in range(len(post) - 10):
        if (post[i:i + 10] == 1).mean() >= 0.7:
            return i
    return len(post)


def summarise(runner, pair, seed):
    out = []
    for ratio in RATIOS:
        rng = np.random.default_rng(seed)
        m = dict(pre_acc=[], post_acc=[], adapt=[], da_good=[], da_bad=[])
        for _ in range(N_RUNS):
            x = runner(ratio, pair, rng)
            m['pre_acc'].append((x['choice'][N_BASE - 50:N_BASE] == 0).mean())
            m['post_acc'].append((x['choice'][-50:] == 1).mean())
            m['adapt'].append(adapt_trials(x['choice']))
            good, bad = x['da'][x['reward'] == 1], x['da'][x['reward'] == 0]
            m['da_good'].append(np.abs(good).mean() if len(good) else np.nan)
            m['da_bad'].append(np.abs(bad).mean() if len(bad) else np.nan)
        out.append({k: (float(np.nanmean(v)), float(np.nanstd(v) / np.sqrt(N_RUNS)))
                    for k, v in m.items()})
    return out


if __name__ == '__main__':
    models = {'BeliefTD': run_belief_td, 'ANCCR': run_anccr}
    results = {'mood': MOOD.tolist(), 'ratios': RATIOS.tolist(), 'pairs': PAIRS}
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
    with open(os.path.join(OUT, 'results.json'), 'w') as f:
        json.dump(results, f)
