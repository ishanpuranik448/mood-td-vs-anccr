# Mood and two dopamine models: TD vs ANCCR

**Question:** when mood goes from low to high, how differently do two models of dopamine learning (TD and ANCCR) react to good and bad surprises?

This repo has the code, results and figures for our simulation study. There is no human or animal data, only simulated agents.

## What the experiment does

- **Task.** There are two buttons, and each one gives a reward with its own chance. After 100 trials the two chances secretly swap, and the model gets 100 more trials to notice and switch.
- **Reward pairs.** We use six pairs (% chance of reward): 0/50, 10/60, 20/70, 30/80, 40/90 and 50/100. The gap is always 50 points, so only how common rewards are changes.
- **Mood.** Mood goes from −5 (very sad) to +5 (very happy). It sets how much the model learns from bad surprises compared with good ones: α− / α+ = 4^(−m/5). The two rates always add up to 0.2. Mood turns on at the swap; the first 100 trials are at neutral mood.
- **Runs.** Each model runs 50 times at every mood and pair. That is 2 models × 11 moods × 6 pairs × 50 runs = 6,600 simulated agents.

## The two models

Both models are translated from MATLAB to Python, starting from the code for Qian et al. (2025):

- **Belief-state TD** (`code/td_model.py`), from `td_belief.m` (original author S. J. Gershman). TD learns by comparing the reward it got with the reward it expected. The belief-state version also keeps a guess about whether the buttons have swapped.
- **ANCCR** (`code/anccr_model.py`), from `calculateANCCRperformance.m`, which is a faster copy of the official ANCCR code by Jeong et al. (2022). ANCCR looks back after each reward and works out what caused it.

### What we changed

**TD:** the only change is mood. It uses α+ for positive surprises and α− for negative ones, instead of one learning rate. With α+ = α− you get the original model back.

**ANCCR:** the original model was built for animals watching cues, so it needed a few changes for a button task:

1. **Mood is on the "link"** (how strongly a button leads to reward). The link uses α+ when a press is rewarded and α− when it is not. Mood does not change how much a reward is worth.
2. **Shorter memory window**, about one trial (9.9 s). The original window covered several trials, so a reward got credited to both buttons.
3. **Misses count from trial 1.** In the original, misses only count once the model has learned what causes a reward.
4. **A way to choose buttons.** We define button value = reward worth × (link to reward − link to miss), then pick a button with a softmax rule.

We also raised ANCCR's link learning rate from 0.05 to 0.1 so both models learn at the same overall speed, and used β = 10 for ANCCR's choices (TD uses 5) because ANCCR's button values are much closer together. All the settings are at the top of `code/run_experiment.py`, with comments.

## How to run it

You need Python 3 (we used 3.13).

```
py -m pip install -r requirements.txt
cd code
py run_experiment.py
py make_plots.py
```

On Mac or Linux, type `python3` instead of `py`.

- `run_experiment.py` runs everything and saves `code/results.json`. It takes about 6 to 8 minutes. For a quick test, set `N_RUNS = 5`.
- `make_plots.py` reads `results.json` and saves five figures in `figures/`.

### Fairness checks

The two models are not set up in exactly the same way, so `run_experiment.py` can also rerun everything with one of these changes. Each saves its own `code/results_<name>.json` and leaves `results.json` alone.

```
py run_experiment.py anccr_beta5       # ANCCR picks with beta = 5, like TD
py run_experiment.py td_no_hint        # TD is not told that swaps can happen
py run_experiment.py td_outcome_split  # TD splits learning by reward vs miss, like ANCCR
py run_experiment.py all_fair          # all three at once
```

Every run has its own fixed random seed, so you get the same numbers each time. Run k uses the same seed at every mood, so moods can be compared run by run.

### Checks added after the second review

```
py run_experiment.py mood_from_start   # mood is on from trial 1, not from the swap
py run_experiment.py long_post         # 300 trials after the swap instead of 100
py run_experiment.py sym_slow          # no mood, both learning rates 0.04
py run_experiment.py sym_fast          # no mood, both learning rates 0.16
py run_experiment.py td_beta_matched   # TD with beta 1.6 (about as random as ANCCR)
py run_experiment.py td_wrong_probs    # TD is given reward chances that are 10 points wrong
py run_experiment.py td_plain_beta2    # plain TD (no swap hint) with beta 2
py run_experiment.py td_plain_beta3    # plain TD (no swap hint) with beta 3
py run_experiment.py factor2           # weaker mood dial (4 in the formula becomes 2)
py run_experiment.py factor8           # stronger mood dial (4 becomes 8)
py run_experiment.py anccr_T_half      # ANCCR memory window halved
py run_experiment.py anccr_T_double    # ANCCR memory window doubled
py run_experiment.py runs200           # 200 runs instead of 50 (slow)
```

Each one saves `code/results_<name>.json`. What they showed:

- Low mood does not make a typical TD run faster. It stops TD from getting stuck on the old
  button (at 40/90, 100% of runs switched at mood -5, 76% at neutral, 40% at +5).
- ANCCR almost always switches, but a typical run is much slower at low mood.
- The same pattern holds with mood on from trial 1, with wrong reward chances for TD, with a
  weaker or stronger mood dial and with a shorter or longer ANCCR memory window.
- Plain TD with beta 2 or 3 also gets stuck less at low mood, so this does not need the swap hint.
- With beta 1.6, TD almost never gets stuck and mood barely matters, so the TD result depends
  on TD choosing almost greedily.

`py make_fig6_values.py` draws what each model thinks the two buttons are worth over time.


## What's in the folders

| File | What it is |
|---|---|
| `code/td_model.py` | Belief-state TD model |
| `code/anccr_model.py` | ANCCR model |
| `code/run_experiment.py` | The task, the settings, and the full experiment |
| `code/make_plots.py` | Makes the four figures |
| `code/results.json` | Results: averages plus every run's values |
| `code/results_<name>.json` | Results of the fairness checks (same format) |
| `figures/fig1_switching.png` | Trials to switch after the swap, for each mood |
| `figures/fig2_best_mood.png` | The fastest mood for each model and reward pair |
| `figures/fig3_vs_neutral.png` | Change in trials to switch compared with neutral mood |
| `figures/fig4_dopamine.png` | Size of the dopamine signal on rewarded and missed trials, after the swap |
| `figures/fig5_switched_and_accuracy.png` | Share of runs that switched, and accuracy in the last 50 trials |
| `figures/fig6_button_values.png` | Button values over time at mood -5, 0 and +5 (30/80) |

Shaded bands and bars in the figures are 95% confidence intervals. We get them by re-picking the 50 runs at random 2,000 times.

## Credits and licence

- The model code is translated from the code for Qian, L., et al. (2025), *Prospective contingency explains behavior and dopamine signals during associative learning*, Nature Neuroscience. That code is on Dryad: doi:10.5061/dryad.mkkwh71f6.
- ANCCR is from Jeong, H., et al. (2022), *Mesolimbic dopamine release conveys causal associations*, Science, 378, eabq6740. The official code is at github.com/namboodirilab/ANCCR.
- Belief-state TD is based on `td_belief.m` by S. J. Gershman.

The MIT licence in `LICENSE` covers our own code and changes. The translated parts still belong to their original authors, so please cite them if you use this code.
