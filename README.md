# Mood, belief-state TD and ANCCR

Do two models of dopamine learning react differently to good and bad outcomes as mood changes?

`code/models.py` has Python versions of the two models from Qian, Burrell et al. (2025), ported from their MATLAB code (Dryad, doi:10.5061/dryad.mkkwh71f6):

- **Belief-state TD** (`td_belief.m`, original author S. J. Gershman)
- **ANCCR** (`calculateANCCRperformance.m`, from github.com/namboodirilab/ANCCR)

The only change to the maths is a mood dial: separate learning rates for good and bad outcomes. Set them equal and you get the original models.

`code/run_experiment.py` runs both models on a two-button reversal task: 100 trials, a swap, then 100 more, for six probability pairs and 11 mood levels (-5 to +5).

## Run it

```
py -m pip install -r requirements.txt
cd code
py run_experiment.py
```
