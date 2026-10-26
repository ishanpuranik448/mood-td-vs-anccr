"""
Belief-state TD, ported to Python from td_belief.m (original author
S. J. Gershman) in the code for Qian, Burrell et al. (2025), "Prospective
contingency explains behavior and dopamine signals during associative
learning", Nature Neuroscience. Original MATLAB code: Dryad,
doi:10.5061/dryad.mkkwh71f6 (Code.zip, matlab_simulation_code/).

The original reads a whole event log in one go. This version takes one event
at a time, so an agent can choose between buttons while it learns. The update
equations follow the original line by line.

The one change to the maths is the mood dial: alpha_plus for positive RPEs and
alpha_minus for negative RPEs, instead of one alpha. Set them equal to get the
original model back.
"""
import numpy as np

class BeliefTD:
    """Belief-state TD for a partially observable task (td_belief.m).

    T[s, s']      probability of moving from hidden state s to s'
    O[s, s', e]   probability of seeing event e on the move s -> s'
    """

    def __init__(self, T, O, alpha_plus, alpha_minus, gamma):
        self.T, self.O = T, O
        self.ap, self.am, self.gamma = alpha_plus, alpha_minus, gamma
        S = T.shape[0]
        self.b = np.ones(S) / S          # belief state
        self.w = np.zeros(S)             # weights
        self.resets = 0

    def belief_after(self, e):
        """Belief after seeing event e (no learning). Second value: was it reset?"""
        b = self.b @ (self.T * self.O[:, :, e])
        total = b.sum()
        if not total > 0:                # original: NaN belief -> all mass on last state
            b = np.zeros_like(b); b[-1] = 1
            return b, True
        return b / total, False

    def value(self, b=None):
        return self.w @ (self.b if b is None else b)

    def step(self, e, r):
        """Observe event e with reward r. Returns the RPE ("dopamine")."""
        b0 = self.b                                    # old posterior
        self.b, reset = self.belief_after(e)
        self.resets += reset
        rpe = r + self.w @ (self.gamma * self.b - b0)
        alpha = self.ap if rpe > 0 else self.am        # mood dial (original: one alpha)
        self.w = self.w + alpha * rpe * b0             # weight update
        return rpe
