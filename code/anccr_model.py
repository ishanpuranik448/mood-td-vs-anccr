"""
ANCCR (Adjusted Net Contingency for Causal Relations), ported to Python from
useful_functions/calculateANCCRperformance.m in the code for Qian, Burrell et
al. (2025), "Prospective contingency explains behavior and dopamine signals
during associative learning", Nature Neuroscience. That file is the official
ANCCR code (github.com/namboodirilab/ANCCR) with speed changes. Original
MATLAB code: Dryad, doi:10.5061/dryad.mkkwh71f6 (Code.zip).

The original reads a whole event log in one go. This version takes one event
at a time, so an agent can choose between buttons while it learns. The update
equations follow the original line by line.

The one change to the maths is the mood dial: separate learning rates for the
contingency ("link") update, alpha_plus when a reward arrives and alpha_minus
when a reward is missed (an omission event). Every other event uses alpha.
Set alpha_plus = alpha_minus = alpha to get the original model back.

(alpha_r_plus / alpha_r_minus split the reward-value ("worth") update. That
had no effect here because every reward is worth 1, so the experiment now
keeps them equal.)
"""
import numpy as np

class ANCCR:
    """Adjusted Net Contingency for Causal Relations (calculateANCCRperformance.m).

    Events are integers 0..nstimuli-1 and arrive with a time (seconds) and a
    magnitude (1 for reward, 0 otherwise). omidx rows are [omission event,
    the reward event it is the omission of].
    """

    def __init__(self, nstimuli, T, alpha, k, samplinginterval, w, threshold,
                 minimumrate, beta, alpha_r_plus, alpha_r_minus, omidx=None,
                 alpha_plus=None, alpha_minus=None):
        n = nstimuli
        self.n, self.T = n, T
        self.gamma = np.exp(-1 / T)
        self.alpha = alpha
        # mood dial on the contingency update (None = same as alpha)
        self.ap = alpha if alpha_plus is None else alpha_plus
        self.am = alpha if alpha_minus is None else alpha_minus
        self.k = 1 / T if np.isnan(k) else k
        self.dt = samplinginterval
        self.w, self.threshold, self.minimumrate = w, threshold, minimumrate
        self.beta = np.asarray(beta, float)
        self.arp, self.arm = min(alpha_r_plus, 1), min(alpha_r_minus, 1)
        self.omidx = np.zeros((0, 2), int) if omidx is None else np.atleast_2d(omidx)
        # whether each omission event is used yet (only once its reward has a cause)
        self.omtrue = np.zeros(len(self.omidx), bool)

        self.Eij, self.Ei, self.Mi, self.Delta = (np.zeros(n) for _ in range(4))
        self.Mij, self.PRC, self.SRC, self.NC, self.A = (np.zeros((n, n)) for _ in range(5))
        self.R = np.zeros((n, n))
        self.numevents = np.zeros(n)
        self.Imct = self.beta > threshold          # meaningful causal targets
        self.pending = []                          # events not yet added to Ei
        self.t_last = None
        self.numsampling = 0

    def _unused_omission(self, e):
        rows = self.omidx[:, 0] == e
        return rows.any() and not self.omtrue[rows].any()

    def _baseline(self, t_next):
        """Sample eligibility trace (Mi) between the last event and t_next."""
        first = np.ceil(self.t_last / self.dt - 1e-9)
        times = np.arange(first, np.ceil(t_next / self.dt - 1e-9)) * self.dt
        self.Ei = self.Ei * self.gamma ** self.dt
        if len(times):
            for e, te in self.pending:
                if self._unused_omission(e):
                    continue
                self.Ei[e] += self.gamma ** (times[0] - te)
            self.pending = []
        self.Mi = self.Mi + self.k * self.alpha * (self.Ei - self.Mi)
        for _ in times[1:]:
            self.Ei = self.Ei * self.gamma ** self.dt
            self.Mi = self.Mi + self.k * self.alpha * (self.Ei - self.Mi)
        self.numsampling += len(times)

    def step(self, e, t, mag):
        """Observe event e at time t with magnitude mag. Returns dopamine (DA)."""
        if self.t_last is not None:
            self._baseline(t)

        if self._unused_omission(e):
            # the original carries everything over except ANCCR, which stays zero
            self.A = np.zeros_like(self.A)
            DA = 0.0
        else:
            DA = self._event(e, t, mag)

        self.pending.append((e, t))
        self.t_last = t
        return DA

    def _event(self, e, t, mag):
        n = self.n
        self.numevents[e] += 1
        # mood dial: alpha_plus at a reward, alpha_minus at a missed reward
        if (self.omidx[:, 1] == e).any():
            a = self.ap
        elif (self.omidx[:, 0] == e).any():
            a = self.am
        else:
            a = self.alpha
        A = np.zeros((n, n))
        if self.t_last is not None:
            decay = self.gamma ** (t - self.t_last)
            self.Delta = self.Delta * decay        # recent-event indicator
            self.Eij = self.Eij * decay            # instantaneous eligibility trace
            A = self.A.copy(); A[e, :] = 0         # ANCCR carried over except row e
        self.Delta[e] = 1
        self.Eij[e] += 1
        # average eligibility trace
        self.Mij[:, e] += a * (self.Eij - self.Mij[:, e]) * self.Imct[e]

        # successor (PRC) and predecessor (SRC) representation contingencies
        PRC = self.Mij - self.Mi[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            SRC = PRC * self.Mi[None, :] / self.Mi[:, None]
        SRC[self.Mi / self.T < self.minimumrate, :] = 0

        # only use contingencies after the first instance of each event
        unseen = self.numevents == 0
        PRC[unseen, :] = 0; PRC[:, unseen] = 0
        SRC[unseen, :] = 0; SRC[:, unseen] = 0
        self.R[:, unseen] = 0; self.R[unseen, :] = 0

        NC = self.w * SRC + (1 - self.w) * PRC     # net contingency

        Iedge = NC[:, e] > self.threshold
        Iedge[e] = False
        # once the cause of a reward is known, its omission event is used
        rew_rows = self.omidx[:, 1] == e
        if rew_rows.any() and Iedge.any():
            self.omtrue[rew_rows] = True

        self.R[e, e] = mag
        for ke in range(n):
            Ik = NC[:, ke] > self.threshold
            Ik[ke] = False
            A[ke, :] = NC[ke, :] * self.R[ke, :] - (A * self.Delta[:, None] * Ik[:, None]).sum(0)

        DA = float((A[e, :] * self.Imct).sum())

        om_rows = self.omidx[:, 0] == e
        if om_rows.any():
            rew = self.omidx[om_rows, 1][0]
            self.R[e, rew] = self.R[rew, rew]      # omission of j is valued like j
            self.Imct[e] = True
        self.Imct[e] = self.Imct[e] or DA + self.beta[e] > self.threshold

        # reward value update (positive-DA and negative-DA rules)
        if DA >= 0:
            self.R[:, e] += self.arp * (mag - self.R[:, e])
        elif Iedge.any():
            share = self.Delta[Iedge] / self.numevents[Iedge]
            self.R[Iedge, e] -= self.arm * self.R[Iedge, e] * share / share.sum()

        self.PRC, self.SRC, self.NC, self.A = PRC, SRC, NC, A
        return DA
