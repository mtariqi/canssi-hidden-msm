"""Data-generating process for simulation studies (SAP section 6).

Generates, in order: covariates, random effects, initial instability, latent
paths jointly with care transitions and state-dependent encounter times,
emissions at encounters, then censoring (administrative + loss of eligibility).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import TRANS, BAND_EDGES, HORIZON, DEFAULT_RATE_CENTRES, ModelSpec

ETYPES = {"split": 0, "enc": 1, "adm": 2, "dis": 3, "dth": 4, "cens": 5}


def true_parameters(scenario: dict, n_x=2, n_w=1, n_ind=3):
    """Numerical generating values for a simulation scenario.

    scenario keys: discrimination ('weak'/'strong'), informative (bool),
    heterogeneity (bool). Values are transparent synthetic calibrations
    (SAP section 6) and must be replaced by pilot-count calibrations once the
    ICES feasibility counts are available.
    """
    logq = np.array([np.log(DEFAULT_RATE_CENTRES[t]) for t in TRANS])
    strong = scenario.get("discrimination", "strong") == "strong"
    p = {
        "logq": logq,
        "kappa": np.array([0.0, -0.25, -0.5]),     # risk highest early after discharge
        "beta_adm": np.array([0.20, 0.35])[:n_x],
        "beta_dth": np.array([0.45, 0.25])[:n_x],
        "sigma": 0.6 if scenario.get("heterogeneity", True) else 0.0,
        "eta_dth": 0.7,
        "log_lam0": np.log(5.0 / 365),
        "gamma": np.array([0.4])[:n_w],
        "alpha": 1.0 if scenario.get("informative", True) else 0.0,
        "rho": 0.5 if (scenario.get("informative", True)
                       and scenario.get("heterogeneity", True)) else 0.0,
        "a": np.array([-2.2, -2.0, -2.5])[:n_ind],
        "d": (np.array([2.6, 2.4, 2.8]) if strong else np.array([0.8, 0.7, 0.9]))[:n_ind],
        "pi": np.array([-1.0, 0.4, 0.3])[: 1 + n_x],
    }
    return p


def simulate_covariates(n, rng, missing_frac=0.0):
    """Standardised age, comorbidity burden; encounter covariate = prior use."""
    cov = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.4], [0.1, 0.4, 1.0]])
    Z = rng.multivariate_normal(np.zeros(3), cov, size=n)
    df = pd.DataFrame({"id": np.arange(n), "age_s": Z[:, 0],
                       "comorb_s": Z[:, 1], "prior_util_s": Z[:, 2]})
    if missing_frac > 0:
        # missing at random given prior utilisation
        pmiss = missing_frac * 2 / (1 + np.exp(-Z[:, 2]))
        df.loc[rng.random(n) < np.clip(pmiss, 0, 0.9), "comorb_s"] = np.nan
    return df


def simulate_histories(Xtrue, W, p, rng, cens_time, dep_sd=0.0, miss_y=0.05):
    """Simulate post-index histories for given covariates and censoring times.

    Returns (events df, latent truth df). Events: enc / adm / dis / dth / cens.
    """
    n = len(Xtrue)
    M = len(p["a"])
    xa = Xtrue @ p["beta_adm"]; xd = Xtrue @ p["beta_dth"]; wg = W @ p["gamma"]
    b = p["sigma"] * rng.standard_normal(n)
    pi = 1 / (1 + np.exp(-(p["pi"][0] + Xtrue @ p["pi"][1:])))
    z0 = (rng.random(n) < pi).astype(int)
    q0 = dict(zip(TRANS, np.exp(p["logq"])))
    ek = np.exp(p["kappa"])
    rows, truth = [], []
    for i in range(n):
        t, c, z = 0.0, 0, z0[i]
        fa, fd = np.exp(xa[i] + b[i]), np.exp(xd[i] + p["eta_dth"] * b[i])
        lam = np.exp(p["log_lam0"] + wg[i] + p["rho"] * b[i]) * np.array([1.0, np.exp(p["alpha"])])
        truth.append((i, 0.0, c, z))
        while True:
            k = min(int(np.searchsorted(BAND_EDGES, t, side="right") - 1), 2)
            if c == 0:
                cand = [("flip", q0["rec_C" if z else "on_C"] * ek[k]),
                        ("adm", q0["adm_U" if z else "adm_S"] * ek[k] * fa),
                        ("dth", q0["dth_CU" if z else "dth_CS"] * ek[k] * fd),
                        ("enc", lam[z])]
            else:
                cand = [("flip", q0["rec_H" if z else "on_H"] * ek[k]),
                        ("dis", q0["dis_U" if z else "dis_S"] * ek[k]),
                        ("dth", q0["dth_HU" if z else "dth_HS"] * ek[k] * fd)]
            tot = sum(r for _, r in cand)
            tn = t + rng.exponential(1 / tot)
            edge = BAND_EDGES[k + 1]
            if tn >= cens_time[i]:
                rows.append((i, cens_time[i], "cens", [-1] * M)); break
            if tn >= edge:          # memoryless: restart at the band boundary
                t = edge; continue
            t = tn
            u = rng.random() * tot
            for ev, r in cand:
                u -= r
                if u <= 0:
                    break
            if ev == "flip":
                z = 1 - z; truth.append((i, t, c, z))
            elif ev == "enc":
                eps = dep_sd * rng.standard_normal()       # shared residual => dependence
                pr = 1 / (1 + np.exp(-(p["a"] + p["d"] * z + eps)))
                y = (rng.random(M) < pr).astype(int)
                if rng.random() < miss_y:                  # diagnosis field missing
                    y[:] = -1
                rows.append((i, t, "enc", list(y)))
            elif ev == "adm":
                c = 1; rows.append((i, t, "adm", [-1] * M)); truth.append((i, t, c, z))
            elif ev == "dis":
                c = 0; rows.append((i, t, "dis", [-1] * M)); truth.append((i, t, c, z))
            else:
                rows.append((i, t, "dth", [-1] * M)); break
    ev = pd.DataFrame([(i, t, e) for i, t, e, _ in rows], columns=["id", "time", "event"])
    Y = np.array([y for *_, y in rows], dtype=int).reshape(len(rows), M)
    for m in range(M):
        ev[f"y{m+1}"] = Y[:, m]
    tr = pd.DataFrame(truth, columns=["id", "time", "care", "z"])
    return ev, tr


def simulate_cohort(n, p, rng, elig_rate=0.03 / 365, admin_frac=0.0,
                    dep_sd=0.0, miss_y=0.05, missing_cov_frac=0.0,
                    x_cols=("age_s", "comorb_s"), w_cols=("prior_util_s",)):
    """Simulate one synthetic cohort. Returns (covariates df, events df, truth df)."""
    cov = simulate_covariates(n, rng, 0.0)          # complete covariates for the DGP
    Xtrue = cov[list(x_cols)].to_numpy()
    W = cov[list(w_cols)].to_numpy()
    elig = rng.exponential(1 / elig_rate, n) if elig_rate > 0 else np.full(n, np.inf)
    admin = np.where(rng.random(n) < admin_frac, rng.uniform(60, 365, n), HORIZON)
    cens_time = np.minimum(np.minimum(elig, admin), HORIZON)
    ev, tr = simulate_histories(Xtrue, W, p, rng, cens_time, dep_sd, miss_y)
    ev["id"] = cov["id"].to_numpy()[ev["id"].to_numpy()]
    if missing_cov_frac > 0:   # MAR given prior utilisation, imposed after the DGP
        pm = np.clip(missing_cov_frac * 2 / (1 + np.exp(-cov["prior_util_s"])), 0, 0.9)
        cov.loc[rng.random(n) < pm, "comorb_s"] = np.nan
    return cov, ev, tr


# ----------------------------------------------------------------------
# event history -> forward-recursion steps
# ----------------------------------------------------------------------
class StepData:
    """Patient histories packed as padded arrays for vectorised recursion.

    Patients are sorted by number of steps (descending) so that the active set
    at step s is always a leading slice [:n_active[s]].
    """

    def __init__(self, cov, events, x_cols, w_cols, n_ind=3):
        ev = events.sort_values(["id", "time"], kind="stable")
        ids = cov["id"].to_numpy()
        ycols = [f"y{m+1}" for m in range(n_ind)]
        per = {i: g for i, g in ev.groupby("id", sort=False)}
        seqs = []
        for i in ids:
            g = per[i]
            times = g["time"].to_numpy(); kinds = g["event"].to_numpy()
            Y = g[ycols].to_numpy()
            t_prev, c, steps = 0.0, 0, []
            for t, kd, y in zip(times, kinds, Y):
                for edge in BAND_EDGES[1:-1]:          # split at band edges
                    if t_prev < edge < t:
                        steps.append((edge - t_prev, c, _band(t_prev), 0, [-1] * n_ind))
                        t_prev = edge
                steps.append((t - t_prev, c, _band(t_prev), ETYPES[kd], list(y)))
                if kd == "adm": c = 1
                elif kd == "dis": c = 0
                t_prev = t
            seqs.append(steps)
        order = np.argsort([-len(s) for s in seqs], kind="stable")
        self.ids = ids[order]
        self.nsteps = np.array([len(seqs[o]) for o in order])
        N, S = len(ids), int(self.nsteps.max())
        self.dt = np.zeros((N, S)); self.care = np.zeros((N, S), int)
        self.band = np.zeros((N, S), int); self.etype = np.zeros((N, S), int)
        self.Y = -np.ones((N, S, n_ind), int)
        for r, o in enumerate(order):
            for s, (dt, c, k, e, y) in enumerate(seqs[o]):
                self.dt[r, s], self.care[r, s], self.band[r, s], self.etype[r, s] = dt, c, k, e
                self.Y[r, s] = y
        self.n_active = np.array([(self.nsteps > s).sum() for s in range(S)])
        cv = cov.set_index("id").loc[self.ids]
        self.X = cv[list(x_cols)].to_numpy(float)
        self.W = cv[list(w_cols)].to_numpy(float)
        self.N, self.S, self.n_ind = N, S, n_ind
        self._indicators()

    def _indicators(self):
        self.I1 = (self.Y == 1).astype(float)
        self.I0 = (self.Y == 0).astype(float)

    def subset(self, mask):
        """Return a new StepData restricted to rows in boolean mask (keeps order)."""
        new = object.__new__(StepData)
        for k in ["ids", "nsteps", "dt", "care", "band", "etype", "Y", "X", "W"]:
            setattr(new, k, getattr(self, k)[mask])
        new.N = int(mask.sum()); new.S = int(new.nsteps.max()); new.n_ind = self.n_ind
        for k in ["dt", "care", "band", "etype", "Y"]:
            setattr(new, k, getattr(new, k)[:, :new.S])
        new.n_active = np.array([(new.nsteps > s).sum() for s in range(new.S)])
        new._indicators()
        return new


def _band(t):
    return min(int(np.searchsorted(BAND_EDGES, t, side="right") - 1), 2)
