"""Standardised predictions and primary estimands (SAP section 7).

For each patient covariate profile and each quadrature value of a *new-patient*
random effect, the transition law is propagated from the index discharge
(community, instability drawn from the initial-state model) on a grid with
step h days, piecewise over the baseline time bands. Results are averaged over
the random effect (Gauss-Hermite) and standardised over the target cohort's
covariate distribution.

Estimands
  * death risk at 30/90/365 days              P(X(t) = death)
  * cumulative risk of ANY hospitalisation    via the augmented 'ever-admitted'
    generator (distinct from the probability of *currently* being in hospital)
  * restricted mean days alive outside acute hospital over 365 days
        RMDOH = integral_0^365 P(alive and in community) dt
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .model import rates, generator, gh_nodes, initial_unstable_prob, BAND_EDGES

HORIZONS = (30, 90, 365)


def _propagate(G_by_band, init, h, horizon=365.0):
    """Return grid times and state probabilities (T, n, Q, S). h should divide 30, 60 and 275."""
    E = [expm(G * h) for G in G_by_band]
    grid = np.arange(0.0, horizon + 1e-9, h)
    out = np.empty((grid.size,) + init.shape)
    P = init.copy(); out[0] = P
    for i, t in enumerate(grid[1:], 1):
        k = min(int(np.searchsorted(BAND_EDGES, t - h, side="right") - 1), 2)
        P = np.einsum("nqs,nqst->nqt", P, E[k])
        out[i] = P
    return grid, out


def individual_risks(p, X, W, n_nodes=9, h=5.0):
    """Per-patient predictions integrated over instability and new-patient b.

    Returns dict of arrays: death_t, anyhosp_t (t in HORIZONS) of shape (n,), rmdoh (n,).
    """
    xs, ws = gh_nodes(n_nodes) if p["sigma"] > 0 else (np.zeros(1), np.ones(1))
    b = p["sigma"] * xs
    n = len(X)
    pi = initial_unstable_prob(p, X)
    Gs, Ga = [], []
    for k in range(3):
        r = rates(p, X, W, b, np.full(n, k))
        Gs.append(generator(r)); Ga.append(generator(r, augmented=True))
    Q = b.size
    init5 = np.zeros((n, Q, 5)); init5[..., 0] = (1 - pi)[:, None]; init5[..., 1] = pi[:, None]
    init4 = np.zeros((n, Q, 4)); init4[..., 0] = (1 - pi)[:, None]; init4[..., 1] = pi[:, None]
    grid, P5 = _propagate(Gs, init5, h)
    _, P4 = _propagate(Ga, init4, h)
    wq = ws[None, :]
    out = {}
    for t in HORIZONS:
        i = min(int(np.searchsorted(grid, t - 1e-9)), grid.size - 1)  # first grid point >= t
        out[f"death_{t}"] = (P5[i, ..., 4] * wq).sum(-1)
        out[f"anyhosp_{t}"] = (P4[i, ..., 3] * wq).sum(-1)
    comm = (P5[..., 0] + P5[..., 1]) * wq[None]
    comm = comm.sum(-1)                                       # (T, n)
    out["rmdoh"] = np.trapezoid(comm, grid, axis=0)
    return out


def standardised_estimands(p, X, W, n_nodes=9, h=5.0):
    r = individual_risks(p, X, W, n_nodes, h)
    return {k: float(np.mean(v)) for k, v in r.items()}


def naive_to_params(phi, n_x, n_w):
    """Express the observed 3-state model in the 5-state parameterisation
    (no instability: zero onset rate, identical S/U rates) for prediction."""
    lq = phi[:4]
    logq = np.array([-30.0, -30.0, -30.0, -30.0, lq[0], lq[0], lq[1], lq[1],
                     lq[2], lq[2], lq[3], lq[3]])
    return {"logq": logq, "kappa": np.concatenate([[0.0], phi[4:6]]),
            "beta_adm": phi[6:6 + n_x], "beta_dth": phi[6 + n_x:6 + 2 * n_x],
            "sigma": 0.0, "eta_dth": 0.0, "log_lam0": -30.0, "gamma": np.zeros(n_w),
            "alpha": 0.0, "rho": 0.0, "a": np.zeros(3), "d": np.ones(3),
            "pi": np.concatenate([[-30.0], np.zeros(n_x)])}


def estimand_draws(draw_params, X, W, n_nodes=7, h=5.0):
    """Standardised estimands for a list of parameter dicts (posterior draws)."""
    rows = [standardised_estimands(p, X, W, n_nodes, h) for p in draw_params]
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows]) for k in keys}
