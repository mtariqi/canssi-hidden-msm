"""Joint likelihood by constrained continuous-time forward recursion (SAP section 4).

Between observed contacts / known care transitions the care setting is fixed,
so the forward vector over the latent instability Z in the current care
setting is propagated with the *sub-generator*

    exp{ [Q_cc - D_lambda] * dt },

where Q_cc is the 2x2 block of the generator for care setting c (exits to the
other care setting and to death are retained on the diagonal, i.e. "did not
happen") and D_lambda is the diagonal encounter intensity (community only).
This is what prevents the recursion from integrating over unrecorded observed
care transitions, as required in SAP section 4.

At events the forward vector is multiplied by
    encounter       diag(lambda_z * f(Y | z))
    admission       diag(q_adm,z)         (Z preserved at the transition)
    discharge       diag(q_dis,z)
    death           diag(q_death,c,z)
    censoring/split identity (sum over compatible living states at the end)

The patient random effect b is integrated by Gauss-Hermite quadrature.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .model import rates, gh_nodes, initial_unstable_prob, ModelSpec


def expm2(a, b, c, d, t):
    """Batched closed-form exp(t * [[a, b], [c, d]]) for real matrices with b*c >= 0.

    Returns the four entries (e00, e01, e10, e11). Eigenvalues are m +/- s with
    m = (a + d)/2 and s = sqrt(((a - d)/2)^2 + b c) >= 0.
    """
    m = 0.5 * (a + d)
    s = np.sqrt(0.25 * (a - d) ** 2 + b * c)
    ep = np.exp((m + s) * t)
    em = np.exp((m - s) * t)
    ch = 0.5 * (ep + em)                               # e^{mt} cosh(st)
    small = s * t < 1e-8
    s_safe = np.where(small, 1.0, s)
    sh = np.where(small, t * np.exp(m * t), 0.5 * (ep - em) / s_safe)  # e^{mt} sinh(st)/s
    e00 = ch + sh * (a - m)
    e11 = ch + sh * (d - m)
    return e00, sh * b, sh * c, e11


def _emission_table(p, data):
    """f(Y | z) for every (patient, step): array (N, S, 2).

    Uses precomputed indicator tensors (Y==1, Y==0) so missing indicators and
    non-encounter steps contribute a factor of exactly 1 (SAP section 8:
    missing emissions are treated as unobserved, not as absence).
    """
    out = np.empty(data.Y.shape[:2] + (2,))
    for z in (0, 1):
        eta = p["a"] + p["d"] * z
        lp1 = -np.logaddexp(0.0, -eta)          # log P(Y=1|z)
        lp0 = -np.logaddexp(0.0, eta)           # log P(Y=0|z)
        out[..., z] = data.I1 @ lp1 + data.I0 @ lp0
    return np.exp(out)


def loglik_patients(theta, spec: ModelSpec, data, n_nodes=7):
    """Per-patient marginal log-likelihood contributions, shape (N,)."""
    p = spec.unpack(theta)
    xs, ws = gh_nodes(n_nodes if spec.heterogeneity else 1)
    if not spec.heterogeneity:
        xs, ws = np.zeros(1), np.ones(1)
    bvals = p["sigma"] * xs
    N, Q = data.N, bvals.size
    X, W = data.X, data.W

    # precompute per-band quantities; index [band][care] -> (N, Q)
    off01 = np.empty((3, 2, N, Q)); off10 = np.empty((3, 2, N, Q))
    dg0 = np.empty((3, 2, N, Q)); dg1 = np.empty((3, 2, N, Q))
    ev0 = np.ones((3, 2, 6, N, Q)); ev1 = np.ones((3, 2, 6, N, Q))
    for k in range(3):
        r = rates(p, X, W, bvals, np.full(N, k))
        # community block
        off01[k, 0], off10[k, 0] = r["on_C"], r["rec_C"]
        dg0[k, 0] = -(r["on_C"] + r["adm_S"] + r["dth_CS"] + r["lam_S"])
        dg1[k, 0] = -(r["rec_C"] + r["adm_U"] + r["dth_CU"] + r["lam_U"])
        # hospital block (no outpatient/ED encounters while admitted)
        off01[k, 1], off10[k, 1] = r["on_H"], r["rec_H"]
        dg0[k, 1] = -(r["on_H"] + r["dis_S"] + r["dth_HS"])
        dg1[k, 1] = -(r["rec_H"] + r["dis_U"] + r["dth_HU"])
        # event multipliers (etype 1 enc, 2 adm, 3 dis, 4 death)
        ev0[k, 0, 1], ev1[k, 0, 1] = r["lam_S"], r["lam_U"]
        ev0[k, 0, 2], ev1[k, 0, 2] = r["adm_S"], r["adm_U"]
        ev0[k, 1, 3], ev1[k, 1, 3] = r["dis_S"], r["dis_U"]
        ev0[k, 0, 4], ev1[k, 0, 4] = r["dth_CS"], r["dth_CU"]
        ev0[k, 1, 4], ev1[k, 1, 4] = r["dth_HS"], r["dth_HU"]

    femit = _emission_table(p, data)
    pi = initial_unstable_prob(p, X)
    f0 = np.repeat((1 - pi)[:, None], Q, 1)
    f1 = np.repeat(pi[:, None], Q, 1)
    logc = np.zeros((N, Q))
    for s in range(data.S):
        n = data.n_active[s]
        idx = np.arange(n)
        k, c, e = data.band[:n, s], data.care[:n, s], data.etype[:n, s]
        dt = data.dt[:n, s][:, None]
        e00, e01, e10, e11 = expm2(dg0[k, c, idx], off01[k, c, idx],
                                   off10[k, c, idx], dg1[k, c, idx], dt)
        a0, a1 = f0[:n], f1[:n]
        n0 = a0 * e00 + a1 * e10
        n1 = a0 * e01 + a1 * e11
        n0 = n0 * ev0[k, c, e, idx] * femit[:n, s, 0][:, None]
        n1 = n1 * ev1[k, c, e, idx] * femit[:n, s, 1][:, None]
        tot = n0 + n1
        tot = np.maximum(tot, 1e-300)
        logc[:n] += np.log(tot)
        f0[:n], f1[:n] = n0 / tot, n1 / tot
    # censoring: forward vector already normalised; sum over living states = 1
    return logsumexp(logc + np.log(ws)[None, :], axis=1)


def loglik(theta, spec, data, n_nodes=7):
    return loglik_patients(theta, spec, data, n_nodes).sum()


# ----------------------------------------------------------------------
# Naive comparator: fully observed community/hospital/death Markov model
# ----------------------------------------------------------------------
NAIVE_TRANS = ["adm", "dis", "dth_C", "dth_H"]


def naive_exposure(data):
    """Sufficient statistics of the observed 3-state model from StepData."""
    out = {}
    for k in range(3):
        for c in (0, 1):
            m = (data.band == k) & (data.care == c)
            expo = (data.dt * m)  # exposure in care c during band k
            out[("expo", k, c)] = expo.sum(1)
    # events: by band of the interval in which they occur and care setting
    for name, et, c in [("adm", 2, 0), ("dis", 3, 1), ("dth_C", 4, 0), ("dth_H", 4, 1)]:
        for k in range(3):
            out[(name, k)] = ((data.etype == et) & (data.care == c) & (data.band == k)).sum(1)
    return out


def naive_loglik(phi, data, stats, n_x):
    """phi = [log adm, log dis, log dth_C, log dth_H, kappa2, kappa3, beta_adm, beta_dth]."""
    lq = phi[:4]; kap = np.concatenate([[0.0], phi[4:6]])
    ba = phi[6:6 + n_x]; bd = phi[6 + n_x:6 + 2 * n_x]
    xa = data.X @ ba; xd = data.X @ bd
    ll = np.zeros(data.N)
    for k in range(3):
        ra = np.exp(lq[0] + kap[k] + xa); rdis = np.exp(lq[1] + kap[k])
        rdc = np.exp(lq[2] + kap[k] + xd); rdh = np.exp(lq[3] + kap[k] + xd)
        ll += stats[("adm", k)] * np.log(ra) + stats[("dis", k)] * np.log(rdis)
        ll += stats[("dth_C", k)] * np.log(rdc) + stats[("dth_H", k)] * np.log(rdh)
        ll -= stats[("expo", k, 0)] * (ra + rdc) + stats[("expo", k, 1)] * (rdis + rdh)
    return ll
