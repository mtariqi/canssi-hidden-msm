"""Estimation.

`fit_hidden` finds the posterior mode of the hidden multi-state model and
returns a Laplace (Gaussian) approximation to the posterior. This is the
portable, CPU-light pilot estimator that runs anywhere (including GitHub CI).
The production estimator specified in the proposal is Hamiltonian Monte Carlo
(see stan/hidden_msm.stan and R/fit_stan.R); the Laplace fit is used for
(i) the simulation study at scale, (ii) HMC initialisation, and (iii) as a
check that HMC and the mode agree.

Optimiser: Berndt-Hall-Hall-Hausman (outer product of per-patient scores) with
Levenberg damping and Armijo backtracking. Per-patient scores are obtained by
finite differences of the per-patient log-likelihood vector, so one gradient
costs (P + 1) likelihood evaluations and simultaneously yields the BHHH
information matrix.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize

from .model import ModelSpec, Prior, TRANS
from .likelihood import loglik_patients, naive_loglik, naive_exposure


def _prior_grad(theta, spec, prior):
    mu, sd = prior.means_sds(spec)
    g = -(theta - mu) / sd ** 2
    if spec.heterogeneity:
        j = spec.index["log_sigma"].start
        g[j] = -np.exp(2 * theta[j]) + 1.0
    return g


def per_patient_scores(theta, spec, data, n_nodes, h=1e-5, central=False):
    base = loglik_patients(theta, spec, data, n_nodes)
    G = np.empty((data.N, spec.n_par))
    for j in range(spec.n_par):
        tp = theta.copy(); tp[j] += h
        if central:
            tm = theta.copy(); tm[j] -= h
            G[:, j] = (loglik_patients(tp, spec, data, n_nodes)
                       - loglik_patients(tm, spec, data, n_nodes)) / (2 * h)
        else:
            G[:, j] = (loglik_patients(tp, spec, data, n_nodes) - base) / h
    return base, G


def starting_values(spec: ModelSpec, data, prior: Prior):
    """Data-informed starting values from crude rates and the naive model."""
    nv = fit_naive(data, spec.n_x, laplace=False)
    phi = nv["phi"]
    mu, _ = prior.means_sds(spec)
    th = mu.copy()
    lq = dict(zip(TRANS, th[spec.index["logq"]]))
    lq.update({"adm_S": phi[0] - 0.6, "adm_U": phi[0] + 1.0,
               "dis_S": phi[1] + 0.2, "dis_U": phi[1] - 0.2,
               "dth_CS": phi[2] - 0.8, "dth_CU": phi[2] + 1.0,
               "dth_HS": phi[3] - 0.4, "dth_HU": phi[3] + 0.6})
    th[spec.index["logq"]] = [lq[t] for t in TRANS]
    th[spec.index["kappa"]] = phi[4:6]
    th[spec.index["beta_adm"]] = phi[6:6 + spec.n_x]
    th[spec.index["beta_dth"]] = phi[6 + spec.n_x:6 + 2 * spec.n_x]
    expo_c = sum((data.dt * (data.care == 0)).sum() for _ in [0])
    n_enc = (data.etype == 1).sum()
    th[spec.index["log_lam0"]] = np.log(max(n_enc, 1) / expo_c) - 0.3
    enc = data.etype == 1
    ybar = np.array([(data.I1[..., m][enc]).sum() / max(1, (data.I1[..., m] + data.I0[..., m])[enc].sum())
                     for m in range(spec.n_ind)])
    ybar = np.clip(ybar, 0.02, 0.98)
    th[spec.index["a"]] = np.log(ybar / (1 - ybar)) - 0.7
    th[spec.index["log_d"]] = np.log(1.5)
    if spec.heterogeneity:
        th[spec.index["log_sigma"]] = np.log(0.4)
        th[spec.index["eta_dth"]] = 0.5
    if spec.informative:
        th[spec.index["alpha"]] = 0.5
    return th


def fit_hidden(data, spec: ModelSpec, prior: Prior | None = None, theta0=None,
               n_nodes=7, max_iter=60, tol=1e-3, verbose=False):
    """Posterior mode + Laplace approximation for the hidden multi-state model."""
    prior = prior or Prior()
    th = starting_values(spec, data, prior) if theta0 is None else np.array(theta0, float)
    P0 = prior.precision(th, spec)
    t0 = time.time()

    def objective(t):
        ll = loglik_patients(t, spec, data, n_nodes).sum()
        return -(ll + prior.logpdf(t, spec))

    f = objective(th)
    lam, converged, it = 1e-3, False, 0
    for it in range(1, max_iter + 1):
        base, G = per_patient_scores(th, spec, data, n_nodes)
        g = G.sum(0) + _prior_grad(th, spec, prior)
        H = G.T @ G + prior.precision(th, spec)
        dec = None
        for _ in range(12):
            try:
                step = np.linalg.solve(H + lam * np.diag(np.diag(H)), g)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            dec = float(g @ step)
            alpha_ls, accepted = 1.0, False
            for _ in range(8):
                tn = th + alpha_ls * step
                fn = objective(tn)
                if np.isfinite(fn) and fn <= f - 1e-4 * alpha_ls * dec:
                    accepted = True; break
                alpha_ls *= 0.5
            if accepted:
                th, f = tn, fn
                lam = max(lam / 3, 1e-6)
                break
            lam *= 10
        if verbose:
            print(f"iter {it:3d}  -logpost {f:12.3f}  decrement {dec:9.4g}  damping {lam:.1e}")
        if dec is not None and dec < tol:
            converged = True
            break
    # Laplace approximation at the mode (central-difference scores, BHHH information)
    base, G = per_patient_scores(th, spec, data, n_nodes, central=True)
    H = G.T @ G + prior.precision(th, spec)
    try:
        cov = np.linalg.inv(H)
        cov = 0.5 * (cov + cov.T)
        ok = np.all(np.linalg.eigvalsh(cov) > 0)
    except np.linalg.LinAlgError:
        cov, ok = np.full((spec.n_par, spec.n_par), np.nan), False
    return {"theta": th, "cov": cov, "converged": bool(converged and ok),
            "n_iter": it, "neg_logpost": float(f), "loglik": float(base.sum()),
            "seconds": time.time() - t0, "spec": spec}


def laplace_draws(fit, n_draws, rng):
    return rng.multivariate_normal(fit["theta"], fit["cov"], size=n_draws)


# ----------------------------------------------------------------------
# naive observed-state comparator
# ----------------------------------------------------------------------
def fit_naive(data, n_x, laplace=True):
    stats = naive_exposure(data)
    nev = lambda k: sum(stats[(k, b)].sum() for b in range(3))
    e0 = sum(stats[("expo", b, 0)].sum() for b in range(3))
    e1 = sum(stats[("expo", b, 1)].sum() for b in range(3))
    phi0 = np.concatenate([np.log(np.maximum([nev("adm") / e0, nev("dis") / e1,
                                              nev("dth_C") / e0, nev("dth_H") / e1], 1e-6)),
                           np.zeros(2 + 2 * n_x)])
    obj = lambda ph: -naive_loglik(ph, data, stats, n_x).sum() + 1e-4 * (ph[4:] ** 2).sum()
    res = minimize(obj, phi0, method="BFGS")
    ok = bool(res.success) or float(np.max(np.abs(res.jac))) < 1e-2
    out = {"phi": res.x, "converged": ok, "loglik": -res.fun}
    if laplace:
        H = _num_hessian(obj, res.x)
        out["cov"] = np.linalg.inv(H)
    return out


def _num_hessian(f, x, h=1e-4):
    n = x.size; H = np.empty((n, n)); f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ej = np.zeros(n); ei[i] = h; ej[j] = h
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei) - f(x + ej) + f0) / h ** 2
    return 0.5 * (H + H.T)
