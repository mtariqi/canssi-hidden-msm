"""Simulation study (proposal: 'Evaluation and validation'; SAP section 6).

Design: 2 cohort sizes x 2 indicator discrimination levels x 2 observation
mechanisms x 2 heterogeneity levels = 16 scenarios. Every estimator is fitted
to identical replicates:
  proposed  : hidden MSM with informative observation process + random effect
  hidden_ni : hidden MSM that (incorrectly) assumes non-informative visits
  naive     : fully observed community/hospital/death Markov model
"""
from __future__ import annotations

import itertools, time, zlib
import numpy as np
import pandas as pd

from .model import ModelSpec, Prior
from .simulate import true_parameters, simulate_cohort, simulate_covariates, StepData
from .fit import fit_hidden, fit_naive, laplace_draws
from .predict import standardised_estimands, estimand_draws, naive_to_params
from .missing import impute_covariates

X_COLS, W_COLS = ["age_s", "comorb_s"], ["prior_util_s"]
KEY_PARAMS = ["alpha", "logq[adm_S]", "logq[adm_U]", "log_d[0]", "beta_dth[0]"]


def scenario_grid(sizes):
    rows = []
    for n, disc, inf, het in itertools.product(sizes, ["weak", "strong"], [False, True], [False, True]):
        rows.append({"scenario": f"n{n}_{disc}_{'inf' if inf else 'noninf'}_{'het' if het else 'nohet'}",
                     "n": n, "discrimination": disc, "informative": inf, "heterogeneity": het})
    return rows


def true_estimands(scn, seed=20260101, n_std=5000):
    p = true_parameters(scn)
    cv = simulate_covariates(n_std, np.random.default_rng(seed))
    return standardised_estimands(p, cv[X_COLS].to_numpy(), cv[W_COLS].to_numpy(), n_nodes=15)


def _summ(draws_dict, method, truth):
    out = []
    for k, v in draws_dict.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        out.append({"method": method, "quantity": k, "estimate": float(np.mean(v)),
                    "lower": float(lo), "upper": float(hi), "truth": truth[k]})
    return out


def run_replicate(scn, rep, master_seed, n_draws=80, n_std=200, stress=None):
    """Fit all estimators to one replicate; returns list of result rows."""
    stress = stress or {}
    ss = np.random.SeedSequence([master_seed, zlib.crc32(scn["scenario"].encode()), rep])
    rng = np.random.default_rng(ss)
    p = true_parameters(scn)
    cov, ev, _ = simulate_cohort(scn["n"], p, rng,
                                 dep_sd=stress.get("dep_sd", 0.0),
                                 missing_cov_frac=stress.get("missing_cov_frac", 0.0),
                                 elig_rate=stress.get("elig_rate", 0.03 / 365),
                                 admin_frac=stress.get("admin_frac", 0.0))
    if cov[X_COLS].isna().any().any():
        cov = impute_covariates(cov, X_COLS, ["prior_util_s"], rng, m=1)[0]
    data = StepData(cov, ev, X_COLS, W_COLS)
    truth = scn["truth"]
    std_idx = rng.choice(data.N, size=min(n_std, data.N), replace=False)
    Xs, Ws = data.X[std_idx], data.W[std_idx]
    rows = []
    specs = {"proposed": ModelSpec(2, 1, informative=True, heterogeneity=True),
             "hidden_ni": ModelSpec(2, 1, informative=False, heterogeneity=True)}
    full_true = specs["proposed"].pack(p)
    true_named = dict(zip(specs["proposed"].names, full_true))
    true_named["sigma"] = p["sigma"]
    for method, spec in specs.items():
        t0 = time.time()
        try:
            f = fit_hidden(data, spec)
            dr = laplace_draws(f, n_draws, rng)
            est = estimand_draws([spec.unpack(d) for d in dr], Xs, Ws)
            for r in _summ(est, method, truth):
                rows.append(r)
            for name in KEY_PARAMS + ["sigma"]:
                if name == "sigma" and spec.heterogeneity:
                    j = spec.index["log_sigma"].start
                    v = np.exp(dr[:, j])
                elif name in spec.names:
                    v = dr[:, spec.names.index(name)]
                else:
                    continue
                lo, hi = np.percentile(v, [2.5, 97.5])
                rows.append({"method": method, "quantity": name, "estimate": float(v.mean()),
                             "lower": float(lo), "upper": float(hi), "truth": float(true_named[name])})
            status = {"converged": f["converged"], "n_iter": f["n_iter"], "seconds": time.time() - t0}
        except Exception as e:  # fitting failures stay in the denominator
            status = {"converged": False, "n_iter": -1, "seconds": time.time() - t0,
                      "error": repr(e)[:200]}
        rows.append({"method": method, "quantity": "_status", **status})
    # naive comparator
    t0 = time.time()
    nv = fit_naive(data, 2)
    dr = rng.multivariate_normal(nv["phi"], nv["cov"], size=n_draws)
    est = estimand_draws([naive_to_params(d, 2, 1) for d in dr], Xs, Ws)
    rows += _summ(est, "naive", truth)
    rows.append({"method": "naive", "quantity": "_status", "converged": nv["converged"],
                 "n_iter": 0, "seconds": time.time() - t0})
    for r in rows:
        r.update({"scenario": scn["scenario"], "rep": rep})
    return rows


def summarise(df):
    """Bias, RMSE, coverage, interval width with Monte Carlo SEs (SAP section 6)."""
    st = df[df.quantity == "_status"]
    res = df[df.quantity != "_status"].drop(columns=["converged", "n_iter", "seconds", "error"], errors="ignore").copy()
    conv = st[["scenario", "rep", "method", "converged"]]
    res = res.merge(conv, on=["scenario", "rep", "method"], how="left")
    res["err"] = res.estimate - res.truth
    res["cover"] = ((res.lower <= res.truth) & (res.truth <= res.upper)).astype(float)
    res["width"] = res.upper - res.lower
    g = res[res.converged == True].groupby(["scenario", "method", "quantity"])
    out = g.agg(R=("err", "size"), truth=("truth", "mean"), bias=("err", "mean"),
                sd_err=("err", "std"), rmse=("err", lambda e: np.sqrt(np.mean(e ** 2))),
                coverage=("cover", "mean"), width=("width", "mean")).reset_index()
    out["mcse_bias"] = out.sd_err / np.sqrt(out.R)
    out["mcse_coverage"] = np.sqrt(out.coverage * (1 - out.coverage) / out.R)
    fails = st.groupby(["scenario", "method"]).agg(attempted=("converged", "size"),
                                                   converged=("converged", "sum"),
                                                   mean_seconds=("seconds", "mean")).reset_index()
    return out, fails
