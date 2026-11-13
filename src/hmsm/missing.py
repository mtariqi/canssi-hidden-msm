"""Missing baseline covariates (proposal 'Missing data and censoring'; SAP section 8).

Proper multiple imputation by Bayesian normal linear regression: for each
imputation, regression coefficients and residual variance are drawn from
their posterior before imputing, so between-imputation variability is
propagated. Imputation must be run *within* each training split; it never
uses outcomes from a validation split.

Posterior draws from the Laplace (or HMC) fits on each imputed dataset are
pooled by mixing equal numbers of draws across imputations ("mixing draws"),
which is the Bayesian analogue of Rubin's rules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def impute_covariates(cov: pd.DataFrame, cols, aux_cols, rng, m=5):
    out = []
    for _ in range(m):
        df = cov.copy()
        for c in cols:
            miss = df[c].isna().to_numpy()
            if not miss.any():
                continue
            preds = [a for a in list(aux_cols) + list(cols) if a != c]
            Z = df[preds].copy()
            Z = Z.fillna(Z.mean())
            A = np.column_stack([np.ones(len(df)), Z.to_numpy()])
            y = df[c].to_numpy()
            Ao, yo = A[~miss], y[~miss]
            XtX_inv = np.linalg.inv(Ao.T @ Ao)
            bhat = XtX_inv @ Ao.T @ yo
            resid = yo - Ao @ bhat
            dof = max(len(yo) - A.shape[1], 1)
            s2 = resid @ resid / rng.chisquare(dof)          # draw sigma^2
            bdraw = rng.multivariate_normal(bhat, s2 * XtX_inv)
            df.loc[miss, c] = A[miss] @ bdraw + np.sqrt(s2) * rng.standard_normal(miss.sum())
        out.append(df)
    return out


def fraction_missing_information(estimates, variances):
    """Rubin's FMI for a scalar, used to choose the number of imputations."""
    est, var = np.asarray(estimates), np.asarray(variances)
    m = len(est)
    W, B = var.mean(), est.var(ddof=1)
    T = W + (1 + 1 / m) * B
    return float((1 + 1 / m) * B / T)


def mix_draws(draw_list, rng):
    """Pool posterior draws across imputations with equal weight."""
    n = min(len(d) for d in draw_list)
    return np.concatenate([d[rng.choice(len(d), n, replace=False)] for d in draw_list])
