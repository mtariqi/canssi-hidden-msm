"""Temporal validation (proposal 'Evaluation and validation'; SAP section 7).

Outcomes at horizon t for patient i:
  death_t   : I(T_death <= t)
  anyhosp_t : I(T_first_admission <= t); death before admission is a competing
              event whose outcome status (0) is known once it occurs.
Censoring is handled by inverse probability of censoring weights (IPCW) with
the Kaplan-Meier estimate G of the censoring survival function:
  status known at t with event/competing event at T_i <= t -> weight 1 / G(T_i-)
  still under observation at t                              -> weight 1 / G(t)
  censored before t without an event                        -> weight 0
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def outcome_times(ev):
    """Per-patient first admission, death and censoring/last-follow-up times."""
    g = ev.groupby("id")
    out = pd.DataFrame({"id": list(g.groups.keys())}).set_index("id")
    out["t_adm"] = ev[ev.event == "adm"].groupby("id").time.min()
    out["t_dth"] = ev[ev.event == "dth"].groupby("id").time.min()
    out["t_cens"] = ev[ev.event == "cens"].groupby("id").time.min()
    out["t_end"] = out[["t_dth", "t_cens"]].min(axis=1)
    return out


def km_censoring(t_end, censored):
    """Kaplan-Meier survival of the censoring time; returns callable G(t-)."""
    times = np.sort(np.unique(t_end[censored]))
    G, surv = [], 1.0
    for u in times:
        at_risk = (t_end >= u).sum()
        d = ((t_end == u) & censored).sum()
        surv *= 1 - d / at_risk
        G.append(surv)
    G = np.array(G)

    def Gminus(t):
        i = np.searchsorted(times, t, side="left") - 1
        return np.where(i >= 0, G[np.clip(i, 0, None)] if G.size else 1.0, 1.0)
    return Gminus


def ipcw_outcome(ot, outcome, t):
    """Binary outcome at t and IPC weights."""
    censored = ot.t_cens.notna() & ot.t_dth.isna()
    G = km_censoring(ot.t_end.to_numpy(), censored.to_numpy())
    if outcome == "death":
        ev_time = ot.t_dth.to_numpy()
        y = (ev_time <= t).astype(float)
        known_time = np.where(ev_time <= t, ev_time, np.nan)
    else:
        ta, td = ot.t_adm.to_numpy(), ot.t_dth.to_numpy()
        hosp = ta <= t
        comp = (~hosp) & (td <= t)
        y = hosp.astype(float)
        known_time = np.where(hosp, ta, np.where(comp, td, np.nan))
    under_obs = ot.t_end.to_numpy() >= t
    w = np.zeros(len(ot))
    has_event = ~np.isnan(known_time)
    w[has_event] = 1.0 / np.maximum(G(known_time[has_event]), 1e-3)
    still = (~has_event) & under_obs
    w[still] = 1.0 / np.maximum(G(np.full(still.sum(), t)), 1e-3)
    return y, w


def brier(pred, y, w):
    """Graf et al. IPCW Brier score: (1/n) sum_i w_i (y_i - p_i)^2."""
    return float(np.sum(w * (y - pred) ** 2) / len(y))


def calibration_table(pred, y, w, n_groups=10):
    q = pd.qcut(pred, n_groups, labels=False, duplicates="drop")
    df = pd.DataFrame({"g": q, "pred": pred, "y": y, "w": w})
    out = df.groupby("g").apply(lambda d: pd.Series({
        "pred": d.pred.mean(), "obs": np.sum(d.w * d.y) / max(np.sum(d.w), 1e-9),
        "n_eff": (d.w > 0).sum()}), include_groups=False).reset_index()
    return out


def calibration_intercept_slope(pred, y, w, iters=50):
    """Weighted logistic recalibration: logit P(y) = a + b * logit(pred)."""
    m = w > 0
    lp = np.log(np.clip(pred[m], 1e-6, 1 - 1e-6) / (1 - np.clip(pred[m], 1e-6, 1 - 1e-6)))
    A = np.column_stack([np.ones(m.sum()), lp]); yy, ww = y[m], w[m]
    beta = np.array([0.0, 1.0])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ beta))
        g = A.T @ (ww * (yy - p))
        H = (A * (ww * p * (1 - p))[:, None]).T @ A
        step = np.linalg.solve(H + 1e-9 * np.eye(2), g)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    # calibration-in-the-large: intercept with slope fixed at 1 (offset model)
    a = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a + lp)))
        a += np.sum(ww * (yy - p)) / np.sum(ww * p * (1 - p))
    return {"cal_intercept": float(a), "cal_slope": float(beta[1])}


def bootstrap_brier_diff(pred_a, pred_b, ot, outcome, t, B, rng):
    """Bootstrap distribution of Brier(a) - Brier(b) with IPCW recomputed each time."""
    n = len(ot); diffs = np.empty(B)
    for bi in range(B):
        idx = rng.integers(0, n, n)
        o = ot.iloc[idx].reset_index(drop=True)
        y, w = ipcw_outcome(o, outcome, t)
        diffs[bi] = brier(pred_a[idx], y, w) - brier(pred_b[idx], y, w)
    return diffs
