"""Temporal validation on the later-calendar holdout cohort (SAP section 7).

IPCW Brier scores, calibration-in-the-large / slope, decile calibration plots,
and bootstrap uncertainty for Brier differences (no claim of improvement from a
single point estimate). Outputs T10, F11.
"""
import os, sys
import numpy as np, pandas as pd, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hmsm.model import ModelSpec
from hmsm.predict import individual_risks, naive_to_params
from hmsm.validation import outcome_times, ipcw_outcome, brier, calibration_table, calibration_intercept_slope, bootstrap_brier_diff
from hmsm import plots
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

proto = yaml.safe_load(open("config/protocol.yaml"))
XC, WC = proto["covariates"]["transitions"], proto["covariates"]["encounter"]
cov = pd.read_csv("data/derived/analysis_covariates.csv.gz")
ev = pd.read_csv("data/derived/events.csv.gz")
ho = cov[cov.index_year >= proto["holdout_from_year"]].reset_index(drop=True)
eh = ev[ev.id.isin(ho.id)]
ot = outcome_times(eh).loc[ho.id].reset_index(drop=True)
X, W = ho[XC].to_numpy(), ho[WC].to_numpy()
rng = np.random.default_rng(11)

def avg_risks(param_list):
    acc = None
    for p in param_list:
        r = individual_risks(p, X, W, n_nodes=7)
        acc = r if acc is None else {k: acc[k] + r[k] for k in r}
    return {k: v / len(param_list) for k, v in acc.items()}

preds = {}
for name, spec in [("proposed", ModelSpec(4, 2)), ("hidden_ni", ModelSpec(4, 2, informative=False))]:
    z = np.load(f"results/fits/{name}.npz")
    dr = rng.multivariate_normal(z["theta"], z["cov"], 40)
    preds[name] = avg_risks([spec.unpack(t) for t in dr])
z = np.load("results/fits/naive.npz")
preds["naive"] = avg_risks([naive_to_params(t, 4, 2) for t in rng.multivariate_normal(z["theta"], z["cov"], 40)])

rows = []
for outcome in ["death", "anyhosp"]:
    for t in (30, 90, 365):
        y, w = ipcw_outcome(ot, outcome, t)
        for m, pr in preds.items():
            p = pr[f"{outcome}_{t}"]
            ci = calibration_intercept_slope(p, y, w)
            row = {"outcome": outcome, "horizon": t, "model": m, "observed_ipcw": np.sum(w * y) / np.sum(w),
                   "mean_pred": p.mean(), "brier": brier(p, y, w), **ci}
            if m != "proposed":
                dd = bootstrap_brier_diff(pr[f"{outcome}_{t}"], preds["proposed"][f"{outcome}_{t}"], ot, outcome, t, 200, rng)
                row["brier_minus_proposed"] = f"{dd.mean():.4f} ({np.percentile(dd,2.5):.4f}, {np.percentile(dd,97.5):.4f})"
            rows.append(row)
t10 = pd.DataFrame(rows).round(4)
t10.to_csv("results/tables/T10_temporal_validation.csv", index=False)
print(t10.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
for ax, (outcome, t, ttl) in zip(axes, [("death", 365, "365-day death"), ("anyhosp", 365, "365-day any hospitalisation")]):
    y, w = ipcw_outcome(ot, outcome, t)
    for m, pr in preds.items():
        ct = calibration_table(pr[f"{outcome}_{t}"], y, w, n_groups=5)
        plots.calibration(ax, ct.pred.to_numpy(), ct.obs.to_numpy(), ct.n_eff.to_numpy(), plots.LABELS[m], plots.PAL[m])
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1]); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Predicted risk (quintile mean)"); ax.set_ylabel("Observed (IPCW)"); ax.set_title(ttl)
axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
fig.suptitle(f"Temporal validation, held-out later-calendar cohort (N={len(ho)}, synthetic)", fontweight="bold")
fig.tight_layout(); fig.savefig("results/figures/F11_calibration.png", dpi=150); plt.close(fig)
